#!/usr/bin/env python3
"""Small, training-free AG News private-vote generation pilot (Non-DP).

This is a one-step Aug-PE-inspired feasibility pilot, not the published Aug-PE
algorithm or a DP release. Private records are used only for local candidate
voting. No private text is included in generator prompts or public metadata.
"""

import argparse
import gc
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from non_dp_client_synthetic import load_staged_clients, normalize_generated_text


LABELS = {
    "1": ("World", ["election", "diplomacy", "peace talks", "natural disaster", "international aid", "border policy", "regional government", "migration policy"]),
    "2": ("Sports", ["football", "basketball", "tennis", "athletics", "baseball", "cycling", "swimming", "motorsport"]),
    "3": ("Business", ["corporate earnings", "retail sales", "bank lending", "trade tariffs", "commodity prices", "manufacturing output", "employment figures", "stock markets"]),
    "4": ("Science and Technology", ["space", "computing", "biotechnology", "telecommunications", "research", "software", "robotics", "electronics"]),
}

CATEGORY_RULES = {
    "1": "Focus on governments, countries and international events. Avoid technology product news.",
    "2": "Focus on a sporting competition, athlete or team result.",
    "3": "Focus on money, sales, profits, prices, jobs or markets. Avoid software, gadgets, data centers and scientific research.",
    "4": "Focus on scientific findings, engineering or technology products. Avoid corporate earnings and stock prices.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_preflight(index, min_free_mib):
    import subprocess
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(value.strip()) for value in line.split(",")) for line in raw.splitlines()]
    selected = next((row for row in rows if row[0] == index), None)
    if selected is None or selected[1] < min_free_mib or selected[2] > 20:
        raise RuntimeError("GPU%d lacks guarded capacity: %s" % (index, selected))
    return {"index": selected[0], "free_mib": selected[1], "utilization_percent": selected[2]}


def clean(text):
    text = re.sub(r"(?i)^\s*(headline|article|news article|category)\s*:\s*", "", text.strip())
    return re.sub(r"\s+", " ", text).strip(' \"\'')


def generate_candidates(args, task_tokenizer, forbidden):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(0.30, args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.generator_model, torch_dtype=torch.float16,
        attn_implementation="eager", local_files_only=True).to("cuda:%d" % args.gpu)
    model.eval()
    result = {label: [] for label in LABELS}
    seen = set(forbidden)
    for label, (name, topics) in LABELS.items():
        attempts = 0
        while len(result[label]) < args.candidates_per_label:
            attempts += 1
            if attempts > args.candidates_per_label * 8:
                raise RuntimeError("too few usable generated candidates for label %s" % label)
            topic = topics[(attempts - 1) % len(topics)]
            prompt = (
                "Write one original, plausible English news brief in the %s category, "
                "on the broad topic of %s. Use a headline followed by one short "
                "sentence. Total length 30 to 42 words. Output only the news brief. "
                "Do not copy an existing article or include a category label. %s" %
                (name, topic, CATEGORY_RULES[label]))
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            torch.manual_seed(args.seed * 100000 + int(label) * 1000 + attempts)
            with torch.inference_mode():
                generated = model.generate(
                    encoded, attention_mask=torch.ones_like(encoded),
                    max_new_tokens=90, do_sample=True, temperature=0.85,
                    top_p=0.92, pad_token_id=tokenizer.eos_token_id)
            text = clean(tokenizer.decode(generated[0, encoded.shape[-1]:], skip_special_tokens=True))
            key = normalize_generated_text(text)
            length = len(task_tokenizer.encode(text, add_special_tokens=True))
            if key in seen or len(text.split()) < 24 or length > 64:
                continue
            seen.add(key)
            result[label].append(text)
            print("Generated label %s: %d/%d (attempt %d)" % (
                label, len(result[label]), args.candidates_per_label, attempts), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def embed(model, tokenizer, texts, batch_size=24):
    import torch
    vectors = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=1))
    return torch.cat(vectors, dim=0)


def select_candidates(args, clients, client_ids, candidates):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.embedding_model, local_files_only=True)
    model.eval()
    selected = []
    vote_summary = {}
    for label in LABELS:
        texts = candidates[label]
        candidate_vectors = embed(model, tokenizer, texts)
        used = set()
        for client_id in client_ids:
            private = [row["text"] for row in clients[client_id] if row["label"] == label]
            if not private:
                raise RuntimeError("client %d has no label %s records" % (client_id, label))
            private_vectors = embed(model, tokenizer, private)
            winners = torch.argmax(private_vectors @ candidate_vectors.T, dim=1).tolist()
            votes = Counter(winners)
            ranked = sorted(range(len(texts)), key=lambda i: (-votes[i], i))
            chosen = [i for i in ranked if i not in used][:args.selected_per_client_label]
            if len(chosen) != args.selected_per_client_label:
                raise RuntimeError("candidate pool too small for disjoint selections")
            used.update(chosen)
            selected.extend({"client_id": client_id, "label": label, "text": texts[i]}
                            for i in chosen)
            vote_summary["%d:%s" % (client_id, label)] = {
                "private_records": len(private), "candidate_count": len(texts),
                "selected_votes": [votes[i] for i in chosen]}
    if len(selected) != len(client_ids) * len(LABELS) * args.selected_per_client_label:
        raise AssertionError("selection count mismatch")
    if len({normalize_generated_text(row["text"]) for row in selected}) != len(selected):
        raise AssertionError("selected text duplicates")
    return selected, vote_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-json-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--task-tokenizer", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--client-ids", nargs="+", type=int, default=[1, 21])
    parser.add_argument("--candidates-per-label", type=int, default=32)
    parser.add_argument("--selected-per-client-label", type=int, default=8)
    parser.add_argument("--min-free-mib", type=int, default=14000)
    args = parser.parse_args()
    if args.candidates_per_label < len(args.client_ids) * args.selected_per_client_label:
        parser.error("candidate pool cannot cover disjoint client selections")
    if args.output_dir.exists():
        parser.error("output directory exists; refusing to overwrite private results")
    preflight = gpu_preflight(args.gpu, args.min_free_mib)
    from transformers import AutoTokenizer
    task_tokenizer = AutoTokenizer.from_pretrained(args.task_tokenizer, local_files_only=True)
    clients, client_ids, label_vocab, _, descriptor = load_staged_clients(
        args.client_json_dir, args.client_ids)
    if set(label_vocab) != set(LABELS):
        raise ValueError("source AG News label vocabulary differs from fixed pilot prompts")
    forbidden = {normalize_generated_text(row["text"]) for rows in clients.values() for row in rows}
    candidates = generate_candidates(args, task_tokenizer, forbidden)
    selected, votes = select_candidates(args, clients, client_ids, candidates)
    args.output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    candidate_path = args.output_dir / "candidates.jsonl"
    with candidate_path.open("w") as handle:
        for label, texts in candidates.items():
            for text in texts:
                handle.write(json.dumps({"label": label, "text": text}, ensure_ascii=False) + "\n")
    os.chmod(candidate_path, 0o600)
    path = args.output_dir / "synthetic.jsonl"
    with path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)
    manifest = {
        "schema_version": 1, "status": "complete", "privacy": "Non-DP",
        "method": "one_step_private_embedding_vote_pilot_not_full_aug_pe",
        "seed": args.seed, "client_ids": client_ids, "generator_model": args.generator_model,
        "embedding_model": args.embedding_model, "candidates_per_label": args.candidates_per_label,
        "selected_per_client_label": args.selected_per_client_label,
        "selected_records": len(selected), "selected_sha256": sha256(path),
        "candidate_records": sum(len(texts) for texts in candidates.values()),
        "candidate_sha256": sha256(candidate_path),
        "selected_task_token_lengths": {
            "min": min(len(task_tokenizer.encode(row["text"], add_special_tokens=True)) for row in selected),
            "max": max(len(task_tokenizer.encode(row["text"], add_special_tokens=True)) for row in selected)},
        "source_staging_manifest_sha256": descriptor["staging_manifest_sha256"],
        "gpu_preflight": preflight, "vote_summary": votes,
        "generator_gpu_memory_fraction_cap": 0.30,
        "no_private_text_in_generator_prompts": True,
        "official_test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "selected_records": len(selected),
                      "selected_sha256": sha256(path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
