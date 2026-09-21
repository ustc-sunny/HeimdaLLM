"""Small, dependency-light privacy leakage audit for released synthetic text.

The audit is intentionally a diagnostic, not a formal DP proof. It compares
member and non-member records against the released synthetic set using exact
normalised matches and TF-IDF character n-gram nearest-neighbour scores.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", required=True)
    parser.add_argument("--nonmembers", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    members = read_jsonl(args.members)
    nonmembers = read_jsonl(args.nonmembers)
    synthetic = read_jsonl(args.synthetic)
    released = [normalise(row["text"]) for row in synthetic]
    released_set = set(released)
    candidates = members + nonmembers
    labels = np.asarray([1] * len(members) + [0] * len(nonmembers))
    texts = [normalise(row["text"]) for row in candidates]

    exact = np.asarray([float(text in released_set) for text in texts])
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(texts + released)
    candidate_matrix = matrix[: len(texts)]
    released_matrix = matrix[len(texts) :]
    cosine = (candidate_matrix @ released_matrix.T).toarray()
    nearest = cosine.max(axis=1) if len(released) else np.zeros(len(texts))
    auc = float(roc_auc_score(labels, nearest)) if len(set(labels)) == 2 else None

    result = {
        "members": len(members),
        "nonmembers": len(nonmembers),
        "released": len(released),
        "exact_member_rate": float(exact[: len(members)].mean()) if members else 0.0,
        "exact_nonmember_rate": float(exact[len(members) :].mean()) if nonmembers else 0.0,
        "nearest_cosine_member_mean": float(nearest[: len(members)].mean()) if members else 0.0,
        "nearest_cosine_nonmember_mean": float(nearest[len(members) :].mean()) if nonmembers else 0.0,
        "membership_attack_auc": auc,
        "interpretation": "higher AUC or a member/non-member score gap indicates leakage risk; this audit does not replace an accountant",
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
