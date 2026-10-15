# Frozen public AG News guidance inputs

These files are the **public-only replay** of the fixed Qwen candidate pool and two public selection controls used in the October 2026 AG News pilot. They contain no client training records or private vote counts, and are not a DP release. The public-only replay matched the original pool bytes for this particular seed and pool; that observation does not generalize to other generation runs.

| File | Purpose | SHA-256 |
|---|---|---|
| `pool/candidates.jsonl` | 128 candidates, 32 per class | `7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f` |
| `random/synthetic.jsonl` | 64 public deterministic-random selections | `79773d11e68e9bd128b1852b6eb942debd23a3add32eb1f329685f43c65fcee3` |
| `nli/synthetic.jsonl` | 64 public NLI selections | `52810c8290a47533048d82a33bc92279c50d4df77f3628a1539031ba2905194a` |

Each JSONL has its original manifest in the same directory. `nli/quality.json` reports 59/64 auxiliary classifier label agreement; this is **not** downstream accuracy. The public NLI arm has not yet completed 50-round FL training.

Consume these exact files and validate their hashes. The original private-vote and matched-real selections are deliberately absent: they are Non-DP outputs derived from source client records, and must be regenerated or transferred through a restricted channel when the new server is available.
