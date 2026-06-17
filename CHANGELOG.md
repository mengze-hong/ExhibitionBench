# ExhibitionBench — CHANGELOG

## 2026-06-12 21:30 — Full sanitization + results included
- Renamed all env vars to generic `LLM_API_*` and `LLM_OPENWEIGHT_*` for public release
- Removed all references to internal hostnames, org names, and proxy services from code and comments
- Copied all 337 result JSONs (stripped run_meta sensitive fields) + 5 analysis subdirs (52 files)
- Verified: zero sensitive patterns across all .py / .md / .json files

## 2026-06-12 21:00 — Initial public release
- Created `github_repo_museum/` repository structure
- Sanitized all internal API endpoints (replaced with environment-variable placeholders)
- Copied and organized: evaluation/, baselines/, analysis/, system/, scripts/, data/, results_summary/
- Added README files for each subfolder
- Added .env.example credential template
- Added .gitignore
