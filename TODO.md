## TODO: Improvements and Fixes

### P0 — Critical fixes (blockers)
- [x] Restore or implement `OpenCodeUsageAnalyzer`
  - [x] Add analyzer that discovers projects under `~/.local/share/opencode/project/`, parses session info and message files, and builds `SessionSummary` objects
  - [x] Populate `SessionSummary.model` as `"<provider>/<modelID>"` when available; fallback to provider/model defaults if missing
  - [x] Set `SessionSummary.provider` (e.g., `groq`, `anthropic`, `openai`) when present
  - [x] Compute per-session cost and set `summary.cost` using the shared pricing helper
  - [x] Wire it up in `main.py`; feature works when not running `--claude-only`
  - [x] Add `print_summary` for OpenCode (provider/model/project breakdowns)
  - [x] Definition of done: `uv run main.py` runs without NameError and shows OpenCode stats (sessions, tokens, provider/model breakdowns) when data exists

- [x] Fix cost reporting in unified summary
  - [x] Ensure OpenCode sessions have `summary.cost` set by the analyzer
  - [x] Verify `print_unified_summary` uses actual OpenCode costs (no implicit zeros)
  - [x] Definition of done: unified summary shows a non-zero OpenCode cost when OpenCode usage exists

- [x] Centralize model ID normalization for pricing lookups
  - [x] Add a helper (e.g., `normalize_model_id_for_pricing(model: str) -> str`) that strips prefixes (`anthropic/`, `openrouter/`, `openai/`, `groq/`) and nested prefixes (`openrouter/openai/gpt-4o-mini` → `gpt-4o-mini`)
  - [x] Use it in: Claude cost calc, OpenCode cost calc, comparison summary, token-cost breakdowns
  - [x] Definition of done: pricing lookups succeed for prefixed and nested-prefixed model IDs

### P1 — Pricing, caching, and resilience
- [x] Pricing cache TTL and bypass flag
  - [x] Add TTL (e.g., 1 hour) to `ModelsDotDev.get_pricing()` in-memory cache
  - [x] Support `--no-cache` to bypass the cache and refresh once
  - [ ] Optional: write-through disk cache at `~/.cache/claude-usage-analyzer/pricing.json` with timestamp
  - [x] Definition of done: subsequent runs within TTL do not hit the network; `--no-cache` forces refresh

- [x] Network resilience for models.dev
  - [x] Add retries with backoff (e.g., 3 attempts, 250ms * 2^n) and a descriptive User-Agent header
  - [x] Validate schema defensively (missing keys, types) and fail gracefully with a clear message when `--quiet` is not set
  - [x] Definition of done: transient network errors do not crash the tool; clear messages appear when appropriate

### P1 — Output and UX
- [x] Add `--json` output mode
  - [x] Emit a machine-readable unified summary to stdout when `--json` is passed
  - [x] Minimal schema (example):
    ```json
    {
      "sessions": <int>,
      "messages": <int>,
      "tokens": {"input": <int>, "output": <int>, "cache_create": <int>, "cache_read": <int>},
      "cost": {"total": <float>, "claude": <float>, "opencode": <float>},
      "by_model": [{"model": "provider/model", "tokens": <int>, "cost": <float>}],
      "by_project": [{"project": "name", "tokens": <int>}],
      "recent": [{"ts": "YYYY-MM-DDTHH:MM:SS", "project": "name", "tokens": <int>, "source": "claude|opencode"}]
    }
    ```
  - [x] Definition of done: `uv run main.py --json` prints only JSON; no extra text

- [ ] Project name display controls
  - [ ] Already switched to `os.path.basename`; add `--full-paths` flag to display full paths (with `$HOME` collapsed to `~`)
  - [ ] Definition of done: `--full-paths` shows expanded paths; default remains basename

- [ ] Quiet mode coverage
  - [ ] Audit all prints and warnings; ensure they respect `--quiet` (except fatal errors)
  - [ ] Definition of done: no non-essential output in quiet mode

### P2 — Code structure and maintainability
- [ ] Modularize the codebase
  - [ ] `types.py` — `TokenUsage`, `SessionSummary`, helpers (e.g., model normalization)
  - [ ] `models_api.py` — `ModelsDotDev` (pricing, cache, retries)
  - [ ] `claude.py` — `ClaudeUsageAnalyzer`
  - [ ] `opencode.py` — `OpenCodeUsageAnalyzer`
  - [ ] `cli.py` — argparse, main entrypoint
  - [ ] Update imports accordingly
  - [ ] Definition of done: `uv run main.py` works with new module layout

- [ ] Dev dependencies organization
  - [ ] Move `mypy`, `types-requests` under `[project.optional-dependencies.dev]`
  - [ ] Update README with `uv pip install -e .[dev]` or `uvx` equivalents
  - [ ] Definition of done: prod install pulls only runtime deps; dev install adds tooling

### P2 — Tests and CI
- [ ] Add pytest with fixtures
  - [ ] Claude `.jsonl` fixtures: valid, malformed lines, with/without cache tokens, mixed models
  - [ ] OpenCode fixtures: sample project layout with multiple providers/models, tokens, timestamps
  - [ ] Pricing normalization tests: prefixed and nested-prefixed model IDs
  - [ ] Comparison logic tests: current vs alternative model cost deltas
  - [ ] Definition of done: `uv run pytest` passes locally

- [ ] Type checking and formatting in CI
  - [ ] Add GitHub Actions workflow: run mypy, pytest on push/PR
  - [ ] Consider `ruff` for lint/format + autofix (optional)
  - [ ] Definition of done: CI badge green; PRs run checks

### P2 — Documentation
- [ ] README updates
  - [ ] Align OpenCode cost behavior (currently using models.dev pricing, not “direct cost”) or implement direct-cost ingestion and document it
  - [ ] Document `--json`, `--list-providers`, `--list-models`, `--compare-model`, `--full-paths`
  - [ ] Add realistic examples and sample outputs

- [ ] `CLAUDE.md` updates
  - [ ] Match pricing cache behavior (TTL, refresh)
  - [ ] Update architecture sections after modularization

- [ ] Packaging metadata and license
  - [ ] Set a real `description` in `pyproject.toml`
  - [ ] Add `LICENSE` file and section in README

### P3 — Nice-to-haves
- [ ] Per-token-type costs in unified summary (like Claude summary)
  - [ ] When pricing available, show input/output/cache costs next to token totals

- [ ] Progress indicators
  - [ ] For large scans, print light progress (or a single-line spinner) unless `--quiet`

- [ ] Logging abstraction
  - [ ] Replace ad-hoc prints with a minimal logger respecting verbosity levels

---

## Implementation notes
- Prefer a single source of truth for pricing lookups and model normalization to avoid drift.
- Keep error handling non-intrusive during scans; aggregate and show a concise error summary at the end in verbose mode.
- Ensure new flags are added to `--help` with clear, concise descriptions.

## Quick verification checklist
- [ ] `uv run main.py` works (Claude-only environment)
- [ ] `uv run main.py --opencode-only` works when OpenCode data present
- [ ] `uv run main.py --json` emits valid JSON
- [ ] `uv run main.py --compare-model anthropic/claude-3-5-haiku-20241022` prints comparison without errors
- [ ] `uv run mypy` passes
- [ ] `uv run pytest` passes
- [ ] CI runs mypy + pytest on PRs