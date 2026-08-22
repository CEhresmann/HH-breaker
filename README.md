# HR-HH-Breaker

Personal job-search automation: tailor a resume + cover letter to a job posting and
(optionally) auto-apply, at hh.ru scale.

Built on top of [hr-breaker](https://github.com/btseytlin/hr-breaker) (MIT-licensed) -
an LLM-driven resume optimizer with a filter pipeline (ATS simulation, keyword
matching, hallucination detection) and HTML→PDF rendering via WeasyPrint. This repo
adds a batch layer on top: search hh.ru vacancies by trigger keyword, tailor a resume
per vacancy, generate a cover letter, and (optionally, behind an explicit `--live`
flag) submit the application through hh.ru's API.

## Status

Personal-use project, incrementally imported/built as separate PRs for review:

1. **README** (this PR)
2. **hr-breaker base** - vendored snapshot of the upstream optimizer/filter/rendering
   pipeline, unmodified
3. **hh.ru auto-apply pipeline** - `autoapply/` package + `cover_letter` agent + CLI
   (`hr-breaker autoapply auth|run`), built on top of (2)

Later PRs will land the same way - one reviewable unit at a time.

## Why a separate repo

Keeps this personal automation layer decoupled from upstream `hr-breaker`, so upstream
updates can be pulled in deliberately (as their own PR) instead of fighting a fork's
merge conflicts.

## Important caveat: hh.ru's apply flow and per-vacancy PDFs

hh.ru's "отклик" (apply) API references one of *your existing resumes already
published on hh.ru* by `resume_id` - there is no way to attach a different, freshly
tailored PDF to each individual application through their API. In `--live` mode, only
the **cover letter** is genuinely personalized per vacancy in the application hh.ru
receives. The tailored PDF is still generated and saved locally per vacancy (useful
for your own records, or to send manually if an employer asks for a resume by email
outside of hh.ru's own flow) - it is just not attached to the automated application
itself. See `src/hr_breaker/autoapply/pipeline.py`'s module docstring for detail.

**LinkedIn is intentionally not supported here.** LinkedIn has no legitimate
applicant-facing "apply" API for third parties, and third-party application
automation sits in a ToS gray/violation zone enforced by automated detection. This
project only automates against hh.ru's official, documented API.

## Setup

```bash
uv sync
cp .env.example .env   # set GEMINI_API_KEY (or another LiteLLM-supported provider)
uv run hr-breaker serve            # web UI, one resume <-> one job posting
```

For the auto-apply pipeline, you additionally need an hh.ru API application
(register at https://dev.hh.ru/) and a one-time OAuth token:

```bash
uv run hr-breaker autoapply auth --client-id X --client-secret Y --redirect-uri Z
uv run hr-breaker autoapply run --profile my-profile -t python -t "data engineer"          # dry run
uv run hr-breaker autoapply run --profile my-profile -t python --live --max-apply 5        # applies for real
```

See `CLAUDE.md` for full architecture, filter pipeline, and configuration reference.

## Safety defaults

- Auto-apply is dry-run unless you pass `--live`.
- `--max-apply` caps how many real applications get sent in a single run.
- A local SQLite store (`.cache/autoapply.sqlite3`) dedupes vacancies across runs so
  the same posting is never processed (or applied to) twice.

## License

MIT (inherited from upstream hr-breaker).
