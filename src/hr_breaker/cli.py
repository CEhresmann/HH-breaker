"""CLI interface for HR-Breaker."""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import click

from hr_breaker.agents import extract_name, parse_job_posting
from hr_breaker.config import get_settings
from hr_breaker.models import (
    GeneratedPDF,
    JobPosting,
    ResumeSource,
    SUPPORTED_LANGUAGES,
    get_language_safe,
    resolve_target_language,
)
from hr_breaker.models.profile import document_needs_extraction, get_document_extraction
from hr_breaker.orchestration import optimize_for_job
from hr_breaker.services import PDFStorage
from hr_breaker.services.pdf_storage import generate_run_id
from hr_breaker.services.pdf_parser import load_resume_content
from hr_breaker.utils.optimization_telemetry import IterationETA, telemetry_reporter


# ---------------------------------------------------------------------------
# Live progress reporting - prints directly to stdout via click.echo, independent
# of the logging module/LOG_LEVEL so it's always visible.
# ---------------------------------------------------------------------------

_TICK_INTERVAL_S = 5.0
_LINE_WIDTH = 100  # padding to overwrite a longer previous line when using \r


class _LiveProgress:
    """Prints a self-updating line while an LLM call is in flight, driven by
    optimization_telemetry's "start"/"done" reporter events."""

    def __init__(self):
        self._inflight: dict[str, float] = {}
        self._task: asyncio.Task | None = None

    def _report(self, payload: dict) -> None:
        component = payload.get("component", "?")
        if payload.get("event") == "start":
            self._inflight[component] = time.monotonic()
        elif payload.get("event") == "done":
            start = self._inflight.pop(component, None)
            elapsed = time.monotonic() - start if start is not None else 0.0
            click.echo(f"  ✓ {component} done ({elapsed:.0f}s)".ljust(_LINE_WIDTH))

    async def _tick_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_TICK_INTERVAL_S)
                if not self._inflight:
                    continue
                now = time.monotonic()
                if len(self._inflight) == 1:
                    (component, start), = self._inflight.items()
                    line = f"  … {component} running ({now - start:.0f}s)"
                    click.echo("\r" + line.ljust(_LINE_WIDTH), nl=False)
                else:
                    for component, start in self._inflight.items():
                        click.echo(f"  … {component} running ({now - start:.0f}s)")
        except asyncio.CancelledError:
            pass

    async def __aenter__(self) -> "_LiveProgress":
        self._cm = telemetry_reporter(self._report)
        self._cm.__enter__()
        self._task = asyncio.create_task(self._tick_loop())
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._cm.__exit__(*exc_info)
        if self._inflight:
            click.echo("".ljust(_LINE_WIDTH))  # clear any leftover \r-overwritten line


@asynccontextmanager
async def live_progress():
    """Set up live per-LLM-call progress reporting for the duration of the block."""
    progress = _LiveProgress()
    async with progress:
        yield progress


# ---------------------------------------------------------------------------
# Helpers for extraction state display
# ---------------------------------------------------------------------------

def _format_extraction_state(doc) -> str:
    status = str(doc.metadata.get("extraction_status") or "").lower()
    if status == "empty":
        return "empty extraction"
    if status == "failed":
        return "failed extraction"
    if get_document_extraction(doc) is not None:
        return "extracted"
    return "no extraction"


def _print_extraction_result(doc) -> str:
    status = str(doc.metadata.get("extraction_status") or "").lower()
    if status == "done":
        click.echo(" ok")
        return "done"
    if status == "empty":
        click.echo(" empty")
        return "empty"
    click.echo(f" unexpected status: {status or 'missing'}")
    return "unexpected"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """HR-Breaker: Optimize resumes for job postings."""
    pass


OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# profile subcommand group
# ---------------------------------------------------------------------------

@cli.group()
def profile():
    """Manage profile archives."""
    pass


@profile.command("list")
def profile_list():
    """List all profiles."""
    from hr_breaker.services.profile_store import ProfileStore

    store = ProfileStore()
    profiles = store.list_profiles()
    if not profiles:
        click.echo("No profiles found. Create one with: hr-breaker profile create <name>")
        return
    for p in profiles:
        name_part = f" ({p.full_name})" if p.full_name else ""
        doc_count = len(store.list_documents(p.id))
        click.echo(f"  {p.id:30s}  {p.display_name}{name_part}  [{doc_count} doc(s)]")


@profile.command("create")
@click.argument("name")
@click.option("--first-name", default=None, help="Candidate first name")
@click.option("--last-name", default=None, help="Candidate last name")
@click.option("--instructions", "-i", default=None, help="Standing instructions for the optimizer")
def profile_create(name: str, first_name: str | None, last_name: str | None, instructions: str | None):
    """Create a new profile archive.

    NAME: Display name for the profile (e.g. "John Doe")
    """
    from hr_breaker.services.profile_store import ProfileStore

    store = ProfileStore()
    p = store.create_profile(name, first_name=first_name, last_name=last_name, instructions=instructions)
    click.echo(f"Created profile: {p.id}  ({p.display_name})")


@profile.command("show")
@click.argument("profile_id")
def profile_show(profile_id: str):
    """Show profile details and its documents."""
    from hr_breaker.services.profile_store import ProfileStore

    store = ProfileStore()
    p = store.get_profile(profile_id)
    if p is None:
        raise click.ClickException(f"Profile not found: {profile_id}")

    click.echo(f"ID:           {p.id}")
    click.echo(f"Name:         {p.display_name}")
    if p.full_name:
        click.echo(f"Full name:    {p.full_name}")
    if p.instructions:
        click.echo(f"Instructions: {p.instructions}")
    click.echo(f"Updated:      {p.updated_at.strftime('%Y-%m-%d %H:%M')}")

    docs = store.list_documents(p.id)
    click.echo(f"\nDocuments ({len(docs)}):")
    if not docs:
        click.echo("  (none — add with: hr-breaker profile add <profile-id> <file>)")
    for doc in docs:
        extraction_state = _format_extraction_state(doc)
        incl = "+" if doc.included_by_default else "-"
        click.echo(f"  [{incl}] {doc.id[:12]}  {doc.title:40s}  [{doc.kind}, {extraction_state}]")


@profile.command("add")
@click.argument("profile_id")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--extract", is_flag=True, help="Run fact extraction immediately after adding")
@click.option("--exclude", is_flag=True, help="Add as excluded by default")
def profile_add(profile_id: str, files: tuple[Path, ...], extract: bool, exclude: bool):
    """Add one or more files to a profile.

    PROFILE_ID: Target profile ID (see: hr-breaker profile list)
    FILES:      One or more file paths to add
    """
    from hr_breaker.services.profile_store import ProfileStore

    store = ProfileStore()
    p = store.get_profile(profile_id)
    if p is None:
        raise click.ClickException(f"Profile not found: {profile_id}")

    added_ids: list[str] = []
    for file_path in files:
        click.echo(f"  Adding {file_path.name}...", nl=False)
        try:
            doc = store.add_upload(
                profile_id,
                filename=file_path.name,
                data=file_path.read_bytes(),
                included_by_default=not exclude,
            )
            added_ids.append(doc.id)
            click.echo(f" ok ({doc.id[:12]})")
        except Exception as exc:
            click.echo(f" failed: {exc}")

    if extract and added_ids:
        click.echo("Extracting facts...")

        async def run_extract():
            for doc_id in added_ids:
                click.echo(f"  {doc_id[:12]}...", nl=False)
                try:
                    updated = await store.extract_document_content(profile_id, doc_id)
                    if updated is None:
                        click.echo(" missing")
                        continue
                    _print_extraction_result(updated)
                except Exception as exc:
                    click.echo(f" failed: {exc}")

        asyncio.run(run_extract())
    elif added_ids and not extract:
        click.echo(f"Tip: run 'hr-breaker backfill --profile {profile_id}' to extract facts.")


# ---------------------------------------------------------------------------
# optimize command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("resume_path", required=False, type=click.Path(path_type=Path), default=None)
@click.argument("job_input", required=False, default=None)
@click.option(
    "--profile", "-p", "profile_id",
    default=None,
    help="Use a profile archive instead of a resume file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    envvar="HR_BREAKER_OUTPUT",
)
@click.option(
    "--max-iterations", "-n", type=int, default=None, envvar="HR_BREAKER_MAX_ITERATIONS"
)
@click.option(
    "--debug/--no-debug",
    "-d/-D",
    default=True,
    help="Save all iterations as PDFs to output/debug/ (default: on)",
    envvar="HR_BREAKER_DEBUG",
)
@click.option(
    "--seq",
    "-s",
    is_flag=True,
    help="Run filters sequentially (default: parallel)",
    envvar="HR_BREAKER_SEQ",
)
@click.option(
    "--no-shame",
    is_flag=True,
    help="Lenient mode: allow aggressive content stretching",
    envvar="HR_BREAKER_NO_SHAME",
)
@click.option(
    "--lang",
    "-l",
    type=click.Choice(
        ["from_job", "from_resume"] + [lang.code for lang in SUPPORTED_LANGUAGES],
        case_sensitive=False,
    ),
    default=None,
    help="Language mode: from_job (detect from job), from_resume (detect from resume), or ISO code.",
)
@click.option(
    "--instructions",
    "-i",
    type=str,
    default=None,
    help="Instructions for the optimizer (extra experience, emphasis areas)",
)
@click.option(
    "--docs",
    default=None,
    help="Comma-separated document IDs to include (profile mode only; default: all included_by_default)",
)
def optimize(
    resume_path: Path | None,
    job_input: str | None,
    profile_id: str | None,
    output: Path | None,
    max_iterations: int | None,
    debug: bool,
    seq: bool,
    no_shame: bool,
    lang: str | None,
    instructions: str | None,
    docs: str | None,
):
    """Optimize a resume for a job posting.

    Direct upload mode:

        hr-breaker optimize resume.txt https://example.com/job

    Profile archive mode:

        hr-breaker optimize --profile <id> https://example.com/job

    JOB_INPUT: URL or path to a file containing the job description.
    """
    if profile_id is None and resume_path is None:
        raise click.UsageError(
            "Provide either RESUME_PATH or --profile <id>.\n"
            "  Direct:  hr-breaker optimize resume.txt <job>\n"
            "  Profile: hr-breaker optimize --profile <id> <job>"
        )
    if profile_id is not None and resume_path is not None:
        raise click.UsageError("Cannot use both RESUME_PATH and --profile at the same time.")

    # Handle: hr-breaker optimize --profile id <job>  (job ends up in resume_path slot)
    effective_job_input = job_input
    if profile_id is not None and job_input is None and resume_path is not None:
        effective_job_input = str(resume_path)
        resume_path = None

    if effective_job_input is None:
        raise click.UsageError("JOB_INPUT is required (URL or path to job description).")

    if resume_path is not None and not resume_path.exists():
        raise click.ClickException(f"Resume file not found: {resume_path}")

    job_text = _get_job_text(effective_job_input)

    pdf_storage = PDFStorage()
    run_id = generate_run_id()
    debug_dir: Path | None = None

    effective_max_iterations = max_iterations if max_iterations is not None else get_settings().max_iterations
    eta_tracker = IterationETA(effective_max_iterations)

    def on_iteration(i, optimized, validation):
        status = "PASS" if validation.passed else "FAIL"
        scores = ", ".join(
            f"{r.filter_name}:{r.score:.2f}/{r.threshold:.2f}"
            for r in validation.results
            if not r.skipped
        )
        elapsed, eta = eta_tracker.tick(i)
        click.echo(f"  Iteration {i + 1}/{effective_max_iterations}: {status} [{scores}] ({elapsed:.0f}s, ETA {eta:.0f}s)")

        if debug and debug_dir:
            if optimized.html:
                debug_html = debug_dir / f"iteration_{i + 1}.html"
                debug_html.write_text(optimized.html, encoding="utf-8")
            elif optimized.data:
                debug_json = debug_dir / f"iteration_{i + 1}.json"
                debug_json.write_text(optimized.data.model_dump_json(indent=2), encoding="utf-8")
            if optimized.pdf_bytes:
                debug_pdf = debug_dir / f"iteration_{i + 1}.pdf"
                debug_pdf.write_bytes(optimized.pdf_bytes)
                click.echo(f"    Debug: saved {debug_pdf}")
            else:
                click.echo("    Debug: no PDF (render failed)")

    settings = get_settings()
    lang_mode = lang or settings.default_language

    async def run_optimization():
        nonlocal debug_dir

        pre_parsed_job = None
        if profile_id is not None:
            source, pre_parsed_job = await _build_profile_source(profile_id, job_text, docs_filter=docs)
            first_name = source.first_name
            last_name = source.last_name
            resume_lang_code = source.language_code or "en"
            name_str = f"{first_name or ''} {last_name or ''}".strip()
            click.echo(f"Profile: {profile_id}" + (f"  ({name_str})" if name_str else ""))
        else:
            resume_content = load_resume_content(resume_path)
            first_name, last_name, resume_lang_code = await extract_name(resume_content)
            click.echo(f"Resume: {first_name or 'Unknown'} {last_name or ''} (lang: {resume_lang_code})")
            source = ResumeSource(
                content=resume_content,
                first_name=first_name,
                last_name=last_name,
                language_code=resume_lang_code,
            )

        job = pre_parsed_job or await parse_job_posting(job_text)
        click.echo(f"Job: {job.title} at {job.company} (lang: {job.language_code})")

        target_language = resolve_target_language(lang_mode, job.language_code, resume_lang_code)
        source_lang = get_language_safe(resume_lang_code)

        if debug:
            debug_dir = pdf_storage.generate_debug_dir(job.company, job.title, run_id=run_id)

        mode = "sequential" if seq else "parallel"
        shame_mode = " [no-shame]" if no_shame else ""
        click.echo(f"Optimizing (mode: {mode}{shame_mode}, target: {target_language.english_name})...")

        async with live_progress():
            optimized, validation, _ = await optimize_for_job(
                source,
                max_iterations=max_iterations,
                on_iteration=on_iteration,
                job=job,
                parallel=not seq,
                no_shame=no_shame,
                user_instructions=instructions,
                language=target_language,
                source_language=source_lang,
            )
        return first_name, last_name, source, optimized, validation, job, target_language

    first_name, last_name, source, optimized, validation, job, target_language = asyncio.run(
        run_optimization()
    )
    lang_code = target_language.code

    if not validation.passed:
        click.echo("Warning: Not all filters passed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        output = (
            OUTPUT_DIR
            / pdf_storage.generate_path(
                first_name,
                last_name,
                job.company,
                job.title,
                lang_code=lang_code,
                run_id=run_id,
            ).name
        )

    if not optimized.pdf_bytes:
        raise click.ClickException("No PDF generated (render failed)")
    output.write_bytes(optimized.pdf_bytes)

    pdf_record = GeneratedPDF(
        path=output,
        source_checksum=source.checksum,
        company=job.company,
        job_title=job.title,
        first_name=first_name,
        last_name=last_name,
    )
    pdf_storage.save_record(pdf_record)
    click.echo(f"PDF saved: {output}")


async def _build_profile_source(
    profile_id: str,
    job_text: str,
    *,
    docs_filter: str | None,
) -> "tuple[ResumeSource, JobPosting]":
    """Rank profile documents against the job and return (ResumeSource, JobPosting)."""
    from hr_breaker.services.profile_store import ProfileStore
    from hr_breaker.services.profile_retrieval import rank_profile_documents, synthesize_profile_resume_source

    store = ProfileStore()
    p = store.get_profile(profile_id)
    if p is None:
        raise click.ClickException(f"Profile not found: {profile_id}")

    all_docs = store.list_documents(profile_id)
    if not all_docs:
        raise click.ClickException(
            f"Profile '{profile_id}' has no documents. "
            f"Add some with: hr-breaker profile add {profile_id} <file>"
        )

    if docs_filter:
        wanted = {d.strip() for d in docs_filter.split(",")}
        selected = [d for d in all_docs if d.id in wanted or d.id[:12] in wanted]
        if not selected:
            raise click.ClickException(f"No documents matched --docs filter: {docs_filter}")
    else:
        selected = [d for d in all_docs if d.included_by_default] or all_docs

    missing_extraction = [d.title for d in selected if document_needs_extraction(d)]
    if missing_extraction:
        click.echo(
            f"Warning: {len(missing_extraction)} document(s) have no extracted facts "
            f"and will be used as raw text: {', '.join(missing_extraction)}\n"
            f"  Run 'hr-breaker backfill --profile {profile_id}' to fix this."
        )

    job = await parse_job_posting(job_text)
    ranked = await rank_profile_documents(selected, job)
    return synthesize_profile_resume_source(p, selected, ranked), job


# ---------------------------------------------------------------------------
# backfill command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--profile", "-p", "profile_id", default=None, help="Profile ID to backfill (default: all)")
@click.option("--force", is_flag=True, help="Re-extract even if extraction already exists")
def backfill(profile_id: str | None, force: bool):
    """Extract facts from profile documents that are missing extraction data."""
    from hr_breaker.services.profile_store import ProfileStore

    store = ProfileStore()
    if profile_id:
        p = store.get_profile(profile_id)
        if p is None:
            raise click.ClickException(f"Profile not found: {profile_id}")
        profiles = [p]
    else:
        profiles = store.list_profiles()

    if not profiles:
        click.echo("No profiles found.")
        return

    total = done = empty = failed = 0

    async def run():
        nonlocal total, done, empty, failed
        for p in profiles:
            docs = store.list_documents(p.id)
            pending = [d for d in docs if force or document_needs_extraction(d)]
            click.echo(f"Profile '{p.display_name}': {len(pending)}/{len(docs)} document(s) to process")
            for doc in pending:
                total += 1
                click.echo(f"  {doc.title}...", nl=False)
                try:
                    updated = await store.extract_document_content(p.id, doc.id)
                    if updated is None:
                        failed += 1
                        click.echo(" failed: document disappeared")
                        continue
                    result = _print_extraction_result(updated)
                    if result == "done":
                        done += 1
                    elif result == "empty":
                        empty += 1
                    else:
                        failed += 1
                except Exception as exc:
                    failed += 1
                    click.echo(f" failed: {exc}")

    asyncio.run(run())
    click.echo(f"\nDone: {done} extracted, {empty} empty, {failed} failed, {total} total")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

@cli.command("list")
def list_history():
    """List generated PDFs."""
    pdf_storage = PDFStorage()
    pdfs = pdf_storage.list_all()

    if not pdfs:
        click.echo("No PDFs generated yet")
        return

    for pdf in pdfs:
        exists = "+" if pdf.path.exists() else "-"
        click.echo(
            f"[{exists}] {pdf.path.name} - {pdf.job_title} @ {pdf.company} "
            f"({pdf.timestamp.strftime('%Y-%m-%d %H:%M')})"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_job_text(job_input: str) -> str:
    """Get job text from a file path, or treat the input as raw text."""
    path = Path(job_input)
    if path.exists():
        return path.read_text(encoding="utf-8")

    if job_input.startswith(("http://", "https://")):
        raise click.ClickException(
            "URL scraping is not supported - paste the job description as text or a file instead."
        )

    return job_input


# ---------------------------------------------------------------------------
# autoapply command group (hh.ru)
# ---------------------------------------------------------------------------

@cli.group()
def autoapply():
    """Search hh.ru vacancies by trigger keyword and auto-tailor resume + cover letter.

    Two steps:

        hr-breaker autoapply browser-login   # one-time: log into hh.ru in a browser window
        hr-breaker autoapply run ...          # search + tailor (+ apply with --live)
    """


@autoapply.command()
@click.option("--client-id", required=True, envvar="HH_CLIENT_ID", help="hh.ru app client_id (from dev.hh.ru)")
@click.option("--client-secret", required=True, envvar="HH_CLIENT_SECRET", help="hh.ru app client_secret")
@click.option("--redirect-uri", required=True, envvar="HH_REDIRECT_URI", help="Must match the app's registered redirect URI")
def auth(client_id: str, client_secret: str, redirect_uri: str):
    """One-time OAuth flow: opens the consent URL, then exchanges the code you paste back
    for an access_token/refresh_token. Requires an app registered at https://dev.hh.ru/.
    """
    from hr_breaker.autoapply.hh_client import HHClient, build_authorization_url

    url = build_authorization_url(client_id, redirect_uri)
    click.echo(f"1. Open this URL, log in, and approve access:\n\n   {url}\n")
    click.echo("2. You'll be redirected to your redirect_uri with ?code=... in the URL.")
    code = click.prompt("3. Paste the value of the 'code' query parameter here")

    async def _exchange():
        client = HHClient()
        return await client.exchange_code_for_token(client_id, client_secret, code.strip(), redirect_uri)

    token_data = asyncio.run(_exchange())
    click.echo("\nSuccess. Add these to your .env (access tokens expire - keep the refresh_token too):\n")
    click.echo(f"HH_ACCESS_TOKEN={token_data.get('access_token', '')}")
    click.echo(f"HH_REFRESH_TOKEN={token_data.get('refresh_token', '')}")
    if "expires_in" in token_data:
        click.echo(f"# expires_in: {token_data['expires_in']}s")


@autoapply.command("resumes")
@click.option("--access-token", envvar="HH_ACCESS_TOKEN", required=True, help="hh.ru access token (see 'autoapply auth')")
def autoapply_resumes(access_token: str):
    """List your hh.ru resumes and their IDs (see hh_client module docstring - this
    endpoint is currently 403-blocked regardless of token validity)."""
    from hr_breaker.autoapply.hh_client import HHClient

    async def _list():
        client = HHClient(access_token=access_token)
        return await client.get_my_resumes()

    resumes = asyncio.run(_list())
    if not resumes:
        click.echo("No resumes found on your hh.ru account. Publish one at hh.ru first.")
        return
    for r in resumes:
        click.echo(f"  {r.get('id')}  {r.get('title', '(untitled)')}")


@autoapply.command("browser-login")
@click.option("--profile-dir", type=click.Path(path_type=Path), default=None, help="Where to persist the browser session (default: .cache/hh_browser_profile)")
def autoapply_browser_login(profile_dir: Path | None):
    """One-time interactive login: opens a visible browser window on hh.ru. Log in by
    hand, then press Enter in this terminal to save the session for 'autoapply run
    --live' (closing the browser window alone does not save it - see browser_apply
    module docstring)."""
    from hr_breaker.autoapply.browser_apply import PROFILE_DIR, login_interactively

    asyncio.run(login_interactively(profile_dir or PROFILE_DIR))
    click.echo("Session saved. Run 'hr-breaker autoapply run --live ...' to use it.")


# Mirrors hh_client.EXPERIENCE_VALUES - duplicated (not imported) so this module doesn't
# have to pull in the autoapply package (and its Playwright dependency) at CLI load time.
_EXPERIENCE_VALUES = ("noExperience", "between1And3", "between3And6", "moreThan6")


@autoapply.command("run")
@click.option("--trigger", "-t", "triggers", multiple=True, required=True, help="Trigger keyword to search for (repeatable)")
@click.option("--profile", "profile_id", default=None, help="Profile ID to tailor from (see 'hr-breaker profile')")
@click.option("--resume", "resume_path", default=None, type=click.Path(exists=True, path_type=Path), help="Or: a raw resume file instead of a profile")
@click.option("--area", default=None, help="hh.ru area/region id (see GET /areas) to restrict search to")
@click.option("--experience", default=None, type=click.Choice(_EXPERIENCE_VALUES), help="hh.ru experience filter")
@click.option("--exclude", "excluded_text", default=None, help="Comma-separated words - skip vacancies whose title contains any of them")
@click.option("--per-page", default=100, show_default=True, help="Vacancies fetched per search page (hh.ru caps this at 100)")
@click.option("--max-new", default=10, show_default=True, help="Max never-seen vacancies to tailor per run")
@click.option("--max-iterations", default=2, show_default=True, help="Max optimizer iterations per vacancy - lower is faster but risks a lower-quality resume")
@click.option("--lang", "lang_mode", default="from_job", help="from_job (default), from_resume, en, ru, ...")
@click.option("--live", is_flag=True, help="Actually submit applications via a real browser session (DEFAULT IS DRY RUN)")
@click.option("--max-apply", default=5, show_default=True, help="Safety cap: max applications actually sent in this run")
@click.option("--resume-title", default=None, help="Substring of the resume title to pick, if your hh.ru account has more than one")
@click.option("--headed", is_flag=True, help="Show the apply browser window instead of running headless (useful for the first --live run, or to solve a CAPTCHA)")
@click.option("--browser-profile-dir", type=click.Path(path_type=Path), default=None, help="Browser session dir from 'autoapply browser-login' (default: .cache/hh_browser_profile)")
def autoapply_run(
    triggers, profile_id, resume_path, area, experience, excluded_text, per_page, max_new, max_iterations, lang_mode,
    live, max_apply, resume_title, headed, browser_profile_dir,
):
    """Search hh.ru for TRIGGERS, tailor a resume + cover letter for each new vacancy.

    Without --live: dry run. Tailored PDFs + cover letters are saved under
    output/autoapply/ and tracked in .cache/autoapply.sqlite3 so re-running skips
    vacancies already seen. Nothing is sent to hh.ru.

    With --live: also submits the application through the browser session saved by
    'autoapply browser-login' - the cover letter is personalized per vacancy, but
    hh.ru's apply flow does not support attaching a different PDF per application.
    See `hr_breaker.autoapply.browser_apply` module docstring for detail.
    """
    from hr_breaker.autoapply import run_autoapply
    from hr_breaker.autoapply.browser_apply import PROFILE_DIR

    if profile_id is None and resume_path is None:
        raise click.UsageError("Provide either --profile <id> or --resume <path>")
    profile_dir = browser_profile_dir or PROFILE_DIR
    if live and not profile_dir.exists():
        raise click.UsageError(f"--live requires a browser session - run 'hr-breaker autoapply browser-login' first (expected at {profile_dir})")

    resume_source = None
    if resume_path is not None:
        resume_content = load_resume_content(resume_path)
        first_name, last_name, lang_code = asyncio.run(extract_name(resume_content))
        resume_source = ResumeSource(
            content=resume_content, first_name=first_name, last_name=last_name, language_code=lang_code,
        )

    def on_event(event, data):
        if event == "searched":
            click.echo(f"[{data['trigger']}] found {data['found']} vacancies")
        elif event == "tailored":
            click.echo(f"  ready: {data['company']} - {data['title']} -> {data['pdf_path']}")
        elif event == "applied":
            warning = "" if data.get("cover_letter_sent", True) else " (WARNING: no cover letter attached - see browser_debug/)"
            click.echo(f"  APPLIED: {data['company']} - {data['title']}{warning}")
        elif event == "failed":
            click.echo(f"  failed: {data['company']} - {data['title']}: {data.get('error')}")
        elif event == "iteration":
            click.echo(
                f"    [{data['title']}] iteration {data['iteration']}/{data['max_iterations']}: "
                f"{data['status']} [{data['scores']}] ({data['elapsed']:.0f}s, ETA {data['eta']:.0f}s)"
            )
        elif event == "skipped":
            click.echo(f"  skipped (not applied): {data['company']} - {data['title']}: {data.get('detail')}")
        elif event == "skipped_apply_cap":
            click.echo(f"  tailored but NOT applied (--max-apply cap reached): {data['company']} - {data['title']}")

    if not live:
        click.echo("DRY RUN (no applications will be sent - pass --live to actually apply)\n")

    async def _run():
        async with live_progress():
            return await run_autoapply(
                triggers=list(triggers),
                profile_id=profile_id,
                resume_source=resume_source,
                area=area,
                experience=experience,
                excluded_text=excluded_text,
                per_page=per_page,
                max_new=max_new,
                max_iterations=max_iterations,
                max_apply_per_run=max_apply,
                lang_mode=lang_mode,
                live=live,
                resume_title=resume_title,
                browser_headless=not headed,
                browser_profile_dir=profile_dir,
                on_event=on_event,
            )

    summary = asyncio.run(_run())

    click.echo(
        f"\nDone. found={summary.found} already_seen={summary.already_seen} "
        f"tailored={summary.tailored} failed={summary.failed} applied={summary.applied} "
        f"skipped_apply_cap={summary.skipped_apply_cap}"
    )


if __name__ == "__main__":
    cli()
