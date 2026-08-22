"""Batch pipeline: search hh.ru vacancies by trigger keywords, tailor a resume + cover
letter per new vacancy, optionally submit the application through hh.ru's API.

hh.ru's apply flow references an existing resume by `resume_id`, not an arbitrary
per-application PDF (see hh_client module docstring). With `live=True`, only the
cover letter is sent as part of the real application; the tailored PDF is generated
and saved locally but not attached to it.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hr_breaker.agents import write_cover_letter
from hr_breaker.models import JobPosting, ResumeSource
from hr_breaker.models.language import get_language_safe, resolve_target_language
from hr_breaker.orchestration import optimize_for_job
from hr_breaker.services.pdf_storage import generate_run_id

from .hh_client import HHApiError, HHClient, Vacancy
from .state_store import AutoApplyStore

logger = logging.getLogger(__name__)

AUTOAPPLY_OUTPUT_DIR = Path("output/autoapply")


@dataclass
class AutoApplyRunSummary:
    searched_triggers: list[str] = field(default_factory=list)
    found: int = 0
    already_seen: int = 0
    tailored: int = 0
    failed: int = 0
    applied: int = 0
    skipped_apply_cap: int = 0
    dry_run: bool = True
    vacancies: list[dict] = field(default_factory=list)


def _vacancy_to_job_posting(vacancy: Vacancy, language_code: str = "ru") -> JobPosting:
    """Build a JobPosting directly from hh.ru's structured vacancy data - no LLM call needed."""
    return JobPosting(
        title=vacancy.name,
        company=vacancy.employer_name,
        requirements=vacancy.key_skills,
        keywords=vacancy.key_skills,
        language_code=language_code,
        description=vacancy.description,
        raw_text=vacancy.description,
    )


async def _resolve_source(
    *,
    profile_id: str | None,
    resume_source: ResumeSource | None,
    job: JobPosting,
    docs_filter: str | None,
) -> ResumeSource:
    """Get a ResumeSource either directly, or synthesized from a profile ranked against `job`."""
    if resume_source is not None:
        return resume_source
    if profile_id is None:
        raise ValueError("Either resume_source or profile_id must be provided")

    from hr_breaker.services.profile_retrieval import rank_profile_documents, synthesize_profile_resume_source
    from hr_breaker.services.profile_store import ProfileStore

    store = ProfileStore()
    profile = store.get_profile(profile_id)
    if profile is None:
        raise ValueError(f"Profile not found: {profile_id}")

    all_docs = store.list_documents(profile_id)
    if not all_docs:
        raise ValueError(f"Profile '{profile_id}' has no documents")

    if docs_filter:
        wanted = {d.strip() for d in docs_filter.split(",")}
        selected = [d for d in all_docs if d.id in wanted or d.id[:12] in wanted]
    else:
        selected = [d for d in all_docs if d.included_by_default] or all_docs

    ranked = await rank_profile_documents(selected, job)
    return synthesize_profile_resume_source(profile, selected, ranked)


async def run_autoapply(
    *,
    triggers: list[str],
    profile_id: str | None = None,
    resume_source: ResumeSource | None = None,
    docs_filter: str | None = None,
    area: str | None = None,
    excluded_text: str | None = None,
    per_page: int = 20,
    max_new: int = 10,
    max_apply_per_run: int = 5,
    lang_mode: str = "from_job",
    live: bool = False,
    access_token: str | None = None,
    hh_resume_id: str | None = None,
    max_iterations: int | None = None,
    store: AutoApplyStore | None = None,
    output_dir: Path = AUTOAPPLY_OUTPUT_DIR,
    on_event: Callable[[str, dict], None] | None = None,
) -> AutoApplyRunSummary:
    """Run one pass of the auto-apply pipeline.

    `live=False` (default): vacancies are found, deduped, resumes tailored, cover
    letters written, PDFs saved - nothing is sent to hh.ru. `live=True` requires
    `access_token` + `hh_resume_id` and submits applications, capped at
    `max_apply_per_run` per call.
    """
    store = store or AutoApplyStore()
    hh = HHClient(access_token=access_token)
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(event: str, data: dict) -> None:
        if on_event:
            on_event(event, data)

    summary = AutoApplyRunSummary(searched_triggers=list(triggers), dry_run=not live)

    excluded_words = [w.strip().lower() for w in (excluded_text or "").split(",") if w.strip()]

    candidates: list[tuple[str, Vacancy]] = []  # (trigger, vacancy)
    for trigger in triggers:
        vacancies = await hh.search_vacancies(text=trigger, area=area, per_page=per_page)
        emit("searched", {"trigger": trigger, "found": len(vacancies)})
        for v in vacancies:
            summary.found += 1
            if store.seen(v.id):
                summary.already_seen += 1
                continue
            if excluded_words and any(w in v.name.lower() for w in excluded_words):
                continue
            candidates.append((trigger, v))
            if len(candidates) >= max_new:
                break
        if len(candidates) >= max_new:
            break

    for trigger, vacancy in candidates:
        outcome = {"vacancy_id": vacancy.id, "title": vacancy.name, "company": vacancy.employer_name}
        store.upsert(
            vacancy.id, "seen",
            title=vacancy.name, company=vacancy.employer_name, url=vacancy.url,
            trigger_keyword=trigger, raw=vacancy.raw,
        )
        emit("processing", outcome)

        try:
            vacancy = await hh.get_vacancy_detail(vacancy.id)
            await asyncio.sleep(0.5)  # be a polite visitor to hh.ru's website, not just the API
            job = _vacancy_to_job_posting(vacancy)
            source = await _resolve_source(
                profile_id=profile_id, resume_source=resume_source, job=job, docs_filter=docs_filter
            )
            target_language = resolve_target_language(lang_mode, job.language_code, source.language_code)
            source_language = get_language_safe(source.language_code)

            optimized, validation, job = await optimize_for_job(
                source,
                job=job,
                max_iterations=max_iterations,
                parallel=True,
                language=target_language,
                source_language=source_language,
            )

            if not optimized.pdf_bytes:
                raise RuntimeError("PDF render failed")

            run_id = generate_run_id()
            pdf_path = output_dir / f"{run_id}_{vacancy.id}.pdf"
            pdf_path.write_bytes(optimized.pdf_bytes)

            cover_letter = await write_cover_letter(optimized, job, language=target_language)

            store.upsert(
                vacancy.id, "ready",
                cover_letter=cover_letter, pdf_path=str(pdf_path),
            )
            summary.tailored += 1
            outcome.update({
                "status": "ready", "pdf_path": str(pdf_path), "cover_letter": cover_letter,
                "filters_passed": validation.passed,
            })
            emit("tailored", outcome)

        except Exception as e:  # noqa: BLE001 - one failed vacancy must not kill the run
            logger.exception(f"Failed to tailor vacancy {vacancy.id}: {e}")
            store.upsert(vacancy.id, "failed", error=str(e))
            summary.failed += 1
            outcome.update({"status": "failed", "error": str(e)})
            emit("failed", outcome)
            summary.vacancies.append(outcome)
            continue

        if not live:
            summary.vacancies.append(outcome)
            continue

        if summary.applied >= max_apply_per_run:
            summary.skipped_apply_cap += 1
            outcome["status"] = "skipped_apply_cap"
            emit("skipped_apply_cap", outcome)
            summary.vacancies.append(outcome)
            continue

        if not hh_resume_id:
            outcome.update({"status": "failed", "error": "live=True but no hh_resume_id provided"})
            store.upsert(vacancy.id, "failed", error=outcome["error"])
            summary.failed += 1
            emit("failed", outcome)
            summary.vacancies.append(outcome)
            continue

        try:
            await hh.apply_to_vacancy(hh_resume_id, vacancy.id, message=cover_letter)
            store.upsert(vacancy.id, "applied")
            summary.applied += 1
            outcome["status"] = "applied"
            emit("applied", outcome)
        except HHApiError as e:
            logger.exception(f"Failed to apply to vacancy {vacancy.id}: {e}")
            store.upsert(vacancy.id, "failed", error=str(e))
            summary.failed += 1
            outcome.update({"status": "failed", "error": str(e)})
            emit("failed", outcome)

        summary.vacancies.append(outcome)

    return summary
