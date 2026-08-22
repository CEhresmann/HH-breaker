"""Cover letter generation - used by the auto-apply pipeline."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from hr_breaker.config import get_flash_model, get_model_settings
from hr_breaker.models import JobPosting, OptimizedResume
from hr_breaker.models.language import Language, get_language_safe
from hr_breaker.utils.retry import run_with_retry

SYSTEM_PROMPT = """You are a career coach writing a short cover letter (сопроводительное письмо)
that accompanies a job application.

INPUT: The candidate's optimized resume (already tailored to this job) and the job posting.

OUTPUT: A short cover letter body, 3-5 short paragraphs, plain text (no HTML, no markdown headers).

RULES:
- Write in the target language given below.
- No fabrication: only reference experience/skills that actually appear in the resume text.
- Open with why this specific role/company is a good fit (use the job title/company name).
- Middle: 1-2 concrete, resume-backed achievements relevant to the role's requirements.
- Close with a short, confident call to action (available to discuss / interview).
- No generic filler ("I am a hard worker", "I am excited to apply"). Be specific and concise.
- Do not repeat the resume verbatim - complement it, don't restate it.
- Target length: 120-220 words. This is a job-board message, not a formal letter - no
  "Dear Hiring Manager" / "Sincerely" boilerplate unless the target language convention expects it.
"""


class CoverLetter(BaseModel):
    text: str = Field(description="Plain-text cover letter body")


def _build_prompt(resume_text: str, job: JobPosting, language: Language) -> str:
    return (
        f"Target language: {language.english_name} ({language.code})\n\n"
        f"JOB POSTING\nTitle: {job.title}\nCompany: {job.company}\n"
        f"Requirements: {', '.join(job.requirements) or 'n/a'}\n"
        f"Description:\n{job.description or job.raw_text}\n\n"
        f"CANDIDATE'S OPTIMIZED RESUME (plain text extracted from the tailored PDF):\n{resume_text}"
    )


def get_cover_letter_agent() -> Agent:
    return Agent(
        get_flash_model(),
        output_type=CoverLetter,
        system_prompt=SYSTEM_PROMPT,
        model_settings=get_model_settings(),
    )


async def write_cover_letter(
    optimized: OptimizedResume,
    job: JobPosting,
    language: Language | None = None,
) -> str:
    """Generate a short cover letter tailored to an already-optimized resume + job posting."""
    resume_text = optimized.pdf_text or optimized.html or ""
    target_language = language or get_language_safe(job.language_code)

    agent = get_cover_letter_agent()
    result = await run_with_retry(
        agent.run, _build_prompt(resume_text, job, target_language)
    )
    return result.output.text.strip()
