"""Cover letter generation - used by the auto-apply pipeline."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from hr_breaker.config import get_flash_model, get_model_settings, get_settings
from hr_breaker.models import JobPosting, OptimizedResume
from hr_breaker.models.language import Language, get_language_safe
from hr_breaker.utils.optimization_telemetry import run_tracked_agent

SYSTEM_PROMPT = """You are a career coach writing a short cover letter (сопроводительное письмо)
that accompanies a job application on a job board (like hh.ru).

INPUT: The candidate's optimized resume (already tailored to this job) and the job posting.

OUTPUT: Plain text only - the letter body itself, nothing else.
- Do NOT include a subject line, title, or heading (e.g. no "Отклик на вакансию").
- Do NOT include any greeting/opener word or line (no "Здравствуйте", "Hello", "ping", or
  similar) unless the target language's convention strongly expects one.
- Start directly with the first substantive sentence.
- 2-3 short paragraphs, no HTML, no markdown.

RULES:
- Write in the target language given below.
- No fabrication: only reference experience/skills that actually appear in the resume text.
- First sentence: why this specific role/company is a good fit (use the job title/company name).
- Middle: 1 concrete, resume-backed achievement most relevant to the role's requirements.
- Close with a short, confident call to action (available to discuss / interview) - one sentence.
- STYLE: short, simple sentences - one idea per sentence. Avoid compound/subordinate-clause-heavy
  sentences, bureaucratic phrasing, and filler words. Write like a direct, competent person would
  write a quick message, not a formal letter.
- No generic filler ("I am a hard worker", "I am excited to apply"). Be specific and concise.
- Do not repeat the resume verbatim - complement it, don't restate it.
- Target length: 80-140 words. This is a job-board message, not a formal letter - no
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
        model_settings=get_model_settings(get_settings().flash_model),
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
    result = await run_tracked_agent(
        agent, _build_prompt(resume_text, job, target_language), component="CoverLetter"
    )
    return result.output.text.strip()
