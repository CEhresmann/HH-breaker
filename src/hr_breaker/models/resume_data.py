"""Structured resume data model for renderer-agnostic resume generation."""

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Contact information section."""

    name: str
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None
    location: str | None = None


class Experience(BaseModel):
    """Work experience entry."""

    company: str
    title: str
    location: str | None = None
    start_date: str
    end_date: str | None = None  # None means "Present"
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """Education entry."""

    institution: str
    degree: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    details: list[str] = Field(default_factory=list)


class Project(BaseModel):
    """Project entry."""

    name: str
    description: str | None = None
    url: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """Structured resume data - renderer-agnostic."""

    contact: ContactInfo
    summary: str | None = None
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)


class RenderResult(BaseModel):
    """Result of rendering a resume to PDF."""

    pdf_bytes: bytes
    page_count: int
    warnings: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
