from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class AnalysisWeights(BaseModel):
    contact: float = Field(default=0.10, ge=0.0, le=1.0)
    skill: float = Field(default=0.40, ge=0.0, le=1.0)
    experience: float = Field(default=0.15, ge=0.0, le=1.0)
    education: float = Field(default=0.15, ge=0.0, le=1.0)
    length: float = Field(default=0.10, ge=0.0, le=1.0)

class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None

class SkillMatchDetail(BaseModel):
    present: bool
    score: float

class FeedbackItem(BaseModel):
    type: str
    text: str

class AnalysisResult(BaseModel):
    metrics: Dict[str, float]
    skill_matches: Dict[str, SkillMatchDetail]
    score: int
    feedback: List[FeedbackItem]
    contact: ContactInfo
    years: float
    education_snippets: List[str]
    similarity: float
    extracted_text: Optional[str] = None

class AnalysisRequestText(BaseModel):
    text: str
    job_title: Optional[str] = "Data Scientist"
    job_description: Optional[str] = ""
    skills: Optional[str] = ""
    seniority: Optional[str] = "Entry (0-2 yrs)"
    weight_contact: float = 0.10
    weight_skill: float = 0.40
    weight_experience: float = 0.15
    weight_education: float = 0.15
    weight_length: float = 0.10

