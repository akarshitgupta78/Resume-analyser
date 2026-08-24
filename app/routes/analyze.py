from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from app.services.pdf_parser import extract_text_from_pdf
from app.services.analyzer import analyze_resume
from app.schemas.analysis import AnalysisResult, AnalysisRequestText

router = APIRouter(prefix="/api", tags=["Analysis"])

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_uploaded_resume(
    file: UploadFile = File(...),
    job_title: Optional[str] = Form("Data Scientist"),
    job_description: Optional[str] = Form(""),
    skills: Optional[str] = Form(""),
    seniority: Optional[str] = Form("Entry (0-2 yrs)"),
    weight_contact: Optional[float] = Form(0.10),
    weight_skill: Optional[float] = Form(0.40),
    weight_experience: Optional[float] = Form(0.15),
    weight_education: Optional[float] = Form(0.15),
    weight_length: Optional[float] = Form(0.10)
):
    # Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name cannot be empty")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file extension. Only PDF files are allowed.")
        
    # Extract text from uploaded file bytes
    try:
        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF file: {str(e)}")
    finally:
        await file.close()
        
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any readable text from the uploaded PDF.")

    # Parse skills
    skills_list = []
    if skills:
        skills_list = [s.strip() for s in skills.split(",") if s.strip()]

    # Collect weights
    weights = {
        "contact": weight_contact if weight_contact is not None else 0.10,
        "skill": weight_skill if weight_skill is not None else 0.40,
        "experience": weight_experience if weight_experience is not None else 0.15,
        "education": weight_education if weight_education is not None else 0.15,
        "length": weight_length if weight_length is not None else 0.10
    }

    try:
        # Perform analysis
        report = analyze_resume(
            text=extracted_text,
            job_desc=job_description or "",
            skills=skills_list,
            seniority=seniority or "Entry (0-2 yrs)",
            weights=weights
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")

@router.post("/analyze-text", response_model=AnalysisResult)
def analyze_text_only(request: AnalysisRequestText):
    # Parse skills
    skills_list = []
    if request.skills:
        skills_list = [s.strip() for s in request.skills.split(",") if s.strip()]

    # Collect weights
    weights = {
        "contact": request.weight_contact,
        "skill": request.weight_skill,
        "experience": request.weight_experience,
        "education": request.weight_education,
        "length": request.weight_length
    }

    try:
        report = analyze_resume(
            text=request.text,
            job_desc=request.job_description or "",
            skills=skills_list,
            seniority=request.seniority or "Entry (0-2 yrs)",
            weights=weights
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

