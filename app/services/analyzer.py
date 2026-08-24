import re
import math
from datetime import datetime
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing: normalize whitespace, remove headers/footers common garbage
    """
    if not text:
        return ""
    # Replace multiple newlines and spaces
    t = re.sub(r'\r', '\n', text)
    t = re.sub(r'\n{2,}', '\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    # Remove weird characters
    t = re.sub(r'[^\x00-\x7F]+', ' ', t)
    return t.strip()

def extract_contact_info(text: str) -> dict:
    """
    Look for email, phone, linkedin, github, location
    """
    info = {}
    # Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    info['email'] = email_match.group(0) if email_match else None
    
    # Phone (many variations)
    phone_match = re.search(r'(\+?\d{2,3}[-.\s]?)?(\(?\d{3,4}\)?[-.\s]?){1,3}\d{3,4}', text)
    info['phone'] = phone_match.group(0) if phone_match else None
    
    # LinkedIn
    linkedin_match = re.search(r'(linkedin\.com\/[A-Za-z0-9-_]+)', text, re.IGNORECASE)
    info['linkedin'] = linkedin_match.group(0) if linkedin_match else None
    
    # Github
    github_match = re.search(r'(github\.com\/[A-Za-z0-9-_]+)', text, re.IGNORECASE)
    info['github'] = github_match.group(0) if github_match else None
    
    # Location -- heuristic: check explicit label or fall back to first few lines of resume
    loc_match = re.search(r'(?:(?:Location|Address|Based in|Lives in|Resident of)[:\s]+)([A-Za-z0-9 ,\-\.]+)', text, re.IGNORECASE)
    if loc_match:
        info['location'] = loc_match.group(1).strip()
    else:
        # Fallback: check first 8 non-empty lines for City, ST / City, Country pattern
        lines = [line.strip() for line in text.split('\n') if line.strip()][:8]
        found_loc = None
        for line in lines:
            # Match City, State (2-3 letter code) or City, Country name
            match = re.search(r'^([A-Z][a-zA-Z\s\.\-]+),\s*([A-Z]{2,3}|[A-Z][a-z]+)$', line)
            # Avoid picking up email-like lines, websites, github/linkedin, or long phone/date numbers
            if match and not re.search(r'@|\bhttp|github|linkedin|\d{4,}', line, re.IGNORECASE):
                found_loc = line
                break
        info['location'] = found_loc
    return info

def find_education(text: str) -> list:
    """
    Simple detection for education keywords like 'Bachelor', 'Master', 'B.Tech' etc.
    """
    ed_keywords = [
        r'Bachelor', r'B\.Sc', r'B\.A', r'BTech', r'B\.Tech', r'BE\b', r'Master', r'M\.Sc', r'MBA', r'Ph\.D', r'Doctoral',
        r'High School', r'School', r'Associate'
    ]
    found = []
    for kw in ed_keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            snippet = extract_snippet(text, m.start(), 80)
            found.append(snippet)
    return found

def find_experience_years(text: str) -> float:
    """
    Heuristic to estimate years of experience by looking for patterns like 'X years' or date ranges (supporting pre-2000 and present)
    """
    # Look for explicit "X years" patterns (with possible floats)
    years_matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s+years?', text, re.IGNORECASE)
    if years_matches:
        years = max(float(x) for x in years_matches)
        return years
        
    # Date ranges (support 1900s, 2000s, and "Present"/"Current")
    # Matches "2018 - 2023" or "1998 to Present"
    pattern = r'\b(19\d{2}|20\d{2})\b[\s\-–to]{1,4}\b(19\d{2}|20\d{2}|present|current)\b'
    ranges = re.findall(pattern, text, re.IGNORECASE)
    years = 0.0
    current_year = datetime.now().year
    
    for start, end in ranges:
        try:
            s = int(start)
            if end.lower() in ['present', 'current']:
                e = current_year
            else:
                e = int(end)
            if e >= s:
                years += (e - s)
        except:
            pass
            
    if years > 0:
        return float(years)
    return 0.0

def extract_snippet(text: str, pos: int, window: int = 60) -> str:
    start = max(0, pos - window//2)
    end = min(len(text), pos + window//2)
    return text[start:end].replace('\n', ' ')

def compute_skill_matches(text: str, skills: list) -> dict:
    """
    For each skill, compute presence using an optimized exact/regex search,
    falling back to sequence matching with rapid heuristics.
    """
    text_lower = text.lower()
    # Normalize punctuation for cleaner text matching
    text_clean = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean)
    tokens = text_clean.split()
    
    result = {}
    for skill in skills:
        s = skill.lower()
        s_clean = re.sub(r'[^a-z0-9\s]', ' ', s)
        s_clean = re.sub(r'\s+', ' ', s_clean).strip()
        
        present = False
        best_ratio = 0.0
        
        # 1. Exact substring check in clean text
        if s_clean in text_clean:
            present = True
            best_ratio = 1.0
        # 2. Exact word boundary check in clean text
        elif re.search(r'\b' + re.escape(s_clean) + r'\b', text_clean):
            present = True
            best_ratio = 1.0
        # 3. Fuzzy match fallback for multi-word or longer terms (>= 4 chars)
        elif len(s_clean) >= 4 and len(tokens) > 0:
            s_words = s_clean.split()
            n_words = len(s_words)
            first_chars = {w[0] for w in s_words if w}
            
            for i in range(len(tokens) - n_words + 1):
                # Optimization: Skip window if current token doesn't match starting chars of the skill
                if tokens[i][0] not in first_chars:
                    continue
                
                window_tokens = tokens[i : i + n_words + 2]
                window = " ".join(window_tokens)
                
                # Check character overlap ratio before invoking SequenceMatcher
                if abs(len(window) - len(s_clean)) > 5:
                    overlap = len(set(window) & set(s_clean))
                    if overlap < len(s_clean) * 0.5:
                        continue
                
                ratio = SequenceMatcher(None, s_clean, window).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                if ratio > 0.82:
                    present = True
                    break
                    
        result[skill] = {"present": present, "score": round(max(best_ratio, 1.0 if present else 0.0), 3)}
    return result

def resume_length_score(text: str) -> float:
    """
    Score based on length: ideal resume ~ 300-1000 words depending on seniority.
    We'll produce a 0-1 score.
    """
    words = len(re.findall(r'\w+', text))
    # ideal between 300 and 800 for many roles
    if words == 0:
        return 0.0
    if words < 200:
        return max(0.0, words / 300.0)  # penalize for too short
    if words <= 800:
        return 1.0
    # slight penalty for very long resumes
    return max(0.0, 1.0 - (words - 800) / 2000.0)

def compute_overall_score(metrics: dict, custom_weights: dict) -> int:
    """
    Combine metrics (all in 0..1) with custom weights to produce 0..100 score
    """
    # Normalize weights to sum to 0.9, leaving 0.1 for the format bonus
    wsum = sum(custom_weights.values()) or 1.0
    normalized_weights = {k: (v / wsum) * 0.9 for k, v in custom_weights.items()}
    
    total = 0.0
    total += metrics.get("skill", 0.0) * normalized_weights.get("skill", 0.4)
    total += metrics.get("contact", 0.0) * normalized_weights.get("contact", 0.1)
    total += metrics.get("education", 0.0) * normalized_weights.get("education", 0.15)
    total += metrics.get("experience", 0.0) * normalized_weights.get("experience", 0.15)
    total += metrics.get("length", 0.0) * normalized_weights.get("length", 0.1)
    
    # 0.1 weight format quality bonus
    total += metrics.get("format", 0.0) * 0.1
    
    # scale to 0..100
    return int(round(min(max(total, 0.0), 1.0) * 100))

def text_similarity(a: str, b: str) -> float:
    """
    Use TF-IDF + cosine similarity to compute similarity of resume to job description.
    Returns 0..1
    """
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=4000)
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        if math.isnan(sim):
            return 0.0
        return float(sim)
    except Exception:
        # fallback to SequenceMatcher on raw text
        return SequenceMatcher(None, a, b).ratio()

def detect_format_quality(text: str) -> float:
    """
    Heuristic: penalize resumes that are all uppercase, or extremely long lines, or a lot of punctuation
    Return 0..1
    """
    if not text:
        return 0.0
    # uppercase ratio
    letters = re.findall(r'[A-Za-z]', text)
    if len(letters) == 0:
        return 0.0
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    # average line length
    lines = [l for l in text.splitlines() if l.strip()]
    avg_len = sum(len(l) for l in lines) / max(1, len(lines))
    # punctuation density
    punct_ratio = sum(1 for c in text if c in '.,;:()[]{}') / max(1, len(text))
    # compute score
    score = 1.0
    if uppercase_ratio > 0.25:
        score -= 0.3
    if avg_len > 200:
        score -= 0.3
    if punct_ratio > 0.05:
        score -= 0.1
    return max(0.0, min(1.0, score))

def generate_feedback(metrics, skill_matches, contact, years, ed, sim, score, skills):
    """
    Generate human readable feedback points and suggestions
    """
    fb = []
    # Overall summary
    if score >= 85:
        fb.append({"type": "positive", "text": "Excellent — your resume is strong and well aligned with typical job expectations."})
    elif score >= 65:
        fb.append({"type": "neutral", "text": "Good — a few improvements will make your resume much stronger."})
    elif score >= 40:
        fb.append({"type": "neutral", "text": "Fair — you have a foundation but should address the issues below."})
    else:
        fb.append({"type": "negative", "text": "Needs improvement — consider the suggestions below to raise your score."})

    # Contact info
    if not contact.get('email'):
        fb.append({"type": "negative", "text": "Add a professional email address at the top of your resume (e.g., firstname.lastname@example.com)."})
    if not contact.get('phone'):
        fb.append({"type": "negative", "text": "Include a phone number so recruiters can contact you easily."})
    if not (contact.get('linkedin') or contact.get('github')):
        fb.append({"type": "suggestion", "text": "Add LinkedIn and/or GitHub links (if relevant) to showcase projects."})
    else:
        if contact.get('linkedin'):
            fb.append({"type": "positive", "text": "LinkedIn link detected — good for hiring managers."})
        if contact.get('github'):
            fb.append({"type": "positive", "text": "GitHub link detected — great for technical candidates."})

    # Skills
    total_skills = len(skills)
    found = [s for s, v in skill_matches.items() if v['present']]
    missing = [s for s, v in skill_matches.items() if not v['present']]
    if total_skills > 0:
        fb.append({"type": "kv", "text": f"Skill match: {len(found)}/{total_skills} keywords detected."})
        if missing:
            fb.append({"type": "negative", "text": f"Consider adding or emphasizing these keywords if they are relevant: {', '.join(missing[:8])}."})
        else:
            fb.append({"type": "positive", "text": "Good: all target keywords were found."})

    # Education
    if ed:
        fb.append({"type": "positive", "text": f"Education entries found (e.g., {ed[0][:80]})."})
    else:
        fb.append({"type": "negative", "text": "No clear education section detected. Add degree names & institutions."})

    # Experience
    if years >= 1:
        fb.append({"type": "positive", "text": f"Detected ~{int(years)} years of experience."})
    else:
        fb.append({"type": "neutral", "text": "Could not reliably detect years of experience. Ensure your job dates are present and clearly formatted (e.g., 2019–2023)."})
    
    # similarity
    if sim > 0.5:
        fb.append({"type": "positive", "text": f"Your resume is reasonably similar to the job description (similarity {sim:.2f})."})
    else:
        fb.append({"type": "suggestion", "text": f"Low similarity to the job description (similarity {sim:.2f}). Add target keywords and match phrasing."})

    # Format
    if metrics['format'] < 0.6:
        fb.append({"type": "negative", "text": "Formatting issues detected: long lines, excessive uppercase, or unusual punctuation. Consider using clear headings and bullet points."})
    else:
        fb.append({"type": "positive", "text": "Formatting looks clean and readable."})

    # length
    length_score = metrics['length']
    if length_score < 0.4:
        fb.append({"type": "negative", "text": "Resume seems short. Add more detail on projects or responsibilities."})
    elif length_score < 0.8:
        fb.append({"type": "positive", "text": "Length is within an acceptable range."})
    else:
        fb.append({"type": "suggestion", "text": "Resume might be long. Consider trimming older or irrelevant details."})

    # concrete tips
    fb.append({"type": "tip", "text": "Tip: use bullet points for achievements, quantify impact (metrics), and tailor your resume per job using keywords."})
    return fb

def analyze_resume(text: str, job_desc: str, skills: list, seniority: str, weights: dict) -> dict:
    metrics = {}
    
    # Contact Info
    contact = extract_contact_info(text)
    contact_score = 0.0
    if contact.get('email'):
        contact_score += 0.5
    if contact.get('phone'):
        contact_score += 0.5
    if contact.get('linkedin'):
        contact_score += 0.25
    if contact.get('github'):
        contact_score += 0.25
    contact_score = min(1.0, contact_score)
    metrics['contact'] = contact_score

    # Education
    ed = find_education(text)
    metrics['education'] = 1.0 if ed else 0.0

    # Experience
    years = find_experience_years(text)
    if seniority.startswith("Entry"):
        target_years = 1
    elif seniority.startswith("Mid"):
        target_years = 3
    else:
        target_years = 6
    metrics['experience'] = min(1.0, years / max(1.0, target_years))

    # Skills Match
    skill_matches = compute_skill_matches(text, skills)
    if len(skills) == 0:
        skill_coverage = 0.0
    else:
        found_count = sum(1 for v in skill_matches.values() if v['present'])
        fuzzy_sum = sum(v['score'] for v in skill_matches.values())
        skill_coverage = (found_count + fuzzy_sum * 0.2) / max(1, len(skills))
        skill_coverage = min(1.0, skill_coverage)
    metrics['skill'] = skill_coverage

    # Length
    metrics['length'] = resume_length_score(text)

    # Format
    metrics['format'] = detect_format_quality(text)

    # Similarity
    sim = text_similarity(text, job_desc)
    metrics['similarity'] = sim

    # Score
    score = compute_overall_score(metrics, weights)

    # Feedback
    feedback = generate_feedback(metrics, skill_matches, contact, years, ed, sim, score, skills)

    return {
        "metrics": metrics,
        "skill_matches": skill_matches,
        "score": score,
        "feedback": feedback,
        "contact": contact,
        "years": years,
        "education_snippets": ed,
        "similarity": sim,
        "extracted_text": text
    }
