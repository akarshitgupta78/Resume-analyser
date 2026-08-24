import base64
import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime

# ------------------------
# Styling (CSS)
# ------------------------
st.set_page_config(page_title="Resume Analyzer", layout="wide", initial_sidebar_state="expanded")

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root{
  --bg: #070b19;
  --card: #0f172a;
  --muted: #94a3b8;
  --accent: #7c3aed;
  --accent-glow: rgba(124, 58, 237, 0.15);
  --glass: rgba(255, 255, 255, 0.02);
  --border: rgba(255, 255, 255, 0.06);
}

/* Base override for premium dark appearance */
.stApp {
  background: linear-gradient(135deg, #070b19 0%, #0b0f19 50%, #0d1225 100%) !important;
  font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

[data-testid="stHeader"] {
  background-color: transparent !important;
}

/* Sidebar styling integration */
[data-testid="stSidebar"] {
  background-color: #080d1e !important;
  border-right: 1px solid var(--border);
}

/* Premium card layout */
.card {
  background: linear-gradient(185deg, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.3) 100%);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
  border: 1px solid var(--border);
  margin-bottom: 20px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card:hover {
  border-color: rgba(124, 58, 237, 0.3);
  box-shadow: 0 12px 35px rgba(124, 58, 237, 0.06);
  transform: translateY(-2px);
}

/* Custom styled buttons */
.stButton>button {
  width: 100%;
  border-radius: 12px;
  padding: 12px 24px;
  background: linear-gradient(90deg, #7c3aed, #06b6d4);
  color: white !important;
  font-weight: 600;
  font-family: 'Outfit', sans-serif;
  border: none;
  box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
  transition: all 0.2s ease;
}

.stButton>button:hover {
  box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
  transform: scale(1.02);
  border: none !important;
}

/* Header style layout */
.header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 30px;
}

.logo {
  background: linear-gradient(135deg, #06b6d4, #7c3aed);
  border-radius: 12px;
  padding: 10px;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 24px;
  color: white;
  box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
}

.kv {
  color: var(--muted);
  font-size: 13px;
}

.small {
  font-size: 13px;
  color: var(--muted);
}

.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  background: rgba(124, 58, 237, 0.1);
  color: #c084fc;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid rgba(124, 58, 237, 0.2);
}

.score-big {
  font-size: 54px;
  font-weight: 800;
  color: #ffffff;
  background: linear-gradient(120deg, #ffffff, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.1;
  margin: 10px 0;
}

.feedback-good {
  color: #10b981;
  font-weight: 600;
}

.feedback-bad {
  color: #f87171;
  font-weight: 600;
}
</style>
"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000"

# -------------------------
# Helper Functions
# -------------------------

def header():
    st.markdown(
        """
        <div class="header">
            <div class="logo">RA</div>
            <div>
                <h2 style="margin:0;padding:0">Resume Analyzer</h2>
                <div class="small">Instant resume feedback · keyword matching · scoring · suggestions</div>
            </div>
            <div style="flex:1"></div>
            <div class="badge">Powered by FastAPI & Streamlit</div>
        </div>
        """, unsafe_allow_html=True
    )

def sidebar_controls():
    st.sidebar.markdown("## 🔎 Analysis Settings")
    default_skills = "Python, SQL, Machine Learning, Deep Learning, Data Analysis, Excel, Power BI, Java, C++, React"
    job_title = st.sidebar.text_input("Job title (for context)", "Data Scientist")
    job_description = st.sidebar.text_area("Paste the job description (optional) — helps with scoring & similarity", height=180,
                                           value="We are looking for a Data Scientist with experience in Python, SQL, machine learning, and data visualization.")
    skills_input = st.sidebar.text_area("Target skills / keywords (comma separated)", value=default_skills, height=120)
    
    seniority = st.sidebar.selectbox("Seniority level", ["Entry (0-2 yrs)", "Mid (2-5 yrs)", "Senior (5+ yrs)"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Advanced Weights*")
    contact_weight = st.sidebar.slider("Weight: Contact Info (0-100)", 0, 100, 10)
    skill_weight = st.sidebar.slider("Weight: Skills (0-100)", 0, 100, 40)
    exp_weight = st.sidebar.slider("Weight: Experience (0-100)", 0, 100, 15)
    ed_weight = st.sidebar.slider("Weight: Education (0-100)", 0, 100, 15)
    length_weight = st.sidebar.slider("Weight: Length (0-100)", 0, 100, 10)
    
    return {
        "job_title": job_title,
        "job_description": job_description,
        "skills": skills_input,
        "seniority": seniority,
        "weights": {
            "contact": contact_weight / 100.0,
            "skill": skill_weight / 100.0,
            "experience": exp_weight / 100.0,
            "education": ed_weight / 100.0,
            "length": length_weight / 100.0
        }
    }

def render_analysis_results(result, job_title, skills_input):
    metrics = result['metrics']
    score = result['score']
    skill_matches = result['skill_matches']
    contact = result['contact']
    sim = result['similarity']
    years = result['years']
    ed_snips = result['education_snippets']

    # Top row: score + quick stats
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="small">Overall Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-big">{score}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kv">As of {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("*Contact Info*")
        st.markdown(f"- Email: {contact.get('email')}" if contact.get('email') else "- Email: *Not found*")
        st.markdown(f"- Phone: {contact.get('phone')}" if contact.get('phone') else "- Phone: *Not found*")
        st.markdown(f"- LinkedIn: {contact.get('linkedin')}" if contact.get('linkedin') else "- LinkedIn: Not found")
        st.markdown(f"- GitHub: {contact.get('github')}" if contact.get('github') else "- GitHub: Not found")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("*Quick Metrics*")
        st.markdown(f"- Skills matched: *{sum(1 for v in skill_matches.values() if v['present'])}/{len(skill_matches)}*")
        st.markdown(f"- Experience (est.): *{int(years)} years*")
        st.markdown(f"- Education found: *{len(ed_snips)} snippets*")
        st.markdown(f"- Job similarity: *{sim:.2f}*")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Skill chart
    st.subheader("Skill Matches")
    if len(skill_matches) == 0:
        st.info("No target skills specified to compare.")
    else:
        skill_df = pd.DataFrame([
            {"skill": k, "present": v['present'], "score": v['score']}
            for k, v in skill_matches.items()
        ]).sort_values(by=["present", "score"], ascending=[False, False])
        
        # Build interactive Altair horizontal bar chart
        chart = alt.Chart(skill_df).mark_bar(cornerRadiusEnd=6, height=18).encode(
            x=alt.X('score:Q', scale=alt.Scale(domain=[0, 1]), title='Match Score'),
            y=alt.Y('skill:N', sort=None, title='Target Skills'),
            color=alt.Color('present:N', scale=alt.Scale(
                domain=[True, False],
                range=['#10b981', '#f87171']
            ), legend=None),
            tooltip=['skill', 'present', 'score']
        ).properties(
            height=max(120, len(skill_df) * 35)
        ).configure_axis(
            labelColor='#94a3b8',
            titleColor='#cbd5e1',
            gridColor='rgba(255, 255, 255, 0.05)'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(chart, use_container_width=True)

    # Feedback panel
    st.subheader("Feedback & Suggestions")
    for item in result['feedback']:
        t = item['text']
        typ = item.get('type', '')
        if typ == "positive":
            st.markdown(f"- <span class='feedback-good'>✅ {t}</span>", unsafe_allow_html=True)
        elif typ == "negative":
            st.markdown(f"- <span class='feedback-bad'>❌ {t}</span>", unsafe_allow_html=True)
        elif typ == "suggestion":
            st.markdown(f"- 💡 {t}")
        elif typ == "kv":
            st.markdown(f"- *{t}*")
        elif typ == "tip":
            st.markdown(f"- ✨ {t}")
        else:
            st.write(f"- {t}")

    st.markdown("---")
    # Show snippets for Education and top lines
    st.subheader("Detected Education / Key Snippets")
    if ed_snips:
        for s in ed_snips[:5]:
            st.markdown(f"- {s}")
    else:
        st.write("No clear education snippet detected.")

    st.markdown("---")

def show_extracted_text(text):
    st.subheader("Extracted Resume Text (preview)")
    if not text.strip():
        st.info("No text could be extracted from this file.")
    else:
        preview = text[:2000]
        st.code(preview + ("..." if len(text) > 2000 else ""))
        if len(text) > 2000:
            with st.expander("Show full extracted text"):
                st.text_area("Full extracted text", value=text, height=300)

def build_report_markdown(result, filename: str, controls: dict):
    md = []
    md.append(f"# Resume Analysis Report — {filename}")
    md.append(f"*Analyzed on:* {datetime.now().isoformat()}")
    md.append("")
    md.append(f"*Job title (context):* {controls['job_title']}")
    md.append("")
    md.append(f"## Overall Score: {result['score']}/100")
    md.append("")
    md.append("## Key Metrics")
    for k, v in result['metrics'].items():
        md.append(f"- *{k}*: {v:.2f}")
    md.append("")
    md.append("## Contact Info")
    for k, v in result['contact'].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Skill Matches")
    for k, v in result['skill_matches'].items():
        md.append(f"- {k}: present={v['present']}, score={v['score']}")
    md.append("")
    md.append("## Feedback")
    for f in result['feedback']:
        md.append(f"- {f['text']}")
    md.append("")
    md.append("## Similarity to Job Description")
    md.append(f"- similarity: {result['similarity']:.3f}")
    md.append("")
    md.append("----")
    md.append("Generated by Resume Analyzer")
    return "\n".join(md)

# -------------------------
# Main App Flow
# -------------------------
def main():
    header()
    st.markdown("")  # spacing
    controls = sidebar_controls()

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload Resume")
        uploaded_file = st.file_uploader("Upload PDF resume", type=["pdf"], accept_multiple_files=False)
        st.markdown("*Example:* Use a resume with selectable text (not scanned as image).")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Quick Actions")
        
        run_demo = st.button("Run Demo with sample text")
        clear_out = st.button("Clear Output")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Handle clear output
        if clear_out:
            st.rerun()

        # Handle Run Demo
        if run_demo:
            sample_text = """
            John Doe
            john.doe@example.com | +1 555-123-4567 | linkedin.com/in/johndoe | github.com/johndoe
            Summary:
            Data Scientist with 4 years of experience building ML models and dashboards.

            Experience
            2019-2023: Data Scientist at Acme Corp - Built forecasting models using Python, scikit-learn and XGBoost.
            2017-2019: Analyst at Beta Ltd - SQL, Excel, Power BI.

            Education
            Bachelor of Technology in Computer Science, University of Somewhere, 2016

            Skills: Python, SQL, Machine Learning, Deep Learning, Data Visualization, Power BI, Docker
            """
            
            payload = {
                "text": sample_text,
                "job_title": controls["job_title"],
                "job_description": controls["job_description"],
                "skills": controls["skills"],
                "seniority": controls["seniority"],
                "weight_contact": controls["weights"]["contact"],
                "weight_skill": controls["weights"]["skill"],
                "weight_experience": controls["weights"]["experience"],
                "weight_education": controls["weights"]["education"],
                "weight_length": controls["weights"]["length"]
            }
            
            with st.spinner("Analyzing demo resume via API..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/api/analyze-text", json=payload)
                    if res.status_code == 200:
                        result = res.json()
                        render_analysis_results(result, controls['job_title'], controls['skills'])
                        show_extracted_text(sample_text)
                        
                        report_md = build_report_markdown(result, "demo_resume.pdf", controls)
                        st.download_button(
                            label="Download analysis report (Markdown)",
                            data=report_md,
                            file_name="demo_resume_analysis.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error(f"API returned an error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend API: {str(e)}")
            st.stop()

        # Handle file upload
        if uploaded_file is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"*Filename:* {uploaded_file.name}")
            st.markdown(f"*Filesize:* {len(uploaded_file.getvalue())//1024} KB")
            st.markdown("</div>", unsafe_allow_html=True)

            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {
                "job_title": controls["job_title"],
                "job_description": controls["job_description"],
                "skills": controls["skills"],
                "seniority": controls["seniority"],
                "weight_contact": controls["weights"]["contact"],
                "weight_skill": controls["weights"]["skill"],
                "weight_experience": controls["weights"]["experience"],
                "weight_education": controls["weights"]["education"],
                "weight_length": controls["weights"]["length"]
            }

            with st.spinner("Analyzing uploaded resume via API..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/api/analyze", files=files, data=data)
                    if res.status_code == 200:
                        result = res.json()
                        render_analysis_results(result, controls['job_title'], controls['skills'])
                        
                        # Show raw text preview (we don't get the raw text in the main analysis result, 
                        # so let's mock show_extracted_text with a nice summary or add extracted_text to the schema, 
                        # or retrieve raw text. Wait, does the schema have extracted_text? No, AnalysisResult does not have text.
                        # Wait! Should we add extracted_text to AnalysisResult? Or should the UI just say "Text successfully parsed"?
                        # In the original app, show_extracted_text(text) prints the raw extracted text of the resume.
                        # Let's check: did we have extracted_text in our backend schemas?
                        # No, we did not! Let's check: does it make sense to add "extracted_text" to the response model so that 
                        # the UI can preview it? Yes! The original UI had a preview of the extracted resume text.
                        # Let's add 'extracted_text: Optional[str]' to AnalysisResult schema and analyzer service so the frontend can preview it.
                        # Wait! Let's handle it gracefully: if the key is not in result, we print a nice notice, but we can easily update the backend schema and service to include it!
                        # Let's check how we can do that. It's a quick fix in analyzer.py and schemas/analysis.py.
                        # Let's implement it in frontend first, assuming we will add it to the backend shortly.)
                        
                        extracted = result.get("extracted_text", "")
                        if extracted:
                            show_extracted_text(extracted)
                        else:
                            st.info("Note: Text preview is not returned by the server, but the PDF was processed successfully.")
                        
                        report_md = build_report_markdown(result, uploaded_file.name, controls)
                        st.download_button(
                            label="Download analysis report (Markdown)",
                            data=report_md,
                            file_name=f"{uploaded_file.name}_analysis.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error(f"API returned an error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend API: {str(e)}")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### No Resume Uploaded")
            st.markdown("Upload a PDF resume on the left to analyze it. You can also paste a job description and a target skill list in the sidebar.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown("<div class='small'>Built with ❤ — modularized with FastAPI backend and Streamlit frontend.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
