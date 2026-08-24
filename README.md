# AI Resume Analyzer

An intelligent and modular resume analysis system built with **FastAPI** (backend) and **Streamlit** (frontend). It parses PDF resumes, extracts key details (contact info, education, work experience), computes target skill matches using fuzzy token-overlap heuristics, and calculates similarity with a job description using TF-IDF and cosine similarity metrics.

---

## Architecture

The project is structured as a decoupled application:
*   **FastAPI Backend (`app/`)**: Handles core business logic, including PDF parsing, text preprocessing, heuristic extraction, NLP scoring algorithms, and exports REST endpoints.
*   **Streamlit Frontend (`streamlit_app.py`)**: Provides a premium dark-themed dashboard, customizable analysis settings/weights, interactive Altair visualizations, and downloadable analysis reports.

---

## Features

*   **PDF Extraction**: Reads selectable text page-by-page.
*   **Information Parsing**: Extracts emails, phone numbers, social profiles (GitHub, LinkedIn), and location metrics.
*   **Experience & Education Detection**: Identifies degrees, academic institutions, and calculates years of experience.
*   **Target Keyword Evaluation**: Fuzzy string matching to assess candidate skill coverage against customizable target skill lists.
*   **Context Similarity Scoring**: Employs scikit-learn vector models to measure alignment with a job description.
*   **Configurable Weighting Engine**: Adjustable sliders dynamically redistribute the impact of contact detail completeness, skill matching, education, and formatting on the final score.

---

## Quick Start

### 1. Setup Environment
Ensure you have Python 3.10+ installed. Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start Backend API
Run the FastAPI server locally on port 8000:
```bash
python -m uvicorn app.main:app --port 8000 --reload
```
You can access the interactive API docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

### 3. Start Frontend Dashboard
In a separate terminal, launch the Streamlit application:
```bash
streamlit run streamlit_app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

---

## API Endpoints

### `POST /api/analyze`
Accepts a PDF file upload and form parameters (job title, job description, target skills list, seniority, weights) and returns a complete, structured JSON analysis report.

### `POST /api/analyze-text`
Accepts a JSON payload with raw resume text and parameters. Ideal for plain-text inputs or copy-pasted details.