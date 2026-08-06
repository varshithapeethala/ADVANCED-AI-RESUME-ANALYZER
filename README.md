# AI Resume Analyzer 🚀

[![Streamlit App](https://static.streamlit.io/badge-repo.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)

An enterprise-grade, production-quality **AI Resume Analyzer** built in Python. This tool leverages NLP, Machine Learning, and LLMs (Gemini/OpenAI) to parse resumes, evaluate ATS compatibility scores, match candidates to Job Descriptions, and provide career improvement recommendations.

---

## Key Features

- **Multi-format Parsing**: Seamlessly extract text from PDF and DOCX formats using fallback parsing pipelines (`PyMuPDF` + `pdfplumber` + `python-docx`).
- **Deep Skill Extraction**: Automatically categorizes 300+ technical, database, cloud, soft, and specialized skills using advanced word boundary regex filters.
- **ATS Compatibility Scoring**: Dynamic grading (0-100) based on resume formatting, contact detail presence, action verbs density, education structure, and timeline markers.
- **Job Description (JD) Matcher**: Contextual comparison using TF-IDF Vectorizers and Cosine Similarity, showing matching skills, missing skills, keyword coverage, and strengths/weaknesses.
- **Interactive AI Copilot**: When supplied with a `GEMINI_API_KEY` or `OPENAI_API_KEY`, the analyzer generates professional executive summaries, missing skills advice, interview prep questions, and career suggestions. Fallbacks elegantly to offline NLP rules if keys are omitted.
- **Data Visualizations**: Beautiful dashboards generated with `Plotly` (ATS Gauge, Radar Strength Profile, Skill Distributions, and Score Breakdowns).

---

## Project Structure

```text
AI-Resume-Analyzer/
├── app.py                     # Streamlit web application entrypoint
├── requirements.txt           # Package dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Version control exclusions
├── LICENSE                    # MIT license details
├── assets/                    # Image assets, logo, screenshots
├── config/
│   └── settings.py            # Global weights, settings, and credentials
├── models/
│   └── ats_model.py           # ML regressor for resume scoring
├── utils/
│   ├── pdf_parser.py          # PyMuPDF and pdfplumber parsing
│   ├── docx_parser.py         # DOCX parser
│   ├── text_cleaner.py        # Text preprocessing and token cleaning
│   ├── skill_extractor.py     # Skill category extraction matching
│   ├── ats_score.py           # ATS scoring algorithm
│   ├── jd_match.py            # Cosine similarity and keyword overlap
│   ├── recommendation.py      # Bullet-point feedback generator
│   ├── charts.py              # Plotly interactive graphs
│   └── helpers.py             # spaCy auto-downloader and path helpers
├── data/
│   ├── skills.csv             # 300+ skills categorized
│   ├── job_roles.json         # JD templates for direct matching
│   └── stopwords.txt          # Stopwords list
├── notebooks/
│   └── model_training.ipynb   # Model development process
└── tests/
    ├── test_parser.py         # Unit tests for doc/pdf parsers
    └── test_score.py          # Unit tests for scoring/matching
```

---

## Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone or Download this Project
Ensure you have Python 3.12+ installed.

### 2. Create a Virtual Environment
Run the following in your terminal:
```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
Run the command below:
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables (Optional)
To activate the **AI Copilot** features, create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```
*(Alternatively, you can input your API keys directly into the Streamlit sidebar at runtime).*

---

## Running the Application

Launch the Streamlit web server:
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## Running Unit Tests

To run the test suite and verify parser and scoring correctness:
```bash
pip install pytest
pytest
```

---

## Deployment Instructions

### Streamlit Community Cloud (Recommended)
1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your repository.
3. Add your `GEMINI_API_KEY` or `OPENAI_API_KEY` in the **Advanced Settings -> Secrets** section of the Streamlit dashboard.

### Render / Railway
1. Define a `Start Command` of `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
2. Configure environment variables in your deployment dashboard.

---

## Future Roadmap

- [ ] Support OCR extraction for image-only or scanned PDF resumes.
- [ ] Add direct PDF download options for custom AI-generated cover letters.
- [ ] Incorporate fine-tuned local Hugging Face Models to classify experience sectors.
- [ ] Support batch resume comparisons for HR screening.
