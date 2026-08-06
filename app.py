import os
import re
import streamlit as st
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Import utilities
from config.settings import Settings
from utils.helpers import load_spacy_model, get_data_path
from utils.pdf_parser import PDFParser
from utils.docx_parser import DocxParser
from utils.text_cleaner import TextCleaner
from utils.skill_extractor import SkillExtractor
from utils.ats_score import ATSScorer
from utils.jd_match import JDMatcher
from utils.recommendation import RecommendationEngine
from utils.charts import ResumeCharts
from models.ats_model import ATSModel

# Page configuration
st.set_page_config(
    page_title="ATS AI Resume Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injection
st.markdown("""
<style>
    /* Dark slate premium theme adjustments */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #38bdf8;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .skill-tag {
        background-color: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .section-tag {
        background-color: rgba(168, 85, 247, 0.1);
        color: #c084fc;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialization & Caching
@st.cache_resource
def get_nlp_model():
    return load_spacy_model()

@st.cache_resource
def get_analyzers():
    return {
        "cleaner": TextCleaner(),
        "extractor": SkillExtractor(),
        "matcher": JDMatcher(),
        "ml_model": ATSModel()
    }

# Load tools
nlp = get_nlp_model()
tools = get_analyzers()

# Load JDs templates database
@st.cache_data
def load_jd_templates():
    path = get_data_path("job_roles.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

jd_templates = load_jd_templates()

# Session State Initializer
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "api_key_openai" not in st.session_state:
    st.session_state.api_key_openai = Settings.OPENAI_API_KEY
if "api_key_gemini" not in st.session_state:
    st.session_state.api_key_gemini = Settings.GEMINI_API_KEY

# Sidebar controls
st.sidebar.title("💼 AI Resume Analyzer")
st.sidebar.caption("Production Quality Resume Scorer & JD Matcher")

# API Keys Configuration in Sidebar
with st.sidebar.expander("🔑 AI Copilot Settings", expanded=False):
    st.session_state.api_key_gemini = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key_gemini,
        type="password",
        help="Optional: Input to unlock full AI Copilot reviews."
    )
    st.session_state.api_key_openai = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key_openai,
        type="password",
        help="Optional: Fallback AI model key."
    )

# Page Navigation Selection
page = st.sidebar.radio(
    "Navigate Menu",
    ["📊 Dashboard Overview", "🎯 Job Description Matcher", "📝 Detailed Sections Scoring", "🤖 AI Career Copilot"]
)

# Document Upload Section
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or DOCX resume file",
    type=["pdf", "docx"],
    help="Supported formats: PDF, DOCX (Max 10MB)"
)

# Parse uploaded file
if uploaded_file:
    # Check if we need to parse or if it's already in session state
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    
    # Simple cache key
    cache_key = f"{file_name}_{len(file_bytes)}"
    
    if st.session_state.resume_data is None or st.session_state.resume_data.get("cache_key") != cache_key:
        with st.spinner("Processing document and extracting text..."):
            # Save file temporarily to disk for parser input
            temp_path = Path(f"temp_{file_name}")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            # Extract
            if file_name.endswith(".pdf"):
                parsed = PDFParser.parse(str(temp_path))
            else:
                parsed = DocxParser.parse(str(temp_path))
                
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)
                
            if parsed["success"]:
                # Run extraction engines
                text = parsed["text"]
                skills_res = tools["extractor"].extract_skills(text)
                
                # Simple entities parsing via spaCy/regex
                doc = nlp(text[:5000]) # Limit length to prevent parser freeze
                
                # Extracted profile details
                emails = [token.text for token in doc if token.like_email]
                email = emails[0] if emails else ""
                
                # Fetch links
                github = ""
                git_find = re.search(r'github\.com\/[A-Za-z0-9_-]+', text.lower())
                if git_find:
                    github = git_find.group(0)
                    
                linkedin = ""
                li_find = re.search(r'linkedin\.com\/in\/[A-Za-z0-9_-]+', text.lower())
                if li_find:
                    linkedin = li_find.group(0)
                    
                portfolio = ""
                pf_find = re.search(r'(portfolio|website|homepage)\b', text.lower())
                if pf_find:
                    portfolio = "Available in resume"
                
                # Simple Name Extraction (spaCy NER)
                name = ""
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                        name = ent.text
                        break
                if not name:
                    # Fallback to first line
                    lines = [l.strip() for l in text.split("\n") if l.strip()]
                    name = lines[0] if lines else "Applicant Name"

                parsed_info = {
                    "name": name,
                    "email": email,
                    "linkedin": linkedin,
                    "github": github,
                    "portfolio": portfolio
                }

                # Score Resume
                ats_score_details = ATSScorer.calculate_score(text, parsed["page_count"], skills_res["all_extracted"])
                
                # Machine Learning Predictor
                ml_score = tools["ml_model"].predict_readability(
                    ats_score_details["breakdown"]["Formatting & Structure"],
                    ats_score_details["breakdown"]["Contact Information"],
                    ats_score_details["breakdown"]["Skills Density"],
                    ats_score_details["breakdown"]["Experience & Education"],
                    ats_score_details["breakdown"]["Action Verbs"]
                )
                
                # Average them for balanced robustness
                final_ats_score = int(round((ats_score_details["score"] + ml_score) / 2))

                st.session_state.resume_data = {
                    "cache_key": cache_key,
                    "text": text,
                    "name": name,
                    "page_count": parsed["page_count"],
                    "parsed_info": parsed_info,
                    "skills_res": skills_res,
                    "ats_score_details": ats_score_details,
                    "final_score": final_ats_score
                }
                st.success("Resume parsed successfully!")
            else:
                st.error(f"Failed to parse document: {parsed['error']}")
                st.session_state.resume_data = None

# If no resume is uploaded, display upload prompt
if not st.session_state.resume_data:
    st.title("💼 AI Resume Analyzer & Scorer")
    st.write("Welcome! Build a premium profile using our ATS Resume score evaluation tool.")
    
    st.info("👈 Please upload a PDF or DOCX Resume in the sidebar to begin analysis.")
    
    # Mock / Demo Resume load
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### How it works:")
        st.write("1. **Upload Resume**: Provide your PDF or DOCX file.")
        st.write("2. **Scan Compatibility**: Analyze formatting, structure, and keyword density.")
        st.write("3. **Map JDs**: Paste target job descriptions to identify missing skills.")
        st.write("4. **AI Copilot**: Ask specific prep questions or rewrite sections.")
    with col2:
        st.markdown("### Tech Stack Used:")
        st.code("""
- Streamlit (Frontend Dashboard)
- spaCy (Named Entity Recognition)
- PyMuPDF & pdfplumber (PDF extraction)
- Scikit-learn (Similarity calculations)
- Plotly (Dynamic charts)
- Optional: Gemini/OpenAI API (Generative AI)
        """, language="markdown")
else:
    # Active parsed resume context
    data = st.session_state.resume_data
    
    # PAGE 1: DASHBOARD OVERVIEW
    if page == "📊 Dashboard Overview":
        st.title(f"📊 Resume Analysis Dashboard")
        st.markdown(f"### Candidate Profile: **{data['name']}**")
        
        # Stat cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ATS Compatibility</div>
                <div class="metric-value">{data['final_score']} / 100</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Page Count</div>
                <div class="metric-value">{data['page_count']} Page(s)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Skills Identified</div>
                <div class="metric-value">{len(data['skills_res']['all_extracted'])} Skills</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Action Verbs Found</div>
                <div class="metric-value">{len(data['ats_score_details']['verbs_found'])} Verbs</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly charts row
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            gauge_fig = ResumeCharts.create_ats_gauge(data['final_score'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        with chart_col2:
            pie_fig = ResumeCharts.create_score_breakdown_pie(data['ats_score_details']['breakdown'])
            st.plotly_chart(pie_fig, use_container_width=True)

        # Contact and Skills Details
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.subheader("Contact Information Extracted")
            info = data["parsed_info"]
            st.write(f"📧 **Email**: {info['email'] if info['email'] else 'Not Found'}")
            st.write(f"🔗 **LinkedIn**: {info['linkedin'] if info['linkedin'] else 'Not Found'}")
            st.write(f"💻 **GitHub**: {info['github'] if info['github'] else 'Not Found'}")
            st.write(f"🌐 **Portfolio**: {info['portfolio'] if info['portfolio'] else 'Not Found'}")

            # Feedback
            st.subheader("Strengths Identified")
            for strength in data['ats_score_details']['strengths'][:4]:
                st.success(strength)

        with detail_col2:
            st.subheader("Extracted Skills Taxonomy")
            by_category = data["skills_res"]["by_category"]
            if by_category:
                for cat, skills in by_category.items():
                    st.write(f"**{cat}**")
                    tags_html = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])
                    st.markdown(f'<div class="tag-container">{tags_html}</div><br>', unsafe_allow_html=True)
            else:
                st.info("No predefined skills detected. Check skills block configuration.")

    # PAGE 2: JOB DESCRIPTION MATCHER
    elif page == "🎯 Job Description Matcher":
        st.title("🎯 Job Description Matcher")
        st.write("Compare the resume against a target Job Description to identify keyword gaps and suitability.")

        # Option to load templates
        if jd_templates:
            selected_template = st.selectbox(
                "Or, pre-populate with a template job role:",
                ["-- Custom --"] + [t["title"] for t in jd_templates]
            )
            if selected_template != "-- Custom --":
                template_data = next(t for t in jd_templates if t["title"] == selected_template)
                jd_input = template_data["description"]
            else:
                jd_input = ""
        else:
            jd_input = ""

        jd_text = st.text_area("Paste the Job Description here:", value=jd_input, height=250)

        if st.button("Analyze Compatibility"):
            if not jd_text.strip():
                st.warning("Please paste a Job Description first.")
            else:
                with st.spinner("Analyzing keyword overlaps..."):
                    match_result = tools["matcher"].match(data["text"], jd_text)
                    
                    # Columns
                    match_col1, match_col2 = st.columns(2)
                    with match_col1:
                        # Compatibility Guage
                        sim_gauge = ResumeCharts.create_jd_match_gauge(match_result["similarity_score"])
                        st.plotly_chart(sim_gauge, use_container_width=True)
                        st.write(f"**Keyword Coverage**: {match_result['keyword_coverage']}%")

                    with match_col2:
                        st.subheader("Match Breakdown")
                        for strg in match_result["strengths"]:
                            st.success(strg)
                        for weak in match_result["weaknesses"]:
                            st.warning(weak)

                    # Detailed lists
                    st.markdown("---")
                    col_sk1, col_sk2 = st.columns(2)
                    with col_sk1:
                        st.subheader("✅ Overlapping Skills Found")
                        if match_result["matching_skills"]:
                            tags_html = "".join([f'<span class="skill-tag">{s}</span>' for s in match_result["matching_skills"]])
                            st.markdown(f'<div class="tag-container">{tags_html}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No overlapping skills found in description.")

                    with col_sk2:
                        st.subheader("❌ Missing Role Skills")
                        if match_result["missing_skills"]:
                            tags_html = "".join([f'<span class="skill-tag" style="color: #f43f5e; background-color: rgba(244, 63, 94, 0.1); border-color: rgba(244, 63, 94, 0.2);">{s}</span>' for s in match_result["missing_skills"]])
                            st.markdown(f'<div class="tag-container">{tags_html}</div>', unsafe_allow_html=True)
                        else:
                            st.success("Great job! No missing role skills found.")

                    if match_result["missing_keywords"]:
                        st.subheader("💡 Missing General Keywords")
                        st.caption("Consider integrating these industry keywords into your experience descriptions.")
                        tags_html = "".join([f'<span class="section-tag">{s}</span>' for s in match_result["missing_keywords"]])
                        st.markdown(f'<div class="tag-container">{tags_html}</div>', unsafe_allow_html=True)

    # PAGE 3: DETAILED SECTIONS SCORING
    elif page == "📝 Detailed Sections Scoring":
        st.title("📝 Detailed Resume Sections Scoring")
        st.write("Here is a granular breakdown of the resume components and formatting rules.")

        sec_col1, sec_col2 = st.columns(2)
        with sec_col1:
            radar_fig = ResumeCharts.create_radar_chart(data['ats_score_details']['breakdown'])
            st.plotly_chart(radar_fig, use_container_width=True)

        with sec_col2:
            st.subheader("Granular Score Deductions")
            if data['ats_score_details']['deductions']:
                for deduction in data['ats_score_details']['deductions']:
                    st.error(deduction)
            else:
                st.success("Zero formatting, styling, or section deductions detected!")

        # Load standard rule-based recommendations list
        st.markdown("---")
        st.subheader("Recommended Enhancement Steps")
        recs = RecommendationEngine.generate_recommendations(
            data["text"],
            data["parsed_info"],
            data["page_count"],
            data["skills_res"]["all_extracted"]
        )
        for rec in recs:
            st.markdown(f"- {rec}")

    # PAGE 4: AI CAREER COPILOT
    elif page == "🤖 AI Career Copilot":
        st.title("🤖 AI Career Copilot")
        st.write("Generate customized improvements using Large Language Models.")

        # Verify key credentials
        has_gemini = bool(st.session_state.api_key_gemini)
        has_openai = bool(st.session_state.api_key_openai)

        if not (has_gemini or has_openai):
            st.warning("⚠️ Large Language Model API keys are missing. Configure them in the sidebar's **AI Copilot Settings** to unlock custom reviews.")
            
            # Fallback to local rule-based summary generator
            st.subheader("Local Rule-Based Profile Summary")
            st.info("Using local NLP summarization fallbacks.")
            st.write(f"**Applicant**: {data['name']}")
            st.write(f"**Core skills density**: {len(data['skills_res']['all_extracted'])} parsed items.")
            st.write(f"**Page profile status**: {data['page_count']} page(s).")
            
            st.markdown("---")
            st.subheader("Example Interview Q&A (Local Fallback)")
            st.markdown("""
            **Q: Tell me about your primary technical stack.**
            * *A: Based on your resume, be prepared to outline your experience with projects utilizing: **{}**.*
            """.format(", ".join(data['skills_res']['all_extracted'][:5])))
        else:
            # Setup APIs
            ai_choice = st.selectbox("Choose AI Provider:", ["Google Gemini", "OpenAI"] if has_gemini and has_openai else (["Google Gemini"] if has_gemini else ["OpenAI"]))
            
            prompt_type = st.radio(
                "Choose Task:",
                ["Generate Professional Summary", "Suggest Resume Bullet Points Improvements", "Generate Technical Interview Questions", "Suggest Career Development Paths"]
            )

            # Prompts preparation
            skills_txt = ", ".join(data['skills_res']['all_extracted'])
            base_prompt = f"Resume details for {data['name']}:\n"
            base_prompt += f"Skills found: {skills_txt}\n"
            base_prompt += f"Page count: {data['page_count']}\n"
            base_prompt += f"Resume raw content snippet:\n{data['text'][:3000]}\n\n"

            if prompt_type == "Generate Professional Summary":
                prompt = base_prompt + "Write a compelling, professional, 4-sentence summary highlighting the candidate's core expertise and potential job roles."
            elif prompt_type == "Suggest Resume Bullet Points Improvements":
                prompt = base_prompt + "Suggest 3-4 rewritten resume bullet points that include quantifiable achievements and action-oriented verbs."
            elif prompt_type == "Generate Technical Interview Questions":
                prompt = base_prompt + "Formulate 5 advanced technical interview questions based on the candidate's tech stack and project descriptions, along with brief suggested answers."
            else:
                prompt = base_prompt + "Recommend 3 career paths for this developer, indicating key technologies they should learn next to acquire high-level roles."

            if st.button("Generate Copilot Feedback"):
                with st.spinner("Calling API models..."):
                    try:
                        if ai_choice == "Google Gemini":
                            import google.generativeai as genai
                            genai.configure(api_key=st.session_state.api_key_gemini)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content(prompt)
                            output = response.text
                        else:
                            from openai import OpenAI
                            client = OpenAI(api_key=st.session_state.api_key_openai)
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7
                            )
                            output = response.choices[0].message.content
                        
                        st.subheader("Generated AI Insights")
                        st.write(output)
                    except Exception as e:
                        st.error(f"Error calling AI API services: {e}")
