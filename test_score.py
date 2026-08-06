import pytest
from utils.ats_score import ATSScorer
from utils.jd_match import JDMatcher
from utils.skill_extractor import SkillExtractor

def test_skill_extractor():
    extractor = SkillExtractor()
    text = "Proficient in Python, C++, React and PostgreSQL database systems."
    result = extractor.extract_skills(text)
    
    extracted = result["all_extracted"]
    # Check that skills are extracted correctly
    assert "Python" in extracted
    assert "C++" in extracted
    assert "React" in extracted
    assert "PostgreSQL" in extracted
    
    # Check categories structure
    by_cat = result["by_category"]
    assert "Programming Languages" in by_cat
    assert "Python" in by_cat["Programming Languages"]
    assert "C++" in by_cat["Programming Languages"]

def test_ats_scorer():
    # Construct a high scoring resume text
    perfect_cv = (
        "John Doe. Email: john@example.com. Phone: 123-456-7890. linkedin.com/in/johndoe. github.com/johndoe. "
        "Professional Summary: Dedicated Software Engineer. "
        "Experience: Spearheaded and developed backend architectures. Led a team of engineers to optimize database queries. "
        "Built APIs using Python and Django. Worked from 2021 to 2024. "
        "Education: Bachelor of Science in Computer Science. "
        "Projects: Created an automated CI/CD pipeline using Docker. "
        "Certifications: AWS Certified Solutions Architect."
    )
    skills = ["Python", "Django", "Docker", "Git", "PostgreSQL"]
    
    score_res = ATSScorer.calculate_score(perfect_cv, num_pages=1, extracted_skills=skills)
    
    assert score_res["score"] > 60
    assert len(score_res["strengths"]) > len(score_res["deductions"])

def test_jd_matcher():
    matcher = JDMatcher()
    resume_text = "I am a Python developer with Django and SQL database experience."
    jd_text = "We are seeking a Python developer who knows Django, React, and SQL."
    
    match_res = matcher.match(resume_text, jd_text)
    
    # Check similarity is computed
    assert 0.0 <= match_res["similarity_score"] <= 100.0
    
    # Check skills overlap
    assert "Python" in match_res["matching_skills"]
    assert "Django" in match_res["matching_skills"]
    assert "React" in match_res["missing_skills"]
