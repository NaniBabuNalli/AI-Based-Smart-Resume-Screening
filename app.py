import os
from flask import Flask, render_template, request, session, send_file
import pandas as pd
import spacy
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = "uploads"

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Predefined skill set
skill_keywords = {
    # Programming Languages
    "python", "java", "c++", "c#", "sql", "javascript", "typescript", "ruby", "go", "rust",
    
    # Web Development
    "html", "css", "bootstrap", "tailwind", "react", "angular", "vue.js", "node.js", "express.js",
    
    # Databases
    "mysql", "postgresql", "mongodb", "firebase", "oracle", "redis", "cassandra",
    
    # Machine Learning & AI
    "machine learning", "deep learning", "artificial intelligence", "data science",
    "computer vision", "natural language processing", "reinforcement learning",
    
    # ML & Data Science Tools
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
    "seaborn", "huggingface", "openai", "llms", "generative ai", 
    
    # DevOps & Cloud
    "devops", "docker", "kubernetes", "linux", "bash", "shell scripting",
    "aws", "azure", "gcp", "terraform", "ansible", "jenkins", "git", "github", "bitbucket",
    
    # Data Engineering & Big Data
    "big data", "hadoop", "spark", "data engineering", "etl", "databricks",
    
    # Cybersecurity
    "cybersecurity", "penetration testing", "network security", "firewalls",
    
    # Software Testing
    "software testing", "selenium", "junit", "pytest", "robot framework",
    
    # Business Intelligence & Analytics
    "tableau", "power bi", "qlikview", "looker", "snowflake",
    
    # APIs & Backend
    "rest api", "graphql", "fastapi", "flask", "django", "spring boot", "asp.net",
    
    # Other Tech Skills
    "ci/cd", "microservices", "serverless", "blockchain", "web3", "metaverse"
}

# Ensure 'uploads' folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Store screened resumes globally
screened_resumes = []

def extract_skills(text):
    """Extract skills from text using predefined skill matching."""
    doc = nlp(text.lower())
    return list(set(token.text for token in doc if token.text in skill_keywords))

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip().lower()

def calculate_weighted_match_score(resume_skills, job_skills, skill_weights):
    """Calculate weighted similarity score using Cosine Similarity."""
    if not resume_skills or not job_skills:
        return 0.0

    # Create weighted vectors
    resume_vector = []
    job_vector = []

    for skill in job_skills:
        weight = skill_weights.get(skill, 1)  # Default weight is 1 if not specified
        resume_vector.append(weight if skill in resume_skills else 0)
        job_vector.append(weight)

    # Calculate cosine similarity
    similarity = cosine_similarity([resume_vector], [job_vector])[0][0]
    return round(similarity * 100, 2)

def extract_job_details(job_description):
    """Extract job details like title, experience, location, and salary from job description."""
    job_details = {
        "title": "N/A",
        "experience": "N/A",
        "location": "N/A",
        "salary": "N/A"
    }

    # Extract job title (first line)
    title_match = re.search(r"^[^\n]+", job_description)
    if title_match:
        job_details["title"] = title_match.group(0).strip()

    # Extract experience (e.g., "3+ years of experience")
    experience_pattern = r"(\d+\+?\s*(years?|yrs?)\s*(of\s*experience)?)"
    experience_match = re.search(experience_pattern, job_description, re.IGNORECASE)
    if experience_match:
        job_details["experience"] = experience_match.group(0).strip()

    # Extract location (e.g., "New York, NY" or "Remote")
    location_pattern = r"(remote|hybrid|onsite|[\w\s]+,\s*[A-Z]{2})"
    location_match = re.search(location_pattern, job_description, re.IGNORECASE)
    if location_match:
        job_details["location"] = location_match.group(0).strip()

    # Extract salary range (e.g., "$80,000 - $100,000")
    salary_pattern = r"(\$\d{1,3}(,\d{3})*\s*-\s*\$\d{1,3}(,\d{3})*)"
    salary_match = re.search(salary_pattern, job_description)
    if salary_match:
        job_details["salary"] = salary_match.group(0).strip()

    return job_details

def extract_candidate_name(text):
    """Extract candidate name using NLP."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "N/A"

def extract_contact_info(text):
    """Extract email and phone number from resume text."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"

    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)

    return {
        "email": email.group(0) if email else "N/A",
        "phone": phone.group(0) if phone else "N/A"
    }

def extract_education(text):
    """Extract education details from resume text."""
    education_keywords = ["bachelor", "master", "phd", "degree", "diploma", "university", "college"]
    education = []
    for sent in text.split("\n"):
        if any(keyword in sent.lower() for keyword in education_keywords):
            education.append(sent.strip())
    return education if education else ["N/A"]

def extract_experience(text):
    """Extract experience details from resume text."""
    experience_keywords = ["experience", "worked", "intern", "job", "role"]
    experience = []
    for sent in text.split("\n"):
        if any(keyword in sent.lower() for keyword in experience_keywords):
            experience.append(sent.strip())
    return experience if experience else ["N/A"]

def extract_projects(text):
    """Extract projects from resume text."""
    project_keywords = ["project", "developed", "built", "created"]
    projects = []
    for sent in text.split("\n"):
        if any(keyword in sent.lower() for keyword in project_keywords):
            projects.append(sent.strip())
    return projects if projects else ["N/A"]

def extract_years_of_experience(text):
    """Extract years of experience from text."""
    experience_pattern = r"(\d+)\s*(years?|yrs?)"
    match = re.search(experience_pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def extract_candidate_experience(text):
    """Extract candidate's years of experience from resume text."""
    experience_pattern = r"(\d+)\s*(years?|yrs?)"
    matches = re.findall(experience_pattern, text, re.IGNORECASE)
    if matches:
        return max(int(match[0]) for match in matches)
    return 0

def extract_education_requirements(text):
    """Extract education requirements from job description."""
    education_keywords = ["bachelor", "master", "phd", "degree", "diploma", "university", "college"]
    education_requirements = []
    for sent in text.split("\n"):
        if any(keyword in sent.lower() for keyword in education_keywords):
            education_requirements.append(sent.strip())
    return education_requirements if education_requirements else ["N/A"]

def compare_education(candidate_education, job_education):
    """Compare candidate's education with job's education requirements."""
    if not candidate_education or not job_education:
        return "N/A"

    # Check if any of the candidate's education matches the job's education requirements
    for edu in candidate_education:
        if any(req.lower() in edu.lower() for req in job_education):
            return "Yes"
    return "No"

def calculate_statistics(screened_resumes):
    """Calculate statistics for the dashboard."""
    total_resumes = len(screened_resumes)
    suitable_resumes = sum(1 for res in screened_resumes if res["suitable"] == "Yes")
    unsuitable_resumes = total_resumes - suitable_resumes
    suitable_percentage = round((suitable_resumes / total_resumes) * 100, 2) if total_resumes > 0 else 0
    unsuitable_percentage = round((unsuitable_resumes / total_resumes) * 100, 2) if total_resumes > 0 else 0
    average_match_score = round(sum(res["match_score"] for res in screened_resumes) / total_resumes, 2) if total_resumes > 0 else 0

    # Calculate top skills lacking
    skills_lacking = {}
    for res in screened_resumes:
        for skill in res["skills_lacking"]:
            skills_lacking[skill] = skills_lacking.get(skill, 0) + 1
    top_skills_lacking = sorted(skills_lacking.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_resumes": total_resumes,
        "suitable_percentage": suitable_percentage,
        "unsuitable_percentage": unsuitable_percentage,
        "average_match_score": average_match_score,
        "top_skills_lacking": top_skills_lacking
    }

@app.route("/", methods=["GET", "POST"])
def home():
    global screened_resumes

    if request.method == "POST":
        if "job_description" in request.form:
            job_description = request.form.get("job_description", "").strip().lower()
            if job_description:
                session["job_description"] = job_description
                session["job_skills"] = extract_skills(job_description)
                job_details = extract_job_details(job_description)
                session["job_title"] = job_details["title"]
                session["job_experience"] = job_details["experience"]
                session["job_location"] = job_details["location"]
                session["job_salary"] = job_details["salary"]
                session["job_experience_years"] = extract_years_of_experience(job_details["experience"])
                session["job_education"] = extract_education_requirements(job_description)

                # Initialize skill weights (default weight is 1)
                session["skill_weights"] = {skill: 1 for skill in session["job_skills"]}

        elif "resume_pdf" in request.files:
            files = request.files.getlist("resume_pdf")
            for file in files:
                if not file or not file.filename.endswith(".pdf"):
                    continue

                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)

                resume_text = extract_text_from_pdf(file_path)
                resume_skills = extract_skills(resume_text)

                job_skills = session.get("job_skills", [])
                skill_weights = session.get("skill_weights", {})
                match_score = calculate_weighted_match_score(resume_skills, job_skills, skill_weights)
                suitability = "Yes" if match_score >= 50 else "No"

                # Calculate skills lacking
                skills_lacking = list(set(job_skills) - set(resume_skills))
                candidate_experience_years = extract_candidate_experience(resume_text)

                # Check if candidate meets experience requirement
                job_experience_years = session.get("job_experience_years", 0)
                experience_met = "Yes" if candidate_experience_years >= job_experience_years else "No"
                
                candidate_name = extract_candidate_name(resume_text)
                contact_info = extract_contact_info(resume_text)
                projects = extract_projects(resume_text)
                candidate_experience = extract_experience(resume_text)
                candidate_experience = candidate_experience if candidate_experience else ["N/A"]  # Handle empty experience field if present
                
                # Check if candidate meets education requirement
                education = extract_education(resume_text)
                education = education if education else ["N/A"]  # Handle empty education field if present
                
                job_education = session.get("job_education", ["N/A"])
                education_met = compare_education(education, job_education)

                screened_resumes.append({
                    "slno": len(screened_resumes) + 1,
                    "name": candidate_name,
                    "email": contact_info["email"],
                    "phone": contact_info["phone"],
                    "education": education,
                    "education_met": education_met,
                    "experience": candidate_experience,
                    "candidate_experience_years": candidate_experience_years,
                    "projects": projects,
                    "skills": resume_skills,
                    "suitable": suitability,
                    "match_score": match_score,
                    "skills_lacking": skills_lacking,
                    "experience_met": experience_met
                })

                os.remove(file_path)

    # Calculate statistics for the dashboard
    statistics = calculate_statistics(screened_resumes)

    return render_template(
        "index.html",
        job_description=session.get("job_description", ""),
        job_skills=session.get("job_skills", []),
        job_title=session.get("job_title", "N/A"),
        job_experience=session.get("job_experience", "N/A"),
        job_location=session.get("job_location", "N/A"),
        job_salary=session.get("job_salary", "N/A"),
        skill_weights=session.get("skill_weights", {}),
        screened_resumes=screened_resumes,
        statistics=statistics
    )

@app.route("/update-weights", methods=["POST"])
def update_weights():
    if request.method == "POST":
        skill_weights = {}
        for skill in session.get("job_skills", []):
            weight = request.form.get(f"weight_{skill}", "1")
            skill_weights[skill] = int(weight)
        session["skill_weights"] = skill_weights
    return "Weights updated successfully."

@app.route("/download-report")
def download_report():
    global screened_resumes
    df = pd.DataFrame(screened_resumes)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="screened_resumes.csv"
    )

@app.route("/clear-resumes")
def clear_resumes():
    global screened_resumes
    screened_resumes = []
    return "Cleared all resumes."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)