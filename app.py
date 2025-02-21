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

def calculate_match_score(resume_skills, job_skills):
    """Calculate similarity score using Cosine Similarity."""
    if not resume_skills or not job_skills:
        return 0.0

    vectorizer = CountVectorizer().fit_transform([" ".join(resume_skills), " ".join(job_skills)])
    vectors = vectorizer.toarray()

    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
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
    experience_match = re.search(r"(\d+\+?\s*(years?|yrs?)\s*of\s*experience)", job_description, re.IGNORECASE)
    if experience_match:
        job_details["experience"] = experience_match.group(0).strip()

    # Extract location (e.g., "New York, NY" or "Remote")
    location_match = re.search(r"(remote|hybrid|onsite|[\w\s]+,\s*[A-Z]{2})", job_description, re.IGNORECASE)
    if location_match:
        job_details["location"] = location_match.group(0).strip()

    # Extract salary range (e.g., "$80,000 - $100,000")
    salary_match = re.search(r"(\$\d{1,3}(,\d{3})*\s*-\s*\$\d{1,3}(,\d{3})*)", job_description)
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

        elif "resume_pdf" in request.files:
            files = request.files.getlist("resume_pdf")
            for file in files:
                if not file or not file.filename.endswith(".pdf"):
                    continue

                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)

                resume_text = extract_text_from_pdf(file_path)
                resume_skills = extract_skills(resume_text)

                # Extract additional resume details
                candidate_name = extract_candidate_name(resume_text)
                contact_info = extract_contact_info(resume_text)
                education = extract_education(resume_text)
                experience = extract_experience(resume_text)
                projects = extract_projects(resume_text)
                candidate_experience_years = extract_candidate_experience(resume_text)

                job_skills = session.get("job_skills", [])
                match_score = calculate_match_score(resume_skills, job_skills)
                suitability = "Yes" if match_score >= 50 else "No"

                # Calculate skills lacking
                skills_lacking = list(set(job_skills) - set(resume_skills))

                # Check if candidate meets experience requirement
                job_experience_years = session.get("job_experience_years", 0)
                experience_met = "Yes" if candidate_experience_years >= job_experience_years else "No"

                screened_resumes.append({
                    "slno": len(screened_resumes) + 1,
                    "name": candidate_name,
                    "email": contact_info["email"],
                    "phone": contact_info["phone"],
                    "education": education,
                    "experience": experience,
                    "candidate_experience_years": candidate_experience_years,
                    "projects": projects,
                    "skills": resume_skills,
                    "suitable": suitability,
                    "match_score": match_score,
                    "skills_lacking": skills_lacking,
                    "experience_met": experience_met
                })

                os.remove(file_path)

    return render_template(
        "index.html",
        job_description=session.get("job_description", ""),
        job_skills=session.get("job_skills", []),
        job_title=session.get("job_title", "N/A"),
        job_experience=session.get("job_experience", "N/A"),
        job_location=session.get("job_location", "N/A"),
        job_salary=session.get("job_salary", "N/A"),
        screened_resumes=screened_resumes
    )

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