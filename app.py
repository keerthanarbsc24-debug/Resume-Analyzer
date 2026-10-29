
import streamlit as st
import torch
import torch.nn as nn
import joblib
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Required for reading uploaded PDF and DOCX files
from pypdf import PdfReader
from docx import Document
from io import BytesIO


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Resume Analyzer")
st.write(
    "Upload your resume, select your target career, "
    "and discover matching skills and skill gaps."
)


# ==========================================
# 2. TARGET JOB SKILL DATABASE
# ==========================================
# These are predefined skills for each target role.
# You can customize them for your project.

JOB_SKILLS = {
    "Data Scientist": [
        "Python",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Deep Learning",
        "Pandas",
        "NumPy",
        "Data Visualization",
        "Scikit-learn",
        "Power BI",
        "Feature Engineering",
        "Model Evaluation"
    ],

    "Data Analyst": [
        "SQL",
        "Excel",
        "Python",
        "Power BI",
        "Tableau",
        "Statistics",
        "Data Cleaning",
        "Data Visualization",
        "Pandas",
        "Business Intelligence"
    ],

    "Machine Learning Engineer": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "Scikit-learn",
        "PyTorch",
        "TensorFlow",
        "Statistics",
        "Feature Engineering",
        "Model Deployment",
        "APIs",
        "Git",
        "Docker"
    ],

    "Software Developer": [
        "Python",
        "Java",
        "Data Structures",
        "Algorithms",
        "SQL",
        "Git",
        "GitHub",
        "Object-Oriented Programming",
        "REST API",
        "Testing"
    ],

    "Business Analyst": [
        "SQL",
        "Excel",
        "Power BI",
        "Tableau",
        "Statistics",
        "Data Visualization",
        "Business Intelligence",
        "Requirements Analysis",
        "Communication"
    ]
}


# ==========================================
# 3. SKILL ALIASES
# ==========================================
# Different terms can refer to the same skill.

SKILL_ALIASES = {
    "Python": ["python"],
    "SQL": ["sql", "structured query language"],
    "Statistics": ["statistics", "statistical analysis"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Visualization": [
        "data visualization",
        "data visualisation"
    ],
    "Scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],
    "Power BI": ["power bi", "powerbi"],
    "Feature Engineering": ["feature engineering"],
    "Model Evaluation": ["model evaluation", "model validation"],
    "Excel": ["excel", "microsoft excel"],
    "Tableau": ["tableau"],
    "Data Cleaning": ["data cleaning", "data preprocessing"],
    "Business Intelligence": [
        "business intelligence",
        "business analytics"
    ],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow"],
    "Model Deployment": ["model deployment", "deployment"],
    "APIs": ["api", "apis", "rest api", "rest apis"],
    "Git": ["git"],
    "Docker": ["docker"],
    "Java": ["java"],
    "Data Structures": [
        "data structures",
        "data structure"
    ],
    "Algorithms": ["algorithms", "algorithm"],
    "GitHub": ["github"],
    "Object-Oriented Programming": [
        "object-oriented programming",
        "oops",
        "oop"
    ],
    "REST API": ["rest api", "rest apis"],
    "Testing": ["software testing", "unit testing", "testing"],
    "Requirements Analysis": [
        "requirements analysis",
        "requirement analysis"
    ],
    "Communication": ["communication", "communication skills"]
}


# ==========================================
# 4. READ RESUME FILES
# ==========================================
def extract_resume_text(uploaded_file):
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension == ".txt":
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

    elif file_extension == ".pdf":
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    elif file_extension == ".docx":
        document = Document(BytesIO(uploaded_file.getvalue()))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
        ]

        return "\n".join(paragraphs)

    else:
        return ""


# ==========================================
# 5. EXTRACT SKILLS FROM RESUME TEXT
# ==========================================
def find_resume_skills(resume_text):
    text = resume_text.casefold()
    found_skills = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            # Match whole phrases/words rather than partial words.
            pattern = (
                r"(?<!\w)"
                + re.escape(alias.casefold())
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                found_skills.add(skill)
                break

    return found_skills


# ==========================================
# 6. DEEP LEARNING MODEL
# ==========================================
class ResumeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_resources():
    device = torch.device("cpu")

    label_encoder = joblib.load("label_encoder.pkl")

    sentence_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model = ResumeClassifier(
        input_size=384,
        hidden_size=128,
        num_classes=len(label_encoder.classes_)
    )

    state_dict = torch.load(
        "resume_classifier.pth",
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, label_encoder, sentence_encoder


# ==========================================
# 7. LOAD MODEL
# ==========================================
try:
    model, label_encoder, sentence_encoder = load_resources()

except Exception as error:
    st.error("The trained model could not be loaded.")
    st.write("Error details:", str(error))
    st.stop()


# ==========================================
# 8. RESUME INPUT
# ==========================================
st.header("1. Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload a resume file",
    type=["pdf", "docx", "txt"]
)

pasted_resume = st.text_area(
    "Or paste your resume text here",
    height=180,
    placeholder="Paste your resume text if you do not want to upload a file."
)

resume_text = ""

if uploaded_file is not None:
    try:
        resume_text = extract_resume_text(uploaded_file)

        if resume_text.strip():
            st.success("Resume text extracted successfully.")

            with st.expander("Preview extracted resume text"):
                st.text(resume_text[:5000])

        else:
            st.warning(
                "No readable text was found. "
                "Try a text-based PDF, DOCX, or TXT file, "
                "or paste the resume text below."
            )

    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")

# Pasted text is used if no readable uploaded resume is available.
if not resume_text.strip():
    resume_text = pasted_resume


# ==========================================
# 9. SELECT TARGET JOB
# ==========================================
st.header("2. Select Your Target Career")

target_role = st.selectbox(
    "Which job role are you preparing for?",
    list(JOB_SKILLS.keys())
)

st.write(
    f"Required skills are based on the project's predefined "
    f"skill list for **{target_role}**."
)

with st.expander("View skills expected for this role"):
    st.write(", ".join(JOB_SKILLS[target_role]))


# ==========================================
# 10. ANALYZE RESUME
# ==========================================
st.header("3. Analyze Your Resume")

if st.button("Analyze Resume", type="primary"):

    if not resume_text.strip():
        st.warning(
            "Please upload a resume or paste your resume text first."
        )

    else:
        with st.spinner("Analyzing your resume..."):

            # Predict the resume category using the trained model.
            embedding = sentence_encoder.encode([resume_text])

            input_tensor = torch.tensor(
                embedding,
                dtype=torch.float32
            )

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                predicted_index = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

            predicted_category = (
                label_encoder.inverse_transform(
                    [predicted_index]
                )[0]
            )

            confidence = probabilities[
                0, predicted_index
            ].item()

            # Find skills mentioned in the resume.
            resume_skills = find_resume_skills(resume_text)

            # Compare them with the selected role's skills.
            required_skills = set(JOB_SKILLS[target_role])

            matched_skills = sorted(
                resume_skills & required_skills
            )

            missing_skills = sorted(
                required_skills - resume_skills
            )

            match_percentage = (
                len(matched_skills) / len(required_skills) * 100
                if required_skills else 0
            )

        # ----------------------------------
        # RESULTS: CATEGORY
        # ----------------------------------
        st.subheader("Career Category Prediction")

        st.success(
            f"Predicted Resume Category: {predicted_category}"
        )

        st.write(
            f"Model confidence: {confidence:.1%}"
        )

        st.caption(
            "The predicted category is based on the categories "
            "learned during model training. It is not a guarantee "
            "of job suitability."
        )

        st.divider()

        # ----------------------------------
        # RESULTS: SKILL MATCH
        # ----------------------------------
        st.subheader(f"Skill Gap Results: {target_role}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Required Skills",
            len(required_skills)
        )

        col2.metric(
            "Matching Skills",
            len(matched_skills)
        )

        col3.metric(
            "Skills Not Found",
            len(missing_skills)
        )

        st.progress(int(round(match_percentage)))

        st.write(
            f"**Skill coverage: {match_percentage:.1f}%**"
        )

        # ----------------------------------
        # MATCHED SKILLS
        # ----------------------------------
        st.subheader("✅ Skills Found in Your Resume")

        if matched_skills:
            for skill in matched_skills:
                st.write(f"✅ {skill}")
        else:
            st.info(
                "No skills from this role's list were identified "
                "in the resume text."
            )

        # ----------------------------------
        # MISSING SKILLS
        # ----------------------------------
        st.subheader("📌 Skills to Consider Learning")

        if missing_skills:
            for skill in missing_skills:
                st.write(f"📌 {skill}")
        else:
            st.success(
                "All skills in this predefined role list "
                "were found in your resume."
            )

        # ----------------------------------
        # RECOMMENDATIONS
        # ----------------------------------
        st.subheader("🎯 Recommendations")

        if missing_skills:
            st.write(
                f"To prepare for a **{target_role}** role, "
                "consider reviewing these skills:"
            )

            for skill in missing_skills:
                st.write(f"• Learn or practise {skill}")

            st.write(
                "Add a skill to your resume only if you genuinely "
                "have that skill or relevant experience."
            )

        else:
            st.write(
                "Your resume mentions all the skills in this "
                "project's predefined list. Continue building "
                "projects and practical experience."
            )

        st.caption(
            "Note: Skill identification uses keyword matching. "
            "A skill not found in the text may still be known "
            "by the candidate. The role skill lists are examples, "
            "not official requirements for every employer."
        )


# ==========================================
# 11. FOOTER
# ==========================================
st.divider()

st.caption(
    "AI Resume Analyzer | Deep Learning and Rule-Based Skill Analysis"
)
