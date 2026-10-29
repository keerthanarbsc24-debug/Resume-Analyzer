
import ast
import re
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "resume_classifier.pth"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
SKILLS_PATH = BASE_DIR / "career_skills.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. DEEP LEARNING MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# 3. LOAD CAREER SKILLS FROM CSV
# --------------------------------------------------

def parse_skills(value):
    """
    Convert a skill-list value from the CSV into a
    Python list. The CSV stores lists as text.
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(skill).strip()
                for skill in parsed
                if str(skill).strip()
            ]

        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []

    except (ValueError, SyntaxError):
        pass

    # Fallback for comma-separated skill values.
    return [
        skill.strip().strip("'\"")
        for skill in value.split(",")
        if skill.strip()
    ]


@st.cache_data
def load_job_skills():

    if not SKILLS_PATH.exists():
        raise FileNotFoundError(
            "career_skills.csv was not found. "
            "Upload it beside app.py in GitHub."
        )

    df = pd.read_csv(SKILLS_PATH)

    required_columns = {
        "category",
        "job_required_skills"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing CSV columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df.dropna(
        subset=["category", "job_required_skills"]
    ).copy()

    df["category"] = (
        df["category"].astype(str).str.strip()
    )

    df = df[df["category"] != ""]

    # Each category has one consolidated skill list
    # in the reduced career_skills.csv file.
    job_skills = {}

    for _, row in df.iterrows():

        category = row["category"]
        skills = parse_skills(row["job_required_skills"])

        # Remove duplicate skills without changing order.
        unique_skills = list(dict.fromkeys(skills))

        if unique_skills:
            job_skills[category] = unique_skills

    if not job_skills:
        raise ValueError(
            "No career categories or skills were found."
        )

    return job_skills


# --------------------------------------------------
# 4. LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model_resources():

    for path in [MODEL_PATH, ENCODER_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path.name}"
            )

    label_encoder = joblib.load(ENCODER_PATH)

    # This is the same pretrained embedding model
    # used to create the training embeddings.
    sentence_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    input_size = 384
    hidden_size = 128
    num_classes = len(label_encoder.classes_)

    model = ResumeClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes
    ).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, label_encoder, sentence_model


# --------------------------------------------------
# 5. EXTRACT TEXT FROM RESUME FILES
# --------------------------------------------------

def extract_resume_text(uploaded_file):

    filename = uploaded_file.name.lower()

    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".txt"):

        return file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

    elif filename.endswith(".pdf"):

        reader = PdfReader(BytesIO(file_bytes))

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    elif filename.endswith(".docx"):

        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
        ]

        return "\n".join(paragraphs)

    else:
        raise ValueError(
            "Unsupported file type. Upload PDF, DOCX, or TXT."
        )


# --------------------------------------------------
# 6. FIND SKILLS IN THE RESUME
# --------------------------------------------------

def normalize_text(text):
    """
    Normalize text for skill matching.
    """

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9+#.]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def skill_found_in_resume(resume_text, skill):

    normalized_resume = normalize_text(resume_text)
    normalized_skill = normalize_text(skill)

    if not normalized_skill:
        return False

    # Escape special characters and match the complete
    # phrase rather than a partial word.
    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_skill)
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, normalized_resume) is not None


def analyze_skill_gap(resume_text, expected_skills):

    matching_skills = []
    missing_skills = []

    for skill in expected_skills:

        if skill_found_in_resume(resume_text, skill):
            matching_skills.append(skill)
        else:
            missing_skills.append(skill)

    total_skills = len(expected_skills)

    coverage = (
        len(matching_skills) / total_skills * 100
        if total_skills > 0
        else 0
    )

    return matching_skills, missing_skills, coverage


# --------------------------------------------------
# 7. PREDICT RESUME CATEGORY
# --------------------------------------------------

def predict_resume_category(
    resume_text,
    model,
    label_encoder,
    sentence_model
):

    embedding = sentence_model.encode(
        [resume_text],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embedding_tensor = torch.tensor(
        embedding,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(embedding_tensor)

        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_index = predicted_index.item()

    predicted_category = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence_score = confidence.item() * 100

    return predicted_category, confidence_score


# --------------------------------------------------
# 8. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("📄 AI-Powered Resume Analyzer")

st.markdown(
    """
    Analyze your resume, explore career requirements,
    identify skills detected in your resume, and find
    areas to improve for your target career.
    """
)

st.info(
    "The predicted resume category comes from your "
    "trained deep-learning model. The selected target "
    "career is used separately for skill-gap analysis."
)


# Load the career skills.
try:
    JOB_SKILLS = load_job_skills()

except Exception as error:
    st.error(f"Could not load career skills: {error}")
    st.stop()


# Resume input.
st.subheader("1. Upload or enter your resume")

uploaded_file = st.file_uploader(
    "Upload a resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, and TXT."
)

resume_text_input = st.text_area(
    "Or paste your resume text here",
    height=220,
    placeholder=(
        "Paste your resume, including your education, "
        "skills, projects, and work experience..."
    )
)


# Career selector from the dataset.
st.subheader("2. Select your target career")

career_options = sorted(JOB_SKILLS.keys())

selected_role = st.selectbox(
    "Which job role are you preparing for?",
    options=career_options,
    format_func=lambda role: role.replace("-", " ").title()
)


# Display expected skills automatically.
expected_skills = JOB_SKILLS[selected_role]

with st.expander(
    f"View skills expected for {selected_role.replace('-', ' ').title()}",
    expanded=True
):

    st.caption(
        "These skills come from the job-required-skills "
        "column of your dataset."
    )

    st.write(", ".join(expected_skills))


# Analyze button.
if st.button("🔍 Analyze Resume", type="primary"):

    resume_text = ""

    # Read uploaded file if one is provided.
    if uploaded_file is not None:

        try:
            resume_text = extract_resume_text(uploaded_file)

        except Exception as error:
            st.error(
                f"Could not read the uploaded resume: {error}"
            )
            st.stop()

    # Pasted text takes priority if provided.
    if resume_text_input.strip():
        resume_text = resume_text_input.strip()

    if not resume_text.strip():

        st.warning(
            "Please upload a resume or paste your resume text."
        )
        st.stop()

    if len(resume_text.strip()) < 30:

        st.warning(
            "The resume text is very short. "
            "Please provide more resume details."
        )
        st.stop()

    # Load the model and analyze.
    try:

        with st.spinner(
            "Loading the model and analyzing your resume..."
        ):

            model, label_encoder, sentence_model = (
                load_model_resources()
            )

            predicted_category, confidence_score = (
                predict_resume_category(
                    resume_text,
                    model,
                    label_encoder,
                    sentence_model
                )
            )

            matching_skills, missing_skills, coverage = (
                analyze_skill_gap(
                    resume_text,
                    expected_skills
                )
            )

    except Exception as error:

        st.error(
            "Analysis failed. Please check that the model "
            "files and required libraries are available."
        )

        st.exception(error)
        st.stop()


    # ----------------------------------------------
    # 9. DISPLAY RESULTS
    # ----------------------------------------------

    st.divider()
    st.header("3. Resume Analysis Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Resume Category",
            str(predicted_category).replace(
                "-", " "
            ).title()
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence_score:.1f}%"
        )

    st.caption(
        "Confidence is the model's predicted probability, "
        "not a guarantee that the classification is correct."
    )


    # Skill coverage.
    st.subheader(
        f"4. Skill Gap Analysis — "
        f"{selected_role.replace('-', ' ').title()}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Expected Skills",
            len(expected_skills)
        )

    with col2:

        st.metric(
            "Matching Skills",
            len(matching_skills)
        )

    with col3:

        st.metric(
            "Skills Not Found",
            len(missing_skills)
        )

    st.write("**Skill coverage**")

    st.progress(int(round(coverage)))

    st.write(f"{coverage:.1f}% of expected skills detected")


    # Matching skills.
    st.subheader("✅ Matching Skills")

    if matching_skills:

        st.write(", ".join(matching_skills))

    else:

        st.info(
            "No expected skills were detected by keyword matching."
        )


    # Missing skills.
    st.subheader("📌 Skills Not Found in Your Resume")

    if missing_skills:

        st.write(", ".join(missing_skills))

    else:

        st.success(
            "All expected skills were detected in your resume."
        )


    # Recommendations.
    st.subheader("💡 Recommendations")

    if missing_skills:

        st.write(
            "Consider reviewing these skills for your target "
            "career and adding relevant evidence to your resume "
            "if you genuinely have those skills:"
        )

        for skill in missing_skills[:10]:

            st.markdown(f"- **{skill}**")

        if len(missing_skills) > 10:

            st.caption(
                f"{len(missing_skills) - 10} additional "
                "skills were not found."
            )

    else:

        st.write(
            "Your resume mentions all the expected skills "
            "in the selected dataset entry. Consider adding "
            "specific projects or achievements to demonstrate "
            "your experience."
        )


    # Explain limitations.
    st.divider()

    st.caption(
        "Note: Skill matching is based on text keywords. "
        "A skill marked 'not found' may still be a skill you "
        "possess if it is described differently in your resume. "
        "The selected role's skills are based on the available "
        "dataset and may not represent every employer's "
        "requirements."
    )


# Footer.
st.divider()

st.caption(
    "AI Resume Analyzer | Deep Learning Classification "
    "+ Dataset-Based Skill Gap Analysis"
)
