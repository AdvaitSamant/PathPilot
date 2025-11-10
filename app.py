import streamlit as st
import numpy as np
import joblib
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
import time as t
import pandas as pd
import random
import os
from datetime import datetime
import glob

# For Icons
st.markdown("""
<script src="https://unpkg.com/lucide@latest"></script>
<script>
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
});
</script>
""", unsafe_allow_html=True)

def render_header(title, icon_name):
    st.markdown(
        f"""
        <div class="icon-header">
            <span class="material-icon">{icon_name}</span>
            <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------
# Initialize persistent session variables
# -------------------------
if "user_data" not in st.session_state:
    st.session_state.user_data = None

if "predicted_career" not in st.session_state:
    st.session_state.predicted_career = None

if "career_path" not in st.session_state:
    st.session_state.career_date = None


# -------------------------
# Load trained model & encoder
# -------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("career_model_main.pkl")          # Your RandomForest model
    label_encoder = joblib.load("label_encoder.pkl") # Your label encoder for 'Role'
    return model, label_encoder

model, le = load_assets()

def save_user_to_csv(user_info, scores, predicted_career, file_path="user_results.csv"):
    import pandas as pd
    import os

    # Build a single-row dataframe with user info + quiz scores + prediction
    data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Full Name": user_info.get("name", ""),
        "Age": user_info.get("age", ""),
        "Gender": user_info.get("gender", ""),
        "City": user_info.get("city", ""),
        "State": user_info.get("state", ""),
        "Country": user_info.get("country", ""),
        "Career Goal": user_info.get("goal", ""),
        "Hobbies": user_info.get("hobbies", ""),
        "Email": user_info.get("email", ""),
        "CGPA": user_info.get("cgpa", ""),
        "Predicted Career": predicted_career,
    }

    # Add quiz section scores
    for key, val in scores.items():
        data[key] = val if val is not None else ""

    # Convert to DataFrame
    new_entry = pd.DataFrame([data])

    # Append to CSV (create if not exists)
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        # Update if same name already exists (overwrite old record)
        if user_info.get("name", "") in existing["Full Name"].values:
            existing.loc[existing["Full Name"] == user_info["name"], :] = new_entry.values[0]
            existing.to_csv(file_path, index=False)
        else:
            new_entry.to_csv(file_path, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(file_path, index=False)


# -------------------------
# Streamlit Page Config
# -------------------------


with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Quiz", "Chatbot"],
        icons=["house", "list-task", "chat"],
        menu_icon="cast",
        default_index=0,
    )
if selected == "Home":
    # -------------------------
    # Config
    # -------------------------
    st.set_page_config(page_title="PathPilot", page_icon="🎯", layout="wide")

    # -------------------------
    # Helper for Lottie Animation
    # -------------------------
    def load_lottie_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_ai = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_cwqf5i6h.json")

    # -------------------------
    # UI
    # -------------------------
    st.markdown("""
    <style>
    /* === Mint Green Theme for Home Page === */

    /* --- Main Title --- */
    .big-title {
        font-size: 48px !important;
        font-weight: 800;
        text-align: center;
        color: #66BB6A; /* Soft mint green */
        text-shadow: 0px 0px 10px rgba(129, 199, 132, 0.4);
    }

    /* --- Subtext --- */
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #9E9E9E; /* Subtle soft gray */
        margin-bottom: 60px;
    }

    /* --- Feature Boxes (Dark Base with Mint Accents) --- */
    .feature-box {
        background-color: #1A1A1A;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #81C784; /* Mint accent line */
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 0.3s ease-in-out;
    }

    /* --- Hover Glow --- */
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0px 6px 18px rgba(129, 199, 132, 0.35);
        border-left: 5px solid #66BB6A;
        background: linear-gradient(180deg, #222222 0%, #1A1A1A 100%);
    }

    /* --- Feature Box Headings --- */
    .feature-box h4 {
        color: #A5D6A7; /* Light mint for titles */
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* --- Feature Box Text --- */
    .feature-box p {
        color: #CFCFCF; /* Soft readable gray */
        font-size: 16px;
        line-height: 1.5;
    }

    /* --- Responsive polish --- */
    @media (max-width: 768px) {
        .big-title {
            font-size: 36px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


    # -------------------------
    # Title & Animation
    # -------------------------
    st.markdown('<p class="big-title">PathPilot – Navigate Your Future with AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Discover your perfect tech career through AI-powered insights</p>', unsafe_allow_html=True)

    def load_lottie_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            st.warning("⚠️ Failed to load Lottie animation.")
            return None
        return r.json()

    # Load the “Wonder Things” animation
    lottie_wonder = load_lottie_url("https://lottie.host/b2fc68c4-c4ec-4c44-86f9-5ae6b6e59359/JHQbN7Vvq5.json")

    # -------------------------
    # Layout section
    # -------------------------
    col1, col2 = st.columns([1, 1])

    with col1:
        if lottie_wonder:
            st_lottie(
                lottie_wonder,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=350,
                key="wonder_things"
            )
        else:
            st.warning("Animation unavailable 😔")

    with col2:
        st.markdown("""
            ### Welcome to **PathPilot – Your AI Career Mentor**

            Discover where **your potential** can take you with **PathPilot**.
            Using intelligent, data-driven analysis, we help you uncover the career path that fits you best.

            #### What you can do here:
            - Take a quick **Skill & Personality Quiz**
            - Get your **AI-Predicted Career Path**
            - Chat with your **Personal AI Career Mentor**
            - Explore **Insights, Salaries, and Growth Opportunities**

            No endless forms. No generic advice.  
            Just your skills, your potential and **AI that actually gets you.**
            """)

    st.markdown("---")

    # -------------------------
    # Features Section
    # -------------------------
    st.subheader("Why Choose PathPilot?")
    st.markdown("""
    <style>
    /* === Mint Green Edition: Clean, Modern, and Professional === */

    .feature-row {
        display: flex;
        gap: 20px;
        justify-content: center;
        align-items: stretch;
        flex-wrap: wrap;
        margin-top: 20px;
    }

    /* --- Feature Boxes --- */
    .feature-box {
        background-color: #ffffff;
        padding: 24px 20px;
        border-radius: 15px;
        flex: 1;
        min-width: 280px;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease-in-out;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border-left: 5px solid #81C784; /* Mint accent */
    }

    /* --- Hover Glow --- */
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(129, 199, 132, 0.25);
        background: linear-gradient(180deg, #ffffff 0%, #f3fff5 100%);
        border-left: 5px solid #66BB6A;
    }

    /* --- Headings --- */
    .feature-box h4 {
        color: #388E3C;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
        font-size: 18px;
    }

    /* --- Paragraph Text --- */
    .feature-box p {
        color: #4f5952;
        text-align: center;
        font-size: 15px;
        line-height: 1.5;
        flex-grow: 1;
    }

    /* --- Divider Accent --- */
    .feature-box::after {
        content: "";
        display: block;
        width: 60%;
        height: 2px;
        background: linear-gradient(to right, #81C784, transparent);
        margin: 10px auto 0 auto;
        border-radius: 2px;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='feature-row'>
        <div class='feature-box'>
            <h4>Skill Evaluation</h4>
            <p>Answer concise, real-world questions and see your technical and soft skills scored instantly on a 1–10 scale.</p>
        </div>
        <div class='feature-box'>
            <h4>AI Career Prediction</h4>
            <p>PathPilot’s AI model analyzes your strengths to recommend ideal roles — from Data Scientist to Cybersecurity Specialist.</p>
        </div>
        <div class='feature-box'>
            <h4>Smart Career Mentor</h4>
            <p>Chat with an AI mentor trained on your quiz results — get personalized learning paths, tools, and career insights.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------
    # Navigation Hint
    # -------------------------
    st.info("""
    ### Getting Started with PathPilot
    1. Head to the **Quiz** tab and complete the 3-question skill test in each section.  
    2. Hit **Predict My Career** to see your AI-powered career suggestion.  
    3. Jump into the **Chatbot** tab to explore advice, learning resources, and your personalized roadmap.  

    **Pro Tip:** All your progress is saved automatically — no resets, no re-entry.
    """)

    st.markdown("---")
    st.caption("By **Students**, For **Students**, Driven By **PathPilot**.")


elif selected == "Quiz":
    # -----------------------------
    # PAGE SETUP
    # -----------------------------
    st.set_page_config(page_title="PathPilot | AI Career Path Skill Quiz", page_icon="🎯", layout="wide")

    st.markdown("""
    <script>
        window.scrollTo({top: 0, behavior: 'smooth'});
    </script>
    """, unsafe_allow_html=True)

    st.title("Discover Your AI-Driven Career Path")
    st.markdown("""
    <p style='text-align:center; font-size:18px; color:#555;'>
    Answer 3 quick questions per category.<br>
    Your answers will be automatically scored <b>(1–10)</b> based on accuracy!
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .divider {
        width: 60%;
        height: 2px;
        margin: 25px auto 30px auto;
        border-radius: 2px;
        background: linear-gradient(90deg, #66BB6A, #A5D6A7, #66BB6A);
        background-size: 200% 100%;
        animation: glowLine 3s ease-in-out infinite;
    }

    @keyframes glowLine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>

    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # CSS
    # -----------------------------
    st.markdown("""
    <style>
    /* === Light Mode Only — Mint Green Edition === */

    /* --- Question Cards --- */
    .question-card {
        background: #ffffff;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        padding: 15px 20px;
        margin-bottom: 25px;
        transition: all 0.25s ease;
        border-left: 5px solid #81C784;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .question-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(129, 199, 132, 0.3);
    }

    /* --- Question Text --- */
    .question-text {
        font-size: 18px;
        font-weight: 600;
        color: #2b2b2b;
        margin-bottom: 15px;
    }

    /* --- Radio Buttons --- */
    .stRadio > div {
        background: #f9f9f9;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 6px;
        transition: background 0.2s ease, transform 0.15s ease;
        border: 1px solid #eaeaea;
    }
    .stRadio > div:hover {
        background: #f1fff3;
        transform: scale(1.02);
    }
    .stRadio label {
        font-size: 16px !important;
        color: #333 !important;
        font-weight: 500 !important;
    }

    /* --- Selected Option with Mint Glow --- */
    .stRadio [aria-checked="true"] {
        background: linear-gradient(90deg, #A5D6A7, #81C784);
        color: white !important;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(129, 199, 132, 0.4);
        animation: pulseMint 2s infinite ease-in-out;
    }

    /* --- Smooth Pulse Animation --- */
    @keyframes pulseMint {
        0% { box-shadow: 0 0 6px rgba(129, 199, 132, 0.3); }
        50% { box-shadow: 0 0 16px rgba(129, 199, 132, 0.6); }
        100% { box-shadow: 0 0 6px rgba(129, 199, 132, 0.3); }
    }

    /* --- Headers --- */
    h1, h2, h3, h4 {
        text-align: center !important;
        color: #66BB6A !important;
    }

    /* --- Quiz Section Card --- */
    .quiz-section {
        background-color: #ffffff;
        padding: 30px 40px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(129, 199, 132, 0.15);
        margin: 30px auto;
        width: 80%;
    }

    /* --- Subtext --- */
    .subtext {
        text-align: center;
        font-size: 15px;
        color: #4f5952;
        margin-bottom: 20px;
    }

    /* --- Divider --- */
    hr {
        margin-top: 25px;
        margin-bottom: 25px;
        border: 0;
        height: 2px;
        background: linear-gradient(to right, #81C784, transparent);
    }

    /* --- Buttons --- */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #A5D6A7, #81C784);
        color: white;
        border: none;
        padding: 0.75em 2em;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0px 4px 12px rgba(129, 199, 132, 0.4);
        transition: all 0.3s ease-in-out;
        display: block;
        margin: 0 auto;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 16px rgba(129, 199, 132, 0.6);
        background: linear-gradient(90deg, #81C784, #A5D6A7);
    }
    div.stButton > button:first-child:active {
        transform: scale(0.98);
    }
    </style>
    """, unsafe_allow_html=True)



    

    # -----------------------------
    # DATA CLEANING
    # -----------------------------
    def clean_csv_files(folder_path="quiz_data"):
        for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
            try:
                df = pd.read_csv(file_path)
                # Drop completely empty rows
                df.dropna(how="all", inplace=True)

                # Trim whitespace and normalize text
                df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else "")

                # Replace "nan" strings or empty placeholders with actual blanks
                df.replace(["nan", "NaN", "None", " "], "", inplace=True)

                # Drop rows missing core fields
                df = df[df["question"].str.len() > 0]
                df = df[df["answer"].str.len() > 0]

                # Replace missing options with "-"
                for col in ["option1", "option2", "option3"]:
                    df[col] = df[col].apply(lambda x: x if len(str(x)) > 0 else "—")

                df.to_csv(file_path, index=False)
                print(f"✅ Cleaned: {file_path}")
            except Exception as e:
                print(f"⚠️ Error cleaning {file_path}: {e}")


    # -----------------------------
    # SCORING
    # -----------------------------
    def calculate_score(correct_answers, user_answers):
        correct_count = sum([1 for i, ans in enumerate(user_answers) if ans == correct_answers[i]])
        return [1, 4, 7, 10][correct_count]

    # -----------------------------
    # QUESTION HANDLER
    # -----------------------------
    def ask_questions(category, icon_name, csv_file):
        # Category title with Lucide icon
        icons = {
        "memory": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <rect x='2' y='6' width='20' height='12' rx='2'/><path d='M6 10h0'/><path d='M10 10h0'/><path d='M14 10h0'/><path d='M18 10h0'/></svg>""",

        "code": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <polyline points='16 18 22 12 16 6'/><polyline points='8 6 2 12 8 18'/></svg>""",

        "calendar_check": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <rect x='3' y='4' width='18' height='18' rx='2'/><path d='M16 2v4'/><path d='M8 2v4'/><path d='M3 10h18'/><path d='m9 16 2 2 4-4'/></svg>""",

        "message_square": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/></svg>""",

        "globe_2": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <circle cx='12' cy='12' r='10'/><path d='M2 12h20'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>""",

        "target": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg>""",

        "party_popper": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <path d='M2 22 10 10l6 6L2 22z'/><path d='M14 4c0 1 1 2 2 2s2-1 2-2-1-2-2-2'/><path d='M22 10c0 1-1 2-2 2s-2-1-2-2 1-2 2-2'/><path d='M16 6l6-2'/></svg>""",

        "handshake": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <path d='M4 12h2l3 3 3-3 3 3 3-3h2'/><path d='M2 9l4 3h12l4-3'/></svg>""",

        "heart_pulse": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <path d='M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 0 0-7.8 7.8l8.8 8.8 8.8-8.8a5.5 5.5 0 0 0 0-7.8z'/><path d='M12 8l-2 4h3l-1 4 2-4h-3z'/></svg>""",

        "mic": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <rect x='9' y='2' width='6' height='11' rx='3'/><path d='M5 10a7 7 0 0 0 14 0'/><path d='M12 19v3'/></svg>""",

        "refresh_cw": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <polyline points='23 4 23 10 17 10'/><polyline points='1 20 1 14 7 14'/><path d='M3.51 9a9 9 0 0 1 14.13-3.36L23 10'/><path d='M20.49 15A9 9 0 0 1 6.36 18.36L1 14'/></svg>""",

        "trophy": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <path d='M8 21h8v2H8z'/><path d='M12 17a5 5 0 0 0 5-5V3H7v9a5 5 0 0 0 5 5z'/><path d='M4 3h16v2H4z'/><path d='M4 5v2a3 3 0 0 0 3 3h0'/><path d='M20 5v2a3 3 0 0 1-3 3h0'/></svg>""",

        "trending_up": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <polyline points='23 6 13.5 15.5 8.5 10.5 1 18'/><polyline points='17 6 23 6 23 12'/></svg>""",

        "sun": """<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26' fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
        <circle cx='12' cy='12' r='5'/><path d='M12 1v2'/><path d='M12 21v2'/><path d='M4.22 4.22l1.42 1.42'/><path d='M18.36 18.36l1.42 1.42'/><path d='M1 12h2'/><path d='M21 12h2'/><path d='M4.22 19.78l1.42-1.42'/><path d='M18.36 5.64l1.42-1.42'/></svg>"""
    }

        icon_svg = icons.get(icon_name, "")
        st.markdown(f"<h3 style='text-align:center;'>{icon_svg} {category}</h3>", unsafe_allow_html=True)


        if not os.path.exists(csv_file):
            st.warning(f"⚠️ Missing CSV file: {csv_file}")
            return None

        # Read CSV safely and drop useless rows
        df = pd.read_csv(csv_file, dtype=str)
        df = df.dropna(how="all")  # remove empty rows
        df = df.fillna("")  # replace nan with empty string
        df = df[df["question"].str.strip() != ""]  # only rows with valid questions
        df = df[df["answer"].str.strip() != ""]  # only rows with valid answers

        # Shuffle & sample
        if f"{category}_sampled" not in st.session_state:
            st.session_state[f"{category}_sampled"] = df.sample(
                n=min(3, len(df)),
                random_state=random.randint(1, 999999)  # true random each session
            )
        df = st.session_state[f"{category}_sampled"]

        user_answers, correct_answers = [], []

        for i, row in df.iterrows():
            # Skip question if completely empty
            if not row["question"].strip():
                continue

            question = row["question"].strip()

            # Filter options
            options = [opt.strip() for opt in [row.get("option1", ""), row.get("option2", ""), row.get("option3", ""), row.get("answer", "")]
                    if opt and opt.lower() != "nan"]

            # Remove duplicates
            options = list(dict.fromkeys(options))

            # Keep shuffle consistent using session state
            unique_key = f"{category}_{i}_options"
            if unique_key not in st.session_state:
                random.shuffle(options)
                st.session_state[unique_key] = options
            else:
                options = st.session_state[unique_key]

            st.markdown(
                f"""
                <div style='
                    background:#fff;
                    border-radius:12px;
                    padding:15px;
                    margin:10px 0;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);
                    font-size:17px;
                    line-height:1.5;
                '><b>{question}</b></div>
                """, unsafe_allow_html=True)

            user_ans = st.radio("", options, key=f"{category}_{i}", index=None)
            user_answers.append(user_ans)
            correct_answers.append(row["answer"].strip())

        # If user answered all questions
        if all(ans is not None for ans in user_answers):
            return calculate_score(correct_answers, user_answers)
        return None

    # -----------------------------
    # USER INFO
    # -----------------------------
    
    st.title("Tell Us About Yourself")
    st.markdown("""
    <p style='text-align:center; font-size:17px; color:#444; line-height:1.6;'>
    Provide a few quick details so PathPilot can personalize your career insights. <br>
    Your information is processed locally and never shared externally.
    </p>
    """, unsafe_allow_html=True)

    if "user_info" not in st.session_state:
        st.session_state["user_info"] = {}
    

    with st.form("user_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="Advait Samant")
            age = st.number_input("Age", min_value=10, max_value=100, step=1)
            email = st.text_input("Email Address", placeholder="name@example.com")
            goal = st.text_input("Career Goal", placeholder="e.g., Become a Data Scientist")
            df = pd.read_csv("csv_files/UGC Universities.csv")
            institutions = df["Name"].tolist()

            # Provide searchable dropdown
            college = st.selectbox(
                "Current College / University",
                options=institutions,
                index=0,
                help="Start typing to search the name of your institution",
            )

            # If you want to allow "other" entry as well:
            if st.checkbox("My college is not listed"):
                college = st.text_input("Enter your College / University manually", placeholder="e.g., XYZ University")
        with col2:
            city = st.text_input("City", placeholder="Pune")
            state = st.text_input("State", placeholder="Maharashtra")
            country = st.text_input("Country", placeholder="India", value="India")
            hobbies = st.text_area("Hobbies / Interests", placeholder="e.g., Coding, Gaming, Reading, Traveling")
            cgpa = st.text_input("CGPA / Academic Score", placeholder="e.g., 7.8 / 9.0")

        submitted = st.form_submit_button("💾 Save My Info")
        if submitted:
            st.session_state["user_info"] = {
                "name": name.strip(),
                "age": age,
                "college": college.strip(),
                "city": city.strip(),
                "state": state.strip(),
                "country": country.strip(),
                "hobbies": hobbies.strip(),
                "goal": goal.strip(),
                "email": email.strip(),
                "cgpa": cgpa.strip()
            }
            st.success("✅ Info saved successfully!")
            st.balloons()
    st.markdown("""
    <style>
    .divider {
        width: 60%;
        height: 2px;
        margin: 25px auto 30px auto;
        border-radius: 2px;
        background: linear-gradient(90deg, #66BB6A, #A5D6A7, #66BB6A);
        background-size: 200% 100%;
        animation: glowLine 3s ease-in-out infinite;
    }

    @keyframes glowLine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>

    <div class="divider"></div>
    """, unsafe_allow_html=True)
    # -----------------------------
    # QUIZ CATEGORIES
    # -----------------------------
    
    categories = [
        ("Computer Architecture", "memory"),     
        ("Programming Skills", "code"),              
        ("Project Management", "calendar_check"),       
        ("Communication Skills", "message_square"),     
        ("Openness", "globe_2"),                        
        ("Conscientiousness", "target"),              
        ("Extraversion", "party_popper"),            
        ("Agreeableness", "handshake"),                
        ("Emotional Range", "heart_pulse"),          
        ("Conversational Skills", "mic"),               
        ("Openness to Change", "refresh_cw"),           
        ("Hedonism", "trophy"),                          
        ("Self-enhancement", "trending_up"),            
        ("Self-transcendence", "sun"),                  
    ]

    quiz_folder = "quiz_data"
    scores = {}

    for name, emoji in categories:
        csv_path = f"{quiz_folder}/{name.lower().replace(' ', '_')}.csv"
        scores[name] = ask_questions(name, emoji, csv_path)
        st.markdown("""
        <style>
        .divider {
            width: 60%;
            height: 2px;
            margin: 25px auto 30px auto;
            border-radius: 2px;
            background: linear-gradient(90deg, #66BB6A, #A5D6A7, #66BB6A);
            background-size: 200% 100%;
            animation: glowLine 3s ease-in-out infinite;
        }

        @keyframes glowLine {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        </style>

        <div class="divider"></div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    # Load model + label encoder safely
    try:
        model = joblib.load("career_model_main.pkl")
        le = joblib.load("label_encoder.pkl")
    except:
        model = None
        le = None
        st.warning("⚠️ Model files not found. Please upload 'career_model.pkl' and 'label_encoder.pkl'.")

    if st.button("🔮 Predict My Career"):
        if not model or not le:
            st.error("Model not loaded. Upload the model files first.")
        elif all(score is not None for score in scores.values()):
            st.balloons()
            user_data = pd.DataFrame([{
                'Computer Architecture': scores["Computer Architecture"],
                'Programming Skills': scores["Programming Skills"],
                'Project Management': scores["Project Management"],
                'Communication skills': scores["Communication Skills"],
                'Openness': scores["Openness"],
                'Conscientiousness': scores["Conscientiousness"],
                'Extraversion': scores["Extraversion"],
                'Agreeableness': scores["Agreeableness"],
                'Emotional_Range': scores["Emotional Range"],
                'Conversation': scores["Conversational Skills"],
                'Openness to Change': scores["Openness to Change"],
                'Hedonism': scores["Hedonism"],
                'Self-enhancement': scores["Self-enhancement"],
                'Self-transcendence': scores["Self-transcendence"]
            }])

            st.success("✅ Quiz Completed Successfully!")
            st.dataframe(user_data)

            for col in model.feature_names_in_:
                if col not in user_data.columns:
                    user_data[col] = 0

            user_data = user_data[model.feature_names_in_]
            prediction = model.predict(user_data)
            predicted_career = le.inverse_transform(prediction)[0]
            st.session_state.predicted_career = predicted_career
            st.session_state.user_data = user_data
            # Save all user info + results to CSV
            user_info = st.session_state.get("user_info", {})
            save_user_to_csv(user_info, scores, predicted_career)

            if predicted_career == 0:
                career_path = "AI ML Specialist"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **AI ML Specialist**!")
                st.markdown("---")
                st.info("""
                **AI/ML Specialist (India)** 🇮🇳

                An AI/ML Specialist designs and develops machine learning models to solve real-world problems. They handle data preprocessing, model training, algorithm tuning, deployment, and monitoring. They also work with developers to integrate AI into applications and communicate insights to business teams.

                **Key Responsibilities:**
                - Build and train ML/AI models (classification, NLP, vision, etc.)
                - Clean and preprocess datasets
                - Optimize model performance and accuracy
                - Deploy models into production (MLOps, APIs, pipelines)
                - Research new algorithms and AI techniques

                **Required Skills:**
                - Python, Java, SQL/MySQL
                - Libraries: TensorFlow, PyTorch, scikit-learn
                - Data handling and visualization
                - Strong math & statistics background
                - Cloud and MLOps tools (AWS, GCP, Docker)

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹5–10 LPA
                - Mid-level (3–6 yrs): ₹10–22 LPA
                - Senior/Expert: ₹20–50+ LPA
                *(Glassdoor average: ₹23.7 LPA)*

                **Career Growth:**
                AI/ML → Senior ML Engineer → Data Scientist → AI Architect / Research Scientist

                **Why It’s a Great Choice:**
                AI/ML jobs are in huge demand across India. Specialists with strong portfolios and hands-on experience can land top internships and high-paying roles. It’s the perfect field for techies who love solving problems with data, math, and creativity.
                """)
            elif predicted_career == 1:
                career_path = "API Specialist"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **API Specialist**!")
                st.markdown("---")
                st.info("""
                **API Specialist (India)** 🔌

                An API Specialist builds the superhighways of data between apps — making sure services talk to each other smoothly, safely and fast. They design, develop, maintain and monitor APIs, handle integrations, document endpoints, work with security/authentication, and collaborate across front-end/back-end/devops teams.

                **Key Responsibilities:**
                - Design & implement APIs (REST, GraphQL, SOAP etc) for app communication  
                - Manage versioning, rate-limiting, security (OAuth2, JWT), latency & scalability  
                - Document APIs clearly (so other devs don’t wanna pull their hair)  
                - Integrate with third-party services and microservices architecture  
                - Monitor usage, troubleshoot errors, optimise performance  
                - Collaborate with frontend/devops to deploy and manage API ecosystems  

                **Required Skills:**
                - Strong programming: Java, Node.js, Python  
                - Deep understanding of HTTP, requests/responses, JSON/XML, REST/GraphQL  
                - Database savvy (SQL/MySQL + maybe NoSQL)  
                - Knowledge of API tools: Postman, Swagger/OpenAPI spec, API gateways  
                - Basics of DevOps & cloud deployment (because APIs live in the wild)  
                - Good documentation and teamwork skills  

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹3.5-9 LPA (based on API developer data) 
                - Mid-level (~2-5 yrs): ~ ₹10-18 LPA
                - Senior/Lead/API Architect: ~ ₹18-25 LPA+ (and if you’re top tier, way more)

                **Career Growth Path:**
                API Specialist → API Developer/Integration Engineer → API Architect/Lead → Head of API Development or Solutions Architect  

                **Why It’s a Smart Move :**
                - Modern apps rely on APIs, so demand’s strong (especially in product companies & fintech).  
                - You’re already strong in Java & MySQL — those backend skills apply directly.  
                - Combine this role with your AI/ML interests and you can build “intelligent APIs” (yep, even crazier value).  
                - Interfacing between components, integrating systems — those kinds of “connect the dots” skills make you stand out.  
                """)

            elif predicted_career == 2:
                career_path = "Application Support Engineer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Application Support Engineer**!")
                st.markdown("---")
                st.info("""
                **Application Support Engineer (India)** 🛠️

                An Application Support Engineer ensures that software applications run smoothly and efficiently. They troubleshoot issues, provide technical support, and collaborate with development teams to implement fixes and improvements.

                **Key Responsibilities:**
                - Monitor application performance and resolve issues
                - Provide technical support to users and stakeholders
                - Collaborate with development teams to implement fixes and enhancements
                - Document issues and solutions for future reference
                - Participate in on-call support rotation as needed

                **Required Skills:**
                - Strong problem-solving and analytical skills
                - Familiarity with application servers, databases, and cloud platforms
                - Basic programming/scripting skills (Python, Bash, etc.)
                - Excellent communication and teamwork abilities
                - Experience with monitoring and logging tools (e.g., Splunk, ELK stack)

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹3-6 LPA
                - Mid-level (~2-5 yrs): ~ ₹6-12 LPA
                - Senior/Lead: ~ ₹12-20 LPA

                **Career Growth Path:**
                Application Support Engineer → Senior Application Support Engineer → Application Support Manager → Director of Application Support

                **Why It’s a Smart Move :**
                - High demand for skilled support engineers as companies rely on complex applications.
                - Opportunity to work closely with development teams and gain insights into the software development lifecycle.
                - Potential to transition into more specialized roles (e.g., DevOps, Site Reliability Engineering) with additional skills.
                """)

            elif predicted_career == 3:
                career_path = "Business Analyst"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Business Analyst**!")
                st.markdown("---")
                st.info("""
                **Business Analyst (India)** 📊

                A Business Analyst bridges the gap between IT and the business. They analyze business needs, document requirements, and help implement solutions that drive business value.

                **Key Responsibilities:**
                - Gather and document business requirements
                - Analyze data and processes to identify improvement opportunities
                - Collaborate with stakeholders to design and implement solutions
                - Facilitate communication between business and technical teams
                - Support project management activities

                **Required Skills:**
                - Strong analytical and problem-solving skills
                - Excellent communication and interpersonal abilities
                - Familiarity with business process modeling and analysis
                - Basic understanding of IT systems and software development
                - Proficiency in data analysis tools (Excel, SQL, etc.)

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹3-6 LPA
                - Mid-level (~2-5 yrs): ~ ₹6-12 LPA
                - Senior/Lead: ~ ₹12-20 LPA

                **Career Growth Path:**
                Business Analyst → Senior Business Analyst → Business Analysis Manager → Director of Business Analysis

                **Why It’s a Smart Move :**
                - High demand for skilled business analysts as companies seek to improve efficiency and drive growth.
                - Opportunity to work on diverse projects and gain insights into various business functions.
                - Potential to transition into more specialized roles (e.g., Product Management, Project Management) with additional skills.
                """)

            elif predicted_career == 4:
                career_path = "Customer Service Executive"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Customer Service Executive**!")
                st.markdown("---")
                st.info("""
                **Customer Service Executive (India)** 📞

                A Customer Service Executive (CSE) is responsible for handling customer inquiries, resolving issues, and providing information about products and services. They play a crucial role in ensuring customer satisfaction and loyalty.

                **Key Responsibilities:**
                - Respond to customer inquiries via phone, email, or chat
                - Resolve customer issues and complaints in a timely manner
                - Provide product information and support to customers
                - Document customer interactions and feedback
                - Collaborate with other teams to improve customer experience

                **Required Skills:**
                - Excellent communication and interpersonal skills
                - Strong problem-solving abilities
                - Patience and empathy when dealing with customers
                - Basic computer skills and familiarity with CRM software
                - Ability to work in a fast-paced environment

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹2.5-4 LPA
                - Mid-level (~2-5 yrs): ~ ₹4-6 LPA
                - Senior/Lead: ~ ₹6-10 LPA

                **Career Growth Path:**
                Customer Service Executive → Senior Customer Service Executive → Customer Service Manager → Director of Customer Service

                **Why It’s a Smart Move :**
                - High demand for customer service professionals as companies prioritize customer experience.
                - Opportunity to develop strong communication and problem-solving skills.
                - Potential to transition into other roles (e.g., Sales, Marketing) with experience.
                """)

            elif predicted_career == 5:
                career_path = "Cyber Security Specialist"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Cyber Security Specialist**!")
                st.markdown("---")
                st.info("""
                **Cyber Security Specialist (India)** 🔒

                A Cyber Security Specialist is responsible for protecting an organization's computer systems and networks from cyber threats. They implement security measures, monitor for suspicious activity, and respond to security incidents.

                **Key Responsibilities:**
                - Develop and implement security policies and procedures
                - Monitor networks for security breaches and vulnerabilities
                - Respond to security incidents and conduct investigations
                - Collaborate with IT teams to secure systems and applications
                - Stay updated on the latest cyber threats and security trends

                **Required Skills:**
                - Strong knowledge of network security protocols and technologies
                - Experience with security tools (firewalls, intrusion detection systems)
                - Familiarity with compliance standards (ISO 27001, GDPR)
                - Excellent problem-solving and analytical skills
                - Relevant certifications (CISSP, CEH, CompTIA Security+) are a plus

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹4-8 LPA
                - Mid-level (~2-5 yrs): ~ ₹8-15 LPA
                - Senior/Lead: ~ ₹15-30 LPA

                **Career Growth Path:**
                Cyber Security Specialist → Senior Cyber Security Specialist → Cyber Security Manager → Director of Cyber Security

                **Why It’s a Smart Move :**
                - Increasing frequency and sophistication of cyber attacks drives demand for security professionals.
                - Opportunity to work with cutting-edge technologies and protect critical assets.
                - Potential to specialize in areas like Threat Intelligence, Incident Response, or Security Architecture with additional skills.
                """)

            elif predicted_career == 6:
                career_path = "Database Administrator"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Database Administrator**!")
                st.markdown("---")
                st.info("""
                **Database Administrator (India)** 🗄️

                A Database Administrator (DBA) is responsible for managing and maintaining an organization's databases. They ensure the availability, performance, and security of databases while also implementing backup and recovery strategies.

                **Key Responsibilities:**
                - Install, configure, and upgrade database management systems
                - Monitor database performance and optimize queries
                - Implement security measures to protect sensitive data
                - Perform regular backups and disaster recovery testing
                - Collaborate with developers to design and optimize database schemas

                **Required Skills:**
                - Strong knowledge of database management systems (Oracle, SQL Server, MySQL)
                - Proficiency in SQL and database query optimization
                - Experience with database backup and recovery techniques
                - Familiarity with database security best practices
                - Relevant certifications (Oracle DBA, Microsoft SQL Server) are a plus

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹4-8 LPA
                - Mid-level (~2-5 yrs): ~ ₹8-15 LPA
                - Senior/Lead: ~ ₹15-30 LPA

                **Career Growth Path:**
                Database Administrator → Senior Database Administrator → Database Manager → Director of Database Administration

                **Why It’s a Smart Move :**
                - Growing importance of data management and security in organizations.
                - Opportunity to work with advanced database technologies and architectures.
                - Potential to specialize in areas like Data Warehousing, Big Data, or Cloud Databases with additional skills.
                """)

            elif predicted_career == 7:
                career_path = "Graphics Designer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Graphics Designer**!")
                st.markdown("---")
                st.info("""
                **Graphics Designer (India)** 🎨

                A Graphics Designer creates visual content to communicate messages and ideas. They work with various design tools and software to produce graphics for print and digital media.

                **Key Responsibilities:**
                - Develop visual concepts and designs for marketing materials
                - Create logos, brochures, and other branding elements
                - Collaborate with clients and stakeholders to understand design needs
                - Stay updated on design trends and software tools
                - Prepare files for print and digital production

                **Required Skills:**
                - Proficiency in design software (Adobe Creative Suite, CorelDRAW)
                - Strong understanding of color theory, typography, and layout design
                - Excellent creativity and artistic skills
                - Ability to work under tight deadlines and manage multiple projects
                - Strong communication and collaboration abilities

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹3-6 LPA
                - Mid-level (~2-5 yrs): ~ ₹6-12 LPA
                - Senior/Lead: ~ ₹12-20 LPA

                **Career Growth Path:**
                Graphics Designer → Senior Graphics Designer → Art Director → Creative Director

                **Why It’s a Smart Move :**
                - Growing demand for visual content in marketing and advertising.
                - Opportunity to work on diverse projects across industries.
                - Potential to specialize in areas like UI/UX Design, Motion Graphics, or 3D Design with additional skills.
                """)

            elif predicted_career == 8:
                career_path = "Hardware Engineer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Hardware Engineer**!")
                st.markdown("---")
                st.info("""
                **Hardware Engineer (India)** 💻

                A Hardware Engineer designs, develops, and tests computer hardware components and systems. They work on various hardware technologies, including processors, circuit boards, and memory devices.

                **Key Responsibilities:**
                - Design and develop hardware components (PCBs, processors, etc.)
                - Test and validate hardware designs for performance and reliability
                - Collaborate with software engineers to integrate hardware and software
                - Troubleshoot and resolve hardware issues
                - Stay updated on emerging hardware technologies and trends

                **Required Skills:**
                - Strong knowledge of electronics and circuit design
                - Proficiency in hardware description languages (VHDL, Verilog)
                - Experience with simulation and testing tools
                - Excellent problem-solving and analytical skills
                - Relevant certifications (Cisco, CompTIA A+) are a plus

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹4-8 LPA
                - Mid-level (~2-5 yrs): ~ ₹8-15 LPA
                - Senior/Lead: ~ ₹15-30 LPA

                **Career Growth Path:**
                Hardware Engineer → Senior Hardware Engineer → Hardware Architect → Director of Hardware Engineering

                **Why It’s a Smart Move :**
                - Growing demand for hardware engineers in various industries.
                - Opportunity to work on cutting-edge hardware technologies.
                - Potential to specialize in areas like Embedded Systems, IoT, or Robotics with additional skills.
                """)

            elif predicted_career == 9:
                career_path = "Helpdesk Engineer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Helpdesk Engineer**!")
                st.markdown("---")
                st.info("""
                **Helpdesk Engineer (India)** 🛠️

                A Helpdesk Engineer provides technical support and assistance to end-users and organizations. They troubleshoot hardware and software issues, resolve technical problems, and ensure smooth IT operations.

                **Key Responsibilities:**
                - Respond to user inquiries and provide technical support
                - Troubleshoot hardware and software issues
                - Install and configure computer systems and applications
                - Maintain IT documentation and user manuals
                - Collaborate with IT teams to resolve complex issues

                **Required Skills:**
                - Strong knowledge of computer hardware and software
                - Excellent communication and interpersonal skills
                - Problem-solving and analytical thinking abilities
                - Familiarity with helpdesk ticketing systems
                - Relevant certifications (CompTIA A+, ITIL) are a plus

                **Salary in India:**
                - Entry-level (~0-2 yrs): ~ ₹3-6 LPA
                - Mid-level (~2-5 yrs): ~ ₹6-12 LPA
                - Senior/Lead: ~ ₹12-20 LPA

                **Career Growth Path:**
                Helpdesk Engineer → Senior Helpdesk Engineer → IT Support Manager → Director of IT Services

                **Why It’s a Smart Move :**
                - Growing demand for IT support professionals in various industries.
                - Opportunity to work with diverse technologies and systems.
                - Potential to specialize in areas like Cybersecurity, Cloud Computing, or Network Administration with additional skills.
                """)

            elif predicted_career == 10:
                career_path = "Information Security Specialist"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Information Security Specialist**!")
                st.markdown("---")
                st.info("""
                **Information Security Specialist (India)** 🔐

                An Information Security Specialist protects an organization's data, networks, and systems from cyber threats. They design security measures, monitor for attacks, investigate breaches, and ensure compliance with data protection regulations.

                **Key Responsibilities:**
                - Develop and implement security policies and procedures  
                - Monitor networks and systems for security breaches  
                - Perform vulnerability assessments and penetration tests  
                - Manage firewalls, antivirus, and encryption tools  
                - Respond to security incidents and investigate potential threats  
                - Ensure compliance with cybersecurity standards and regulations  
                - Conduct employee training on security awareness  

                **Required Skills:**
                - Strong understanding of networking, operating systems, and cybersecurity principles  
                - Knowledge of firewalls, IDS/IPS, VPNs, and encryption techniques  
                - Familiarity with ethical hacking and vulnerability assessment tools  
                - Understanding of risk management and incident response  
                - Proficiency in scripting languages like Python, Bash, or PowerShell  
                - Certifications such as CEH, CISSP, CompTIA Security+, or CISM are a plus  

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹4–9 LPA  
                - Mid-level (3–6 yrs): ₹10–18 LPA  
                - Senior-level (7+ yrs): ₹20–35+ LPA  

                **Career Growth Path:**
                Information Security Specialist → Security Engineer → Security Architect → Chief Information Security Officer (CISO)

                **Why It’s a Great Career:**
                As cyber threats rise globally, skilled security professionals are in high demand. Companies across industries need experts who can safeguard digital assets, detect vulnerabilities, and ensure system integrity. It’s a challenging but rewarding field that offers strong job stability and career growth.
                """)

            elif predicted_career == 11:
                career_path = "Network Engineer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Network Engineer**!")
                st.markdown("---")
                st.info("""
                **Network Engineer (India)** 🌐

                A Network Engineer designs, builds, and maintains the communication networks that keep organizations connected. They ensure seamless data flow, secure connections, and reliable network performance across local and wide-area networks.

                **Key Responsibilities:**
                - Design and implement LAN, WAN, and wireless networks  
                - Configure and maintain routers, switches, and firewalls  
                - Monitor network performance and troubleshoot connectivity issues  
                - Ensure network security through access control and encryption  
                - Perform regular network maintenance and upgrades  
                - Collaborate with IT teams to support infrastructure scalability  
                - Document network configurations and procedures  

                **Required Skills:**
                - Strong knowledge of networking protocols (TCP/IP, DNS, DHCP, OSPF, BGP)  
                - Hands-on experience with Cisco, Juniper, or similar networking equipment  
                - Understanding of firewalls, VPNs, and network security practices  
                - Familiarity with network monitoring tools like Wireshark or SolarWinds  
                - Ability to diagnose and resolve connectivity and performance issues  
                - Certifications such as CCNA, CCNP, or CompTIA Network+ are a plus  

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹3–6 LPA  
                - Mid-level (3–6 yrs): ₹7–15 LPA  
                - Senior-level (7+ yrs): ₹15–25+ LPA  

                **Career Growth Path:**
                Network Engineer → Senior Network Engineer → Network Architect → Network Manager / IT Infrastructure Lead  

                **Why It’s a Great Career:**
                Network Engineers are the backbone of modern IT infrastructure. With the growth of cloud computing, IoT, and enterprise connectivity, skilled professionals are in high demand. It’s a stable, high-impact role offering solid technical experience and excellent career progression.
                """)

            elif predicted_career == 12:
                career_path = "Project Manager"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Project Manager**!")
                st.markdown("---")
                st.info("""
                **Project Manager (India)** 📋

                A Project Manager oversees the planning, execution, and completion of projects within an organization. They coordinate between teams, manage resources, track progress, and ensure that goals are achieved on time and within budget.

                **Key Responsibilities:**
                - Define project scope, goals, and deliverables  
                - Develop detailed project plans and timelines  
                - Allocate resources and assign responsibilities to team members  
                - Monitor progress, manage risks, and handle issues proactively  
                - Communicate project updates to stakeholders and management  
                - Ensure quality standards and deadlines are met  
                - Evaluate project outcomes and implement improvements  

                **Required Skills:**
                - Strong leadership, organization, and communication skills  
                - Proficiency in project management tools (Jira, Trello, Asana, MS Project)  
                - Understanding of Agile, Scrum, and Waterfall methodologies  
                - Ability to manage budgets, risks, and cross-functional teams  
                - Problem-solving and decision-making under pressure  
                - Certifications such as PMP, PRINCE2, or Certified Scrum Master (CSM) are a plus  

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹5–10 LPA  
                - Mid-level (3–6 yrs): ₹10–20 LPA  
                - Senior-level (7+ yrs): ₹20–35+ LPA  

                **Career Growth Path:**
                Project Coordinator → Project Manager → Senior Project Manager → Program Manager / Project Director  

                **Why It’s a Great Career:**
                Project Managers play a critical role in delivering success across industries. With strong leadership and organizational skills, they ensure smooth collaboration between teams and timely completion of projects. The role offers diverse challenges, strategic impact, and excellent growth potential.
                """)

            elif predicted_career == 13:
                career_path = "Software Developer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Software Developer**!")
                st.markdown("---")
                st.info("""
                **Software Developer (India)** 👨‍💻

                A Software Developer designs, codes, tests, and maintains software applications. They work with various programming languages and frameworks to build solutions that meet user needs and business requirements.

                **Key Responsibilities:**
                - Write clean, efficient, and maintainable code  
                - Collaborate with cross-functional teams to define software requirements  
                - Test and debug applications to ensure functionality and performance  
                - Participate in code reviews and contribute to best practices  
                - Stay updated on emerging technologies and industry trends  
                - Document software design and development processes  

                **Required Skills:**
                - Proficiency in programming languages (Java, Python, C++, etc.)  
                - Familiarity with software development methodologies (Agile, Scrum)  
                - Experience with version control systems (Git, SVN)  
                - Strong problem-solving and analytical skills  
                - Ability to work collaboratively in a team environment  
                - Knowledge of databases and web technologies is a plus  

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹3–8 LPA  
                - Mid-level (3–6 yrs): ₹8–15 LPA  
                - Senior-level (7+ yrs): ₹15–30+ LPA  

                **Career Growth Path:**
                Junior Developer → Software Developer → Senior Developer → Tech Lead / Software Architect  

                **Why It’s a Great Career:**
                Software Developers are in high demand as technology continues to evolve. The role offers opportunities to work on innovative projects, solve complex problems, and contribute to impactful solutions. With continuous learning and skill development, it’s a rewarding career path with strong growth potential.
                """)

            elif predicted_career == 14:
                career_path = "Software Tester"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Software Tester**!")
                st.markdown("---")
                st.info("""
                **Software Tester (India)** 🧩

                A Software Tester ensures that applications and systems work flawlessly before they reach users. They identify bugs, verify fixes, and validate that software meets quality standards and user expectations.

                **Key Responsibilities:**
                - Review software requirements and prepare test plans  
                - Design, execute, and maintain test cases for manual and automated testing  
                - Identify, report, and track software bugs and performance issues  
                - Collaborate with developers to resolve defects and retest fixes  
                - Perform regression, integration, and performance testing  
                - Ensure applications meet functionality, usability, and reliability standards  
                - Document test results and prepare detailed reports for stakeholders  

                **Required Skills:**
                - Strong understanding of software development and testing life cycles (SDLC & STLC)  
                - Knowledge of testing tools like Selenium, JIRA, TestRail, or Postman  
                - Familiarity with automation frameworks and scripting languages (Python, Java, etc.)  
                - Analytical thinking and attention to detail  
                - Understanding of Agile and DevOps environments  
                - Certifications such as ISTQB or CSTE are a plus  

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹3–6 LPA  
                - Mid-level (3–6 yrs): ₹7–12 LPA  
                - Senior-level (7+ yrs): ₹12–20+ LPA  

                **Career Growth Path:**
                Software Tester → QA Engineer → Test Lead → QA Manager / Test Architect  

                **Why It’s a Great Career:**
                Software Testers are the guardians of quality in the tech world. They ensure smooth user experiences and reliable products. With the rise of automation, AI testing, and continuous integration, skilled testers are more in demand than ever — offering stable careers and plenty of growth opportunities.
                """)

            elif predicted_career == 15:
                career_path = "Techinical Writer"
                st.session_state.career_path = career_path
                st.success(f"🎯 Based on your responses, your ideal career path is **Techinical Writer**!")
                st.markdown("---")
                st.info("""
                **Technical Writer (India)** ✍️

                A Technical Writer creates clear, concise, and accurate documentation for software, hardware, and other technical products. They bridge the gap between complex technical concepts and easy-to-understand information for users and developers.

                **Key Responsibilities:**
                - Create and maintain user manuals, API documentation, and developer guides  
                - Collaborate with engineers, designers, and product teams to gather information  
                - Simplify technical jargon into accessible and structured content  
                - Maintain consistency in documentation style and formatting  
                - Review and update documents based on product changes or new features  
                - Work with content management systems and documentation tools  
                - Ensure accuracy, clarity, and adherence to company standards  

                **Required Skills:**
                - Excellent written and verbal communication skills  
                - Strong understanding of technical concepts, software, and APIs  
                - Familiarity with documentation tools like Markdown, Confluence, or Swagger  
                - Basic knowledge of HTML, XML, or Markdown formatting  
                - Ability to collaborate effectively with cross-functional teams  
                - Attention to detail and consistency in writing style  

                **Salary in India:**
                - Entry-level (0–2 yrs): ₹3–6 LPA  
                - Mid-level (3–6 yrs): ₹7–12 LPA  
                - Senior-level (7+ yrs): ₹12–20+ LPA  

                **Career Growth Path:**
                Junior Technical Writer → Technical Writer → Senior Technical Writer → Documentation Manager / Content Strategist  

                **Why It’s a Great Career:**
                Technical Writers play a key role in making complex systems understandable. With growing demand for user-friendly software and API documentation, skilled writers are highly valued. The role combines creativity with technical expertise and offers opportunities across multiple industries.
                """)

            # ==========================
            # PDF GENERATION LOGIC (Fixed)
            # ==========================
            from fpdf import FPDF
            import base64
            import re
            from datetime import datetime

            # ============ SAFE TEXT CLEANER ============
            def safe_text(text):
                """Ensure all text is safe for FPDF and properly encoded."""
                text = str(text) if text is not None else ""
                try:
                    return text.encode("latin-1").decode("latin-1")
                except UnicodeEncodeError:
                    return re.sub(r'[^\x00-\x7F]+', '', text)

            # ============ PDF CLASS ============
            class PDF(FPDF):
                def header(self):
                    # Header shown on each page
                    self.set_font("Helvetica", "B", 14)
                    self.set_text_color(56, 142, 60)
                    self.cell(0, 10, safe_text("PathPilot – AI Career Insight Report"), ln=True, align="C")
                    self.ln(5)

                def footer(self):
                    # Page footer
                    self.set_y(-15)
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(100, 100, 100)
                    self.cell(0, 10, f"Page {self.page_no()}", align="C")

                def section_title(self, title):
                    self.set_font("Helvetica", "B", 12)
                    self.set_text_color(76, 175, 80)
                    self.cell(0, 8, safe_text(title), ln=True)
                    self.ln(3)

                def section_body(self, body):
                    self.set_font("Helvetica", "", 11)
                    self.set_text_color(33, 33, 33)
                    self.multi_cell(0, 7, safe_text(body))
                    self.ln(3)

            # ============ DETAILED MEANINGS ============
            section_meanings = {
                "Computer Architecture": """What it measures:
            Understanding of how computers function at the hardware level — CPU, memory, I/O, and data flow.

            Why it matters:
            Helps you excel in roles like embedded systems, AI infrastructure, or low-level optimization.""",

                "Programming Skills": """What it measures:
            Ability to write efficient, logical, and maintainable code in Python, Java, or C++.

            Why it matters:
            Essential for developers, ML engineers, and backend specialists where logic and clarity rule.""",

                "Project Management": """What it measures:
            Skill in planning, scheduling, and delivering projects effectively.

            Why it matters:
            Critical for roles like Project Manager, Product Lead, and Team Coordinator.""",

                "Communication skills": """What it measures:
            Clarity, confidence, and active listening when sharing or presenting ideas.

            Why it matters:
            Strong communication enables collaboration, leadership, and smooth teamwork.""",

                "Openness": """What it measures:
            Curiosity, creativity, and willingness to explore new concepts.

            Why it matters:
            Encourages innovation — vital in research, design, and AI fields.""",

                "Conscientiousness": """What it measures:
            Responsibility, organization, and consistency.

            Why it matters:
            Predicts reliability and success in structured tech environments.""",

                "Extraversion": """What it measures:
            Confidence and sociability in group or leadership settings.

            Why it matters:
            Useful for leadership, sales engineering, and team collaboration.""",

                "Agreeableness": """What it measures:
            Empathy, kindness, and teamwork orientation.

            Why it matters:
            Enhances collaboration, ideal for customer support or HR-linked roles.""",

                "Emotional_Range": """What it measures:
            Emotional control and ability to handle stress.

            Why it matters:
            Calmness and clarity during challenges help in management and cybersecurity roles.""",

                "Conversation": """What it measures:
            Engagement and communication fluency in discussions.

            Why it matters:
            Strong conversationalists thrive in interviews, consulting, and teamwork.""",

                "Openness to Change": """What it measures:
            Adaptability to new tools, ideas, or challenges.

            Why it matters:
            A must for AI, DevOps, and product innovation roles.""",

                "Hedonism": """What it measures:
            Drive for enjoyment and satisfaction in work.

            Why it matters:
            Inspires creativity — valuable in design, startups, and creative tech sectors.""",

                "Self-enhancement": """What it measures:
            Ambition, recognition-seeking, and personal growth.

            Why it matters:
            Fuels leadership and continuous learning — perfect for entrepreneurs.""",

                "Self-transcendence": """What it measures:
            Ethical awareness and community-focused mindset.

            Why it matters:
            Ideal for AI ethics, healthcare tech, and sustainability-oriented careers."""
            }

            # ============ STREAMLIT REPORT ============
            if st.session_state.get("predicted_career"):
                user_info = st.session_state.get("user_info", {})
                predicted_career = st.session_state.predicted_career
                user_data = st.session_state.user_data

                pdf = PDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)

                # --- USER INFO ---
                pdf.section_title("User Information")
                pdf.section_body(
                    f"Name: {user_info.get('name', 'N/A')}\n"
                    f"Age: {user_info.get('age', 'N/A')}\n"
                    f"Location: {user_info.get('city', '')}, {user_info.get('state', '')}, {user_info.get('country', '')}\n"
                    f"Career Goal: {user_info.get('goal', 'N/A')}\n"
                    f"Hobbies: {user_info.get('hobbies', 'N/A')}\n"
                )

                # --- SCORES ---
                pdf.section_title("Section Scores & Insights")
                for col, val in user_data.iloc[0].items():
                    meaning = section_meanings.get(col, "No description available.")
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(46, 125, 50)
                    pdf.cell(0, 7, safe_text(f"{col}: {val}/10"), ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(70, 70, 70)
                    pdf.multi_cell(0, 6, safe_text(meaning))
                    pdf.ln(4)

                # --- PREDICTED CAREER ---
                pdf.section_title("Predicted Career Path")
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(0, 100, 0)
                pdf.multi_cell(0, 8, safe_text(predicted_career))
                pdf.ln(5)
                pdf.set_font("Helvetica", "", 11)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(
                    0, 7,
                    safe_text("This career path aligns closely with your technical and psychological profile. "
                            "Refer to the app for roadmap, certifications, and real-world project ideas.")
                )

                # --- FOOTER ---
                pdf.set_y(-20)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 10, safe_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), align="C")

                # --- GENERATE DOWNLOAD LINK ---
                try:
                    pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
                except Exception:
                    pdf_bytes = pdf.output(dest="S")

                b64 = base64.b64encode(pdf_bytes).decode()
                file_name = f"Career_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                download_link = f"""
                <div style="text-align:center; margin-top:25px;">
                    <a href="data:application/pdf;base64,{b64}" download="{file_name}"
                    style="
                            background:linear-gradient(90deg,#A5D6A7,#81C784);
                            color:white; font-weight:600;
                            text-decoration:none; padding:12px 30px;
                            border-radius:30px; font-size:16px;
                            box-shadow:0 4px 12px rgba(129,199,132,0.4);
                            transition:0.3s;">
                        ⬇️ Download Career Report
                    </a>
                </div>
                """

                st.success("✅ Your personalized AI Career Report is ready!")
                st.markdown(download_link, unsafe_allow_html=True)

            else:
                st.warning("⚠️ Unable to determine a suitable career path based on the provided responses. Please ensure all questions are answered accurately.")     
            
elif selected == "Chatbot":
    from fpdf import FPDF
    import io
    from datetime import datetime
    import re

    # === State Control ===
    if "warning_acknowledged" not in st.session_state:
        st.session_state.warning_acknowledged = False

    # === Modal Overlay with Countdown ===
    if not st.session_state.warning_acknowledged:
        countdown_container = st.empty()

        for remaining in range(10, 0, -1):
            countdown_container.markdown(f"""
            <style>
            .warning-overlay {{
                position: fixed;
                inset: 0;
                background: rgba(255, 255, 255, 0.96);
                backdrop-filter: blur(3px);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                animation: fadeIn 0.4s ease-in-out;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }}
            .warning-box {{
                background: #ffffff;
                border-radius: 18px;
                padding: 40px 45px 35px;
                width: 460px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(102,187,106,0.25);
                border-top: 5px solid #66BB6A;
                animation: slideUp 0.4s ease-out;
            }}
            .warning-title {{
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
                font-size: 20px;
                font-weight: 800;
                color: #E53935;
                margin-bottom: 14px;
            }}
            .warning-title span {{ font-size: 24px; }}
            .warning-text {{
                font-size: 15px;
                color: #444;
                line-height: 1.65;
                margin-bottom: 20px;
            }}
            .timer {{
                font-size: 14px;
                color: #66BB6A;
                font-weight: 600;
                letter-spacing: 0.3px;
                margin-top: 8px;
            }}
            @keyframes fadeIn {{ from {{opacity:0;}} to {{opacity:1;}} }}
            @keyframes slideUp {{ from {{transform:translateY(25px);opacity:0;}} to {{transform:translateY(0);opacity:1;}} }}
            </style>

            <div class="warning-overlay">
                <div class="warning-box">
                    <div class="warning-title"><span>⚠️</span> Important Notice</div>
                    <p class="warning-text">
                        This AI Career Mentor is designed to <b>guide students</b> by providing
                        <b>career insights and options</b> based on your skills and goals.<br><br>
                        However, before making <b>any major life decisions</b> or taking adverse steps,
                        please consult a <b>qualified professional</b> or academic advisor.
                    </p>
                    <p class="timer">Auto-closing in {remaining} second{'s' if remaining != 1 else ''}...</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            t.sleep(1)

        countdown_container.empty()
        st.session_state.warning_acknowledged = True


        st.toast("Notice acknowledged automatically. Let’s get you career-ready")
        st.rerun()


    def safe_text(text):
        """Removes or replaces unsupported characters for compatibility."""
        try:
            return text.encode("latin-1").decode("latin-1")
        except UnicodeEncodeError:
            # fallback: remove emojis and exotic Unicode characters
            return re.sub(r'[^\x00-\x7F]+', '', text)

    # ========== PAGE STYLE ==========
    st.markdown("""
    <style>
    h1 {
        text-align: center !important;
        color: #66BB6A !important;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(129, 199, 132, 0.4);
    }
    .stAlert > div {
        border-radius: 12px !important;
        border-left: 5px solid #81C784 !important;
        box-shadow: 0 2px 8px rgba(129, 199, 132, 0.15) !important;
    }
    [data-testid="stInfo"] { background-color: #F1FFF3 !important; }
    [data-testid="stWarning"] {
        background-color: #FFF8E1 !important;
        border-left-color: #FFD54F !important;
    }
    .stChatMessage {
        border-radius: 15px !important;
        padding: 12px 18px !important;
        margin: 10px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-size: 16px !important;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: linear-gradient(90deg, #A5D6A7, #81C784);
        color: #fff !important;
        text-align: right;
        border-top-right-radius: 5px !important;
        box-shadow: 0 4px 10px rgba(129, 199, 132, 0.3);
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: #ffffff;
        border-left: 4px solid #81C784;
        color: #2b2b2b !important;
        text-align: left;
        border-top-left-radius: 5px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    [data-testid="stChatInput"] textarea {
        border: 2px solid #A5D6A7 !important;
        border-radius: 10px !important;
        background-color: #F9FFF9 !important;
        color: #2b2b2b !important;
        font-size: 16px !important;
        transition: 0.3s ease;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #66BB6A !important;
        box-shadow: 0 0 10px rgba(129, 199, 132, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== MAIN UI ==========
    st.title("PathPilot Mentor")
    st.info("Welcome to the PathPilot Mentor! Ask me anything about career paths, skills, or job market trends.")
    st.toast("🚀 Welcome aboard! Your PathPilot Mentor is ready to assist you.", icon="🤖")
    # === QUIZ VALIDATION ===
    if "predicted_career" not in st.session_state or st.session_state.predicted_career is None:
        st.warning("⚠️ Please complete the quiz first!")
    else:
        car = st.session_state.get("career_path", st.session_state.get("predicted_career"))
        user_data = st.session_state.get("user_data")
        if user_data is None:
            st.error("User data missing. Please retake the quiz.")
            st.stop()

        skill_summary = "\n".join([f"- {col}: {val}/10" for col, val in user_data.iloc[0].items()])
        user_info = st.session_state.get("user_info", {})
        user_bio = (
            f"👤 User Info\n"
            f"Name: {user_info.get('name', 'Unknown')}\n"
            f"Age: {user_info.get('age', 'N/A')}\n"
            f"Location: {user_info.get('city', '')}, {user_info.get('state', '')}, {user_info.get('country', '')}\n"
            f"Hobbies: {user_info.get('hobbies', 'Not provided')}\n"
            f"Goal: {user_info.get('goal', 'Not provided')}\n"
            f"Email: {user_info.get('email', 'Not provided')}\n"
            f"CGPA: {user_info.get('cgpa', 'Not provided')}\n"
            f"College: {user_info.get('college', 'Not provided')}\n"
        )

        st.info(f"🎯 You’ve been matched with: **{car}**")
        st.caption("Ask anything about your skills, roadmap, certifications, or how to excel in this field.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # === DISPLAY CHAT ===
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # === CHAT INPUT ===
        user_input = st.chat_input("Ask your AI career mentor something...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            system_prompt = f"""
            You are an AI Career Mentor.
            The user's predicted career is: {car}.
            Their skill profile is as follows:
            {skill_summary}
            User info:
            {user_bio}
            Ensure your answers are relevant to the user's country's job market.
            Give clear, actionable career guidance with learning paths, certifications, and project ideas.
            Be concise, motivating, and friendly.
            """
            # Get your API keys from secrets.toml
            api_key_1 = st.secrets["OPENROUTER_API_KEY"]
            api_key_2 = st.secrets["OPENROUTER_API_KEY_2"]

            # Choose which one to use (primary)
            api_key = api_key_1  # or swap with api_key_2 if needed

            # Now build your headers safely
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://PathPilot.streamlit.app",
                "X-Title": "AI Career Mentor"
            }


            payload = {
                "model": "mistralai/voxtral-small-24b-2507",
                "messages": [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
            }

            with st.spinner("Mentor is typing... 💭"):
                t.sleep(random.uniform(0.6, 1.2))
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                except Exception as e:
                    st.error(f"🚨 Network error: {e}")
                    st.stop()

            if response.status_code == 200:
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if not reply:
                    st.warning("⚠️ The mentor didn’t reply. Please try again.")
                    st.json(data)
                else:
                    delay = min(1.8, max(0.4, len(reply) / 250.0))
                    t.sleep(delay)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.toast("✨ Mentor replied!", icon="💡")
                    with st.chat_message("assistant"):
                        st.markdown(reply)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")

        # === 📄 PDF EXPORT BUTTON ===
        if st.session_state.chat_history:
            st.markdown("---")

            # ✅ Mint-styled download button
            st.markdown("""
            <style>
            div.stDownloadButton > button:first-child {
                background: linear-gradient(90deg, #A5D6A7, #81C784);
                color: white;
                font-weight: 600;
                border: none;
                border-radius: 30px;
                padding: 0.7em 2em;
                box-shadow: 0 4px 12px rgba(129,199,132,0.4);
                transition: all 0.3s ease-in-out;
            }
            div.stDownloadButton > button:first-child:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 16px rgba(129,199,132,0.6);
                background: linear-gradient(90deg, #81C784, #66BB6A);
            }
            </style>
            """, unsafe_allow_html=True)

            # === Define PDF class ===
            class PDF(FPDF):
                def header(self):
                    # works for both versions
                    try:
                        self.set_font("Helvetica", "B", 14)
                    except:
                        self.set_font("Arial", "B", 14)
                    self.set_text_color(102, 187, 106)
                    self.cell(0, 10, safe_text("AI Career Mentor - Chat Summary"), ln=True, align="C")
                    self.ln(5)

                def section_title(self, title):
                    try:
                        self.set_font("Helvetica", "B", 12)
                    except:
                        self.set_font("Arial", "B", 12)
                    self.set_text_color(76, 175, 80)
                    self.cell(0, 8, safe_text(title), ln=True)
                    self.ln(3)

                def section_body(self, text):
                    try:
                        self.set_font("Helvetica", "", 11)
                    except:
                        self.set_font("Arial", "", 11)
                    self.set_text_color(0, 0, 0)
                    self.multi_cell(0, 7, safe_text(text))
                    self.ln(4)

            # === Create PDF ===
            pdf = PDF()
            pdf.add_page()

            # Use Helvetica or fallback
            try:
                pdf.set_font("Helvetica", "", 12)
            except:
                pdf.set_font("Arial", "", 12)

            # --- User Info ---
            pdf.section_title("User Information")
            pdf.section_body(user_bio)

            # --- Chat History ---
            pdf.section_title("Chat History")

            for msg in st.session_state.chat_history:
                role = "You" if msg["role"] == "user" else "Mentor"
                if msg["role"] == "assistant":
                    pdf.set_text_color(102, 187, 106)
                else:
                    pdf.set_text_color(33, 33, 33)

                try:
                    pdf.set_font("Helvetica", "B", 11)
                except:
                    pdf.set_font("Arial", "B", 11)

                pdf.cell(0, 7, safe_text(f"{role}:"), ln=True)
                pdf.set_font("Helvetica", "", 11)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 6, safe_text(msg["content"]))
                pdf.ln(3)

            # --- Footer ---
            pdf.set_text_color(100, 100, 100)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 10, safe_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), align="C")

            # === Safe Export for both libraries ===
            try:
                # fpdf2 supports returning str directly
                pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
            except Exception:
                # fpdf classic returns bytes already
                pdf_bytes = pdf.output(dest="S")

            # Convert bytearray → bytes if needed
            if isinstance(pdf_bytes, bytearray):
                pdf_bytes = bytes(pdf_bytes)

            st.download_button(
                label="⬇️ Download Chat as PDF",
                data=pdf_bytes,
                file_name=f"AI_Career_Mentor_Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
