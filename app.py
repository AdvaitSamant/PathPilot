import streamlit as st
import pandas as pd
import numpy as np
import ollama

# --- Page Config ---
st.set_page_config(
    page_title="AI Career Compass",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Career Recommendation Algorithm ---
class CareerRecommendationEngine:
    def __init__(self):
        self.careers = {
            # Technical Careers
            "Software Engineer": {
                "min_coding": 6, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_teamwork": True,
                "keywords": ["coding", "logical", "technical"],
                "description": "Design, develop, and maintain software applications"
            },
            "Data Scientist": {
                "min_coding": 7, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["ML", "Data Science", "Python", "logical"],
                "description": "Extract insights from complex data using statistics and ML"
            },
            "Machine Learning Engineer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 75,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["ML", "AI", "Python", "Data Science"],
                "description": "Build and deploy machine learning models at scale"
            },
            "Cloud Architect": {
                "min_coding": 6, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_certifications": True,
                "keywords": ["Cloud", "AWS", "Azure", "Networking"],
                "description": "Design and manage cloud infrastructure solutions"
            },
            "Cybersecurity Analyst": {
                "min_coding": 5, "min_logical": 8, "min_academic": 65,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["Cybersecurity", "Networking", "logical"],
                "description": "Protect systems and networks from cyber threats"
            },
            "DevOps Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_teamwork": True,
                "keywords": ["Cloud", "AWS", "Azure", "coding"],
                "description": "Automate and streamline development and operations"
            },
            "Full Stack Developer": {
                "min_coding": 7, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["coding", "technical", "hackathons"],
                "description": "Build complete web applications from frontend to backend"
            },
            "Frontend Developer": {
                "min_coding": 6, "min_logical": 5, "min_academic": 55,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["coding", "technical", "creative"],
                "description": "Create beautiful and responsive user interfaces"
            },
            "Backend Developer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 60,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["coding", "Python", "Java", "technical"],
                "description": "Build server-side logic and database systems"
            },
            "Blockchain Developer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["coding", "Cybersecurity", "logical"],
                "description": "Develop decentralized applications and smart contracts"
            },
            "Game Developer": {
                "min_coding": 8, "min_logical": 7, "min_academic": 60,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["coding", "hackathons", "technical"],
                "description": "Create interactive video games and simulations"
            },
            "Mobile App Developer": {
                "min_coding": 7, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["coding", "Java", "Python", "hackathons"],
                "description": "Build applications for iOS and Android platforms"
            },
            "Robotics Engineer": {
                "min_coding": 7, "min_logical": 8, "min_academic": 75,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["coding", "AI", "logical", "olympiads"],
                "description": "Design and build intelligent robotic systems"
            },
            "Embedded Systems Engineer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["coding", "logical", "technical"],
                "description": "Develop software for hardware devices and IoT"
            },
            "Systems Analyst": {
                "min_coding": 5, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["logical", "technical", "Networking"],
                "description": "Analyze and optimize business systems and processes"
            },
            "Network Engineer": {
                "min_coding": 5, "min_logical": 7, "min_academic": 60,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["Networking", "technical", "Cloud"],
                "description": "Design and maintain computer networks"
            },
            "Database Administrator": {
                "min_coding": 6, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["coding", "Data Science", "technical"],
                "description": "Manage and optimize database systems"
            },
            "Quality Assurance Engineer": {
                "min_coding": 5, "min_logical": 7, "min_academic": 60,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["coding", "logical", "technical"],
                "description": "Test software to ensure quality and reliability"
            },
            "Site Reliability Engineer": {
                "min_coding": 7, "min_logical": 8, "min_academic": 65,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["Cloud", "coding", "technical", "logical"],
                "description": "Ensure system reliability and performance at scale"
            },
            
            # Data & Analytics
            "Quantitative Analyst": {
                "min_coding": 7, "min_logical": 9, "min_academic": 80,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["logical", "Data Science", "Python", "olympiads"],
                "description": "Use mathematical models for financial analysis"
            },
            "Business Intelligence Analyst": {
                "min_coding": 5, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["Data Science", "logical", "Management"],
                "description": "Transform data into actionable business insights"
            },
            "Data Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["Python", "Data Science", "Cloud", "coding"],
                "description": "Build and maintain data pipelines and infrastructure"
            },
            
            # AI/ML Specialists
            "Computer Vision Engineer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 75,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["AI", "ML", "Python", "coding"],
                "description": "Develop systems that process and understand images"
            },
            "NLP Engineer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 75,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["AI", "ML", "Python", "Data Science"],
                "description": "Build systems that understand and generate language"
            },
            
            # Research & Innovation
            "Research Scientist": {
                "min_coding": 6, "min_logical": 9, "min_academic": 80,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["AI", "ML", "logical", "olympiads"],
                "description": "Conduct advanced research in specialized fields"
            },
            "Academic Researcher": {
                "min_coding": 5, "min_logical": 8, "min_academic": 85,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["olympiads", "logical", "talent_tests", "reading_writing"],
                "description": "Pursue academic research and publish findings"
            },
            "AI Research Scientist": {
                "min_coding": 8, "min_logical": 9, "min_academic": 85,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["AI", "ML", "Python", "olympiads"],
                "description": "Advance the field of artificial intelligence"
            },
            
            # Management & Business
            "Product Manager": {
                "min_coding": 3, "min_logical": 6, "min_academic": 65,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "public_speaking", "teamwork"],
                "description": "Define product strategy and lead development teams"
            },
            "Business Analyst": {
                "min_coding": 3, "min_logical": 7, "min_academic": 65,
                "tech_oriented": False, "requires_analytical": True,
                "keywords": ["Management", "logical", "Data Science"],
                "description": "Bridge business needs with technical solutions"
            },
            "IT Consultant": {
                "min_coding": 4, "min_logical": 6, "min_academic": 65,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "technical", "public_speaking"],
                "description": "Advise organizations on technology strategy"
            },
            "Project Manager": {
                "min_coding": 3, "min_logical": 6, "min_academic": 65,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "teamwork", "public_speaking"],
                "description": "Plan and execute projects from start to finish"
            },
            "Entrepreneur/Startup Founder": {
                "min_coding": 4, "min_logical": 6, "min_academic": 50,
                "tech_oriented": False, "requires_risk_taking": True,
                "keywords": ["hackathons", "self_learning", "Management"],
                "description": "Build and scale your own business venture"
            },
            
            # Design & Creative
            "UX/UI Designer": {
                "min_coding": 4, "min_logical": 5, "min_academic": 55,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["technical", "creative", "user-focused"],
                "description": "Design intuitive and beautiful user experiences"
            },
            "UX Researcher": {
                "min_coding": 2, "min_logical": 6, "min_academic": 60,
                "tech_oriented": False, "requires_analytical": True,
                "keywords": ["logical", "public_speaking", "teamwork"],
                "description": "Study user behavior to inform design decisions"
            },
            "Product Designer": {
                "min_coding": 4, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["technical", "creative", "teamwork"],
                "description": "Create holistic product design solutions"
            },
            
            # Content & Communication
            "Technical Writer": {
                "min_coding": 3, "min_logical": 5, "min_academic": 65,
                "tech_oriented": True, "requires_writing": True,
                "keywords": ["reading_writing", "technical", "documentation"],
                "description": "Create clear technical documentation and guides"
            },
            "Content Strategist": {
                "min_coding": 2, "min_logical": 5, "min_academic": 60,
                "tech_oriented": False, "requires_writing": True,
                "keywords": ["reading_writing", "public_speaking", "creative"],
                "description": "Plan and manage content across platforms"
            },
            "Digital Marketing Specialist": {
                "min_coding": 2, "min_logical": 5, "min_academic": 55,
                "tech_oriented": False, "requires_creativity": True,
                "keywords": ["public_speaking", "creative", "Data Science"],
                "description": "Create and optimize digital marketing campaigns"
            },
            "SEO Specialist": {
                "min_coding": 3, "min_logical": 6, "min_academic": 55,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["technical", "Data Science", "creative"],
                "description": "Optimize websites for search engine visibility"
            },
            
            # Ethics & Specialized
            "AI Ethics Consultant": {
                "min_coding": 4, "min_logical": 7, "min_academic": 70,
                "tech_oriented": False, "requires_ethics": True,
                "keywords": ["AI", "ML", "reading_writing", "public_speaking"],
                "description": "Ensure responsible and ethical AI development"
            },
            
            # Support & Operations
            "Technical Support Engineer": {
                "min_coding": 4, "min_logical": 6, "min_academic": 55,
                "tech_oriented": True, "requires_communication": True,
                "keywords": ["technical", "public_speaking", "teamwork"],
                "description": "Help users resolve technical issues"
            },
            "Sales Engineer": {
                "min_coding": 3, "min_logical": 5, "min_academic": 55,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["public_speaking", "technical", "Salary"],
                "description": "Bridge technical products with customer needs"
            },
            "HR Tech Specialist": {
                "min_coding": 3, "min_logical": 5, "min_academic": 60,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "public_speaking", "teamwork"],
                "description": "Manage HR technology and people systems"
            }
        }
    
    def calculate_match_score(self, profile, career_req):
        score = 0
        max_score = 0
        
        # Academic requirements (20 points)
        avg_academic = (profile['tenth'] + profile['twelfth'] + profile['ug']) / 3
        if avg_academic >= career_req.get('min_academic', 50):
            score += 20
        else:
            score += (avg_academic / career_req.get('min_academic', 50)) * 20
        max_score += 20
        
        # Technical skills (30 points)
        if profile['coding'] >= career_req.get('min_coding', 0):
            score += 15
        else:
            score += (profile['coding'] / max(career_req.get('min_coding', 1), 1)) * 15
        max_score += 15
        
        if profile['logical'] >= career_req.get('min_logical', 0):
            score += 15
        else:
            score += (profile['logical'] / max(career_req.get('min_logical', 1), 1)) * 15
        max_score += 15
        
        # Career orientation match (15 points)
        if career_req.get('tech_oriented'):
            if profile['mgt_or_tech'] == 'Technical':
                score += 15
        else:
            if profile['mgt_or_tech'] == 'Management':
                score += 15
        max_score += 15
        
        # Skills and certifications match (20 points)
        keyword_matches = 0
        keywords = career_req.get('keywords', [])
        
        for keyword in keywords:
            if keyword in profile.get('workshops', []):
                keyword_matches += 1
            if keyword in profile.get('certifications', []):
                keyword_matches += 1
            if keyword.lower() in str(profile.get('mgt_or_tech', '')).lower():
                keyword_matches += 1
        
        if keywords:
            score += min((keyword_matches / len(keywords)) * 20, 20)
        max_score += 20
        
        # Work ethic bonus (15 points)
        work_ethic = (
            profile['hours_per_day'] / 12 +
            (1 if profile['self_learning'] == 'Yes' else 0) +
            (1 if profile['extra_courses'] == 'Yes' else 0) +
            (1 if profile['hackathons'] == 'Yes' else 0)
        ) / 4
        score += work_ethic * 15
        max_score += 15
        
        return (score / max_score) * 100
    
    def recommend_careers(self, profile, top_n=5):
        scores = {}
        for career, requirements in self.careers.items():
            scores[career] = self.calculate_match_score(profile, requirements)
        
        sorted_careers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_careers[:top_n]

# Initialize engine
engine = CareerRecommendationEngine()

# --- Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
    
    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
    }
    
    .career-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #3b82f6;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .career-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }
    
    label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    .stRadio > label {
        color: #f1f5f9 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .section-header {
        color: #3b82f6;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    .easter-egg {
        background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 8px 32px rgba(236, 72, 153, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .progress-indicator {
        background: rgba(59, 130, 246, 0.2);
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    
    .stat-card {
        text-align: center;
        padding: 1.5rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.recommended_careers = []
    st.session_state.user_profile = {}
    st.session_state.chat_history = []
    st.session_state.profile_filled = False
    st.session_state.skills_filled = False
    st.session_state.preferences_filled = False

# Header
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>🎯 AI Career Compass</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #cbd5e1; margin-top: 0.5rem;'>Discover your perfect career path with AI-powered insights</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Dashboard")
    
    if st.session_state.prediction_made:
        st.success("✅ Assessment Complete!")
        st.markdown(f"**Top Match:** {st.session_state.recommended_careers[0][0]}")
        st.markdown(f"**Score:** {st.session_state.recommended_careers[0][1]:.1f}%")
    else:
        st.markdown("<div class='progress-indicator'>", unsafe_allow_html=True)
        st.markdown("**📝 Profile:** " + ("✅" if st.session_state.profile_filled else "⏳"))
        st.markdown("**💻 Skills:** " + ("✅" if st.session_state.skills_filled else "⏳"))
        st.markdown("**🎯 Preferences:** " + ("✅" if st.session_state.preferences_filled else "⏳"))
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🤖 Features")
    st.markdown("✨ 40+ Career Options")
    st.markdown("🎯 Smart Matching Algorithm")
    st.markdown("💬 AI Career Counselor")
    st.markdown("📊 Detailed Analytics")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("Fill all three tabs (Profile, Skills, Preferences) for best results!")

# Main navigation
main_tab = st.radio("", ["🎓 Career Assessment", "🤖 AI Career Guide"], horizontal=True, label_visibility="collapsed")

if main_tab == "🎓 Career Assessment":
    # Tabs with better visibility
    tab1, tab2, tab3 = st.tabs(["📝 Academic Profile", "💻 Skills & Experience", "🎯 Career Preferences"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📚 Academic Background</div>", unsafe_allow_html=True)
        st.markdown("*Tell us about your academic performance*")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenth = st.slider("10th Grade %", 0, 100, 70, help="Your 10th standard percentage")
        with col2:
            twelfth = st.slider("12th Grade %", 0, 100, 70, help="Your 12th standard percentage")
        with col3:
            ug = st.slider("Undergraduate %", 0, 100, 70, help="Your current/expected UG percentage")
        
        if tenth > 0 or twelfth > 0 or ug > 0:
            st.session_state.profile_filled = True
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>💻 Technical Skills</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            logical = st.slider("Logical Reasoning", 0, 10, 5, help="Rate your logical thinking ability")
            coding = st.slider("Coding Skills", 0, 10, 5, help="Rate your programming proficiency")
        with col2:
            hackathons = st.selectbox("Participated in Hackathons?", ["No", "Yes"])
            hours_per_day = st.slider("Study/Work Hours/Day", 0, 12, 8)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🗣️ Soft Skills</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            public_speaking = st.slider("Public Speaking", 0, 10, 5)
            memory = st.selectbox("Memory", ["Poor", "Medium", "Excellent"])
        with col2:
            reading_writing = st.selectbox("Reading & Writing", ["Poor", "Medium", "Excellent"])
            teamwork = st.selectbox("Team Player?", ["No", "Yes"])
        with col3:
            self_learning = st.selectbox("Self-Learning?", ["No", "Yes"])
            extra_courses = st.selectbox("Extra Courses?", ["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎓 Additional Achievements</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            long_hours = st.selectbox("Can Work Long Hours?", ["No", "Yes"])
            talent_tests = st.selectbox("Taken Talent Tests?", ["No", "Yes"])
            olympiads = st.selectbox("Participated in Olympiads?", ["No", "Yes"])
        with col2:
            workshops = st.multiselect("Workshops Attended", 
                ["ML", "AI", "Data Science", "Cloud", "Cybersecurity", "Blockchain", "IoT", "AR/VR"])
            certifications = st.multiselect("Certifications", 
                ["AWS", "Azure", "Google Cloud", "Python", "Java", "Networking", "Cybersecurity", "Data Analytics"])
        
        if logical > 0 or coding > 0 or public_speaking > 0:
            st.session_state.skills_filled = True
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎯 Career Preferences</div>", unsafe_allow_html=True)
        st.markdown("*Help us understand your career goals*")
        col1, col2 = st.columns(2)
        with col1:
            mgt_or_tech = st.selectbox("Career Direction", ["Management", "Technical"], 
                help="Do you prefer technical roles or management positions?")
            salary_or_work = st.selectbox("Primary Motivation", ["Salary", "Work"], 
                help="What drives you more - compensation or passion for work?")
        with col2:
            worker_type = st.selectbox("Working Style", ["Hard worker", "Smart worker"], 
                help="How would you describe your work approach?")
            introvert = st.selectbox("Personality Type", ["Extrovert", "Introvert"], 
                help="Are you more introverted or extroverted?")
        
        if mgt_or_tech or salary_or_work:
            st.session_state.preferences_filled = True
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction button with better visibility
    st.markdown("<br>", unsafe_allow_html=True)
    
    all_tabs_filled = st.session_state.profile_filled and st.session_state.skills_filled and st.session_state.preferences_filled
    
    if all_tabs_filled:
        st.success("✅ All sections completed! Ready to discover your career path.")
    else:
        st.warning("⚠️ Please complete all three tabs above for accurate recommendations!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Discover My Career Path", use_container_width=True, type="primary"):
            with st.spinner("🔮 Analyzing your profile with AI..."):
                # Convert introvert to yes/no format
                introvert_value = "Yes" if introvert == "Introvert" else "No"
                
                profile = {
                    "tenth": tenth, "twelfth": twelfth, "ug": ug, 
                    "logical": logical, "coding": coding,
                    "hackathons": hackathons, "public_speaking": public_speaking, 
                    "memory": memory, "reading_writing": reading_writing, 
                    "self_learning": self_learning, "teamwork": teamwork,
                    "extra_courses": extra_courses, "long_hours": long_hours, 
                    "talent_tests": talent_tests, "olympiads": olympiads, 
                    "hours_per_day": hours_per_day, "mgt_or_tech": mgt_or_tech,
                    "salary_or_work": salary_or_work, "worker_type": worker_type, 
                    "introvert": introvert_value, "workshops": workshops, 
                    "certifications": certifications
                }
                
                # Easter egg detection
                is_all_zero = (tenth == 0 and twelfth == 0 and ug == 0 and 
                              logical == 0 and coding == 0 and public_speaking == 0 and 
                              hours_per_day == 0)
                
                if is_all_zero:
                    st.markdown("<div class='easter-egg'>", unsafe_allow_html=True)
                    st.markdown("### 🎭 Interesting Profile Detected!")
                    st.markdown("**Recommended Career:** Professional Couch Potato 🛋️")
                    st.markdown("*Match Score: 100% (You're perfectly qualified for doing absolutely nothing!)*")
                    st.markdown("<br>**Alternative Careers:** Netflix Critic 📺 | Professional Napper 😴 | Furniture Tester 🪑")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("😄 Just kidding! Please fill in your actual details to get real career recommendations.")
                    st.balloons()
                else:
                    recommendations = engine.recommend_careers(profile, top_n=5)
                    st.session_state.prediction_made = True
                    st.session_state.recommended_careers = recommendations
                    st.session_state.user_profile = profile
                    st.balloons()

    # Display results
    if st.session_state.prediction_made:
        # Check if it's the easter egg scenario
        profile = st.session_state.user_profile
        is_all_zero = (profile['tenth'] == 0 and profile['twelfth'] == 0 and profile['ug'] == 0 and 
                      profile['logical'] == 0 and profile['coding'] == 0 and 
                      profile['public_speaking'] == 0 and profile['hours_per_day'] == 0)
        
        if not is_all_zero:
            st.markdown("<br><div class='success-box'>", unsafe_allow_html=True)
            st.markdown(f"🎯 **Your Top Career Match**")
            st.markdown(f"<h2 style='margin: 0.5rem 0; font-size: 2.5rem;'>{st.session_state.recommended_careers[0][0]}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 1.3rem; margin: 0.5rem 0;'>Match Score: {st.session_state.recommended_careers[0][1]:.1f}%</p>", unsafe_allow_html=True)
            career_desc = engine.careers[st.session_state.recommended_careers[0][0]]['description']
            st.markdown(f"<p style='font-size: 1rem; opacity: 0.9;'>{career_desc}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📊 Your Top 5 Career Matches</div>", unsafe_allow_html=True)
            for i, (career, score) in enumerate(st.session_state.recommended_careers, 1):
                color = "#3b82f6" if i == 1 else "#6366f1" if i == 2 else "#8b5cf6" if i == 3 else "#a855f7"
                career_desc = engine.careers[career]['description']
                st.markdown(f"""
                <div class='career-card' style='border-left-color: {color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                        <strong style='font-size: 1.2rem; color: #f1f5f9;'>{i}. {career}</strong>
                        <span style='color: {color}; font-weight: 600; font-size: 1.1rem;'>{score:.1f}%</span>
                    </div>
                    <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0;'>{career_desc}</p>
                    <div style='background: linear-gradient(90deg, {color} {score}%, rgba(255,255,255,0.1) {score}%); 
                                height: 8px; border-radius: 4px; margin-top: 0.5rem;'></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analytics section
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📈 Your Profile Analytics</div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            avg_academic = (profile['tenth'] + profile['twelfth'] + profile['ug']) / 3
            
            with col1:
                st.markdown(f"""
                <div class='stat-card' style='background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.3);'>
                    <div style='font-size: 2.5rem;'>📊</div>
                    <div style='font-size: 1.8rem; font-weight: 600; color: #3b82f6;'>{round(avg_academic, 1)}%</div>
                    <div style='color: #cbd5e1; font-size: 0.9rem;'>Academic Avg</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='stat-card' style='background: rgba(99, 102, 241, 0.1); border-color: rgba(99, 102, 241, 0.3);'>
                    <div style='font-size: 2.5rem;'>💻</div>
                    <div style='font-size: 1.8rem; font-weight: 600; color: #6366f1;'>{round((profile['logical'] + profile['coding'])/2, 1)}/10</div>
                    <div style='color: #cbd5e1; font-size: 0.9rem;'>Technical Skills</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='stat-card' style='background: rgba(139, 92, 246, 0.1); border-color: rgba(139, 92, 246, 0.3);'>
                    <div style='font-size: 2.5rem;'>🗣️</div>
                    <div style='font-size: 1.8rem; font-weight: 600; color: #8b5cf6;'>{profile['public_speaking']}/10</div>
                    <div style='color: #cbd5e1; font-size: 0.9rem;'>Communication</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class='stat-card' style='background: rgba(168, 85, 247, 0.1); border-color: rgba(168, 85, 247, 0.3);'>
                    <div style='font-size: 2.5rem;'>⏱️</div>
                    <div style='font-size: 1.8rem; font-weight: 600; color: #a855f7;'>{profile['hours_per_day']} hrs</div>
                    <div style='color: #cbd5e1; font-size: 0.9rem;'>Daily Commitment</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Start Over", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.session_state.profile_filled = False
                    st.session_state.skills_filled = False
                    st.session_state.preferences_filled = False
                    st.rerun()
            with col2:
                if st.button("💬 Get AI Guidance", use_container_width=True, type="primary"):
                    st.rerun()

else:  # AI Career Guide tab
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🤖 AI Career Counselor</div>", unsafe_allow_html=True)
    
    if not st.session_state.prediction_made:
        st.warning("⚠️ Please complete the Career Assessment first to get personalized guidance!")
        if st.button("📝 Go to Assessment", use_container_width=True, type="primary"):
            st.rerun()
    else:
        st.success(f"✅ Your profile is ready! I'm your AI career advisor. Ask me anything!")
        
        # Display user profile summary
        with st.expander("📋 Your Profile Summary", expanded=False):
            profile = st.session_state.user_profile
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📚 Academic Performance:** {(profile['tenth'] + profile['twelfth'] + profile['ug'])/3:.1f}%  
                **💻 Coding Skills:** {profile['coding']}/10  
                **🧠 Logical Reasoning:** {profile['logical']}/10  
                **🗣️ Public Speaking:** {profile['public_speaking']}/10  
                **⏱️ Daily Hours:** {profile['hours_per_day']} hours
                """)
            with col2:
                st.markdown(f"""
                **🎯 Career Direction:** {profile['mgt_or_tech']}  
                **💰 Motivation:** {profile['salary_or_work']}  
                **🏃 Working Style:** {profile['worker_type']}  
                **👤 Personality:** {"Introvert" if profile['introvert'] == "Yes" else "Extrovert"}  
                **🎓 Workshops:** {len(profile['workshops'])} attended
                """)
            
            st.markdown("**🏆 Top Career Recommendations:**")
            for i, (career, score) in enumerate(st.session_state.recommended_careers[:3], 1):
                st.markdown(f"{i}. **{career}** - {score:.1f}% match")
        
        # Chat interface
        st.markdown("---")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("<div style='max-height: 400px; overflow-y: auto; padding: 1rem;'>", unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style='background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #3b82f6;'>
                        <strong style='color: #3b82f6;'>You:</strong> <span style='color: #e2e8f0;'>{message['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #10b981;'>
                        <strong style='color: #10b981;'>AI Counselor:</strong> <span style='color: #e2e8f0;'>{message['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown("**💡 Suggested Questions:**")
            col1, col2 = st.columns(2)
            suggestions = [
                "What skills should I develop for my top career match?",
                "What certifications would benefit me most?",
                "How can I transition into my recommended career?",
                "What are the salary prospects for my top matches?",
                "What's the daily routine like in my top career?",
                "What are the growth opportunities in my field?"
            ]
            
            for i, suggestion in enumerate(suggestions[:6]):
                if i % 2 == 0:
                    with col1:
                        if st.button(f"💭 {suggestion[:40]}...", key=f"suggest_{i}", use_container_width=True):
                            st.session_state.chat_history.append({"role": "user", "content": suggestion})
                            
                            # Process with AI
                            profile = st.session_state.user_profile
                            careers = st.session_state.recommended_careers
                            
                            context = f"""
User Profile Summary:
- Academic Average: {(profile['tenth'] + profile['twelfth'] + profile['ug'])/3:.1f}%
- Coding Skills: {profile['coding']}/10
- Logical Reasoning: {profile['logical']}/10
- Public Speaking: {profile['public_speaking']}/10
- Career Direction: {profile['mgt_or_tech']}
- Motivation: {profile['salary_or_work']}
- Working Style: {profile['worker_type']}
- Daily Hours: {profile['hours_per_day']}
- Workshops: {', '.join(profile['workshops']) if profile['workshops'] else 'None'}
- Certifications: {', '.join(profile['certifications']) if profile['certifications'] else 'None'}

Top 5 Recommended Careers:
{chr(10).join([f'{i+1}. {career} ({score:.1f}% match) - {engine.careers[career]["description"]}' for i, (career, score) in enumerate(careers)])}

User Question: {suggestion}

Provide helpful, personalized career guidance based on this profile. Be specific and actionable.
"""
                            
                            try:
                                response = ollama.chat(model='phi3:mini', messages=[
                                    {"role": "system", "content": "You are an expert AI career counselor. Provide clear, actionable, and personalized career advice. Be encouraging and specific."},
                                    {"role": "user", "content": context}
                                ])
                                ai_response = response['message']['content']
                            except Exception as e:
                                ai_response = f"Based on your profile, here's my advice:\n\nFor your top match ({careers[0][0]}), I recommend:\n1. Focus on developing your {profile['mgt_or_tech'].lower()} skills\n2. Consider certifications in your areas of interest\n3. Build a strong portfolio showcasing your work\n4. Network with professionals in the field\n\n(Note: Ollama integration - ensure phi3:mini model is installed and running)"
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                            st.rerun()
                else:
                    with col2:
                        if st.button(f"💭 {suggestion[:40]}...", key=f"suggest_{i}", use_container_width=True):
                            st.session_state.chat_history.append({"role": "user", "content": suggestion})
                            
                            # Process with AI (same as above)
                            profile = st.session_state.user_profile
                            careers = st.session_state.recommended_careers
                            
                            context = f"""
User Profile Summary:
- Academic Average: {(profile['tenth'] + profile['twelfth'] + profile['ug'])/3:.1f}%
- Coding Skills: {profile['coding']}/10
- Logical Reasoning: {profile['logical']}/10
- Public Speaking: {profile['public_speaking']}/10
- Career Direction: {profile['mgt_or_tech']}
- Motivation: {profile['salary_or_work']}
- Working Style: {profile['worker_type']}
- Daily Hours: {profile['hours_per_day']}
- Workshops: {', '.join(profile['workshops']) if profile['workshops'] else 'None'}
- Certifications: {', '.join(profile['certifications']) if profile['certifications'] else 'None'}

Top 5 Recommended Careers:
{chr(10).join([f'{i+1}. {career} ({score:.1f}% match) - {engine.careers[career]["description"]}' for i, (career, score) in enumerate(careers)])}

User Question: {suggestion}

Provide helpful, personalized career guidance based on this profile. Be specific and actionable.
"""
                            
                            try:
                                response = ollama.chat(model='phi3:mini', messages=[
                                    {"role": "system", "content": "You are an expert AI career counselor. Provide clear, actionable, and personalized career advice. Be encouraging and specific."},
                                    {"role": "user", "content": context}
                                ])
                                ai_response = response['message']['content']
                            except Exception as e:
                                ai_response = f"Based on your profile, here's my advice:\n\nFor your top match ({careers[0][0]}), I recommend:\n1. Focus on developing your {profile['mgt_or_tech'].lower()} skills\n2. Consider certifications in your areas of interest\n3. Build a strong portfolio showcasing your work\n4. Network with professionals in the field\n\n(Note: Ollama integration - ensure phi3:mini model is installed and running)"
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                            st.rerun()
        
        # Chat input
        st.markdown("---")
        user_question = st.text_area("💬 Ask your question:", height=100, 
            placeholder="e.g., What skills should I focus on? What's the career growth potential?")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            if st.button("Send 📤", use_container_width=True, type="primary"):
                if user_question:
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    
                    profile = st.session_state.user_profile
                    careers = st.session_state.recommended_careers
                    
                    context = f"""
User Profile Summary:
- Academic Average: {(profile['tenth'] + profile['twelfth'] + profile['ug'])/3:.1f}%
- Coding Skills: {profile['coding']}/10
- Logical Reasoning: {profile['logical']}/10
- Public Speaking: {profile['public_speaking']}/10
- Career Direction: {profile['mgt_or_tech']}
- Motivation: {profile['salary_or_work']}
- Working Style: {profile['worker_type']}
- Daily Hours: {profile['hours_per_day']}
- Workshops: {', '.join(profile['workshops']) if profile['workshops'] else 'None'}
- Certifications: {', '.join(profile['certifications']) if profile['certifications'] else 'None'}

Top 5 Recommended Careers:
{chr(10).join([f'{i+1}. {career} ({score:.1f}% match) - {engine.careers[career]["description"]}' for i, (career, score) in enumerate(careers)])}

User Question: {user_question}

Provide helpful, personalized career guidance based on this profile. Be specific, actionable, and encouraging.
"""
                    
                    with st.spinner("🤔 Thinking..."):
                        try:
                            response = ollama.chat(model='phi3:mini', messages=[
                                {"role": "system", "content": "You are an expert AI career counselor with deep knowledge of tech and business careers. Provide clear, actionable, and personalized advice. Be encouraging, specific, and realistic."},
                                {"role": "user", "content": context}
                            ])
                            ai_response = response['message']['content']
                        except Exception as e:
                            ai_response = f"I'd be happy to help! Based on your profile:\n\n**Your Strengths:**\n- {profile['mgt_or_tech']} orientation\n- {profile['worker_type']} approach\n- {profile['hours_per_day']} hours daily commitment\n\n**For your top career match ({careers[0][0]}):**\n1. Focus on building relevant skills\n2. Consider getting certified in key technologies\n3. Build a portfolio to showcase your work\n4. Network with professionals in the field\n\n(Note: For AI responses, ensure Ollama is running with phi3:mini model installed: `ollama pull phi3:mini`)"
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                        st.rerun()
        
        with col2:
            if st.button("Clear Chat 🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("Export 📥", use_container_width=True):
                chat_export = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
                st.download_button("Download", chat_export, "career_chat.txt", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem; margin-top: 2rem;'>
    <p style='font-size: 0.9rem;'>🤖 Powered by Advanced AI • Built with ❤️ using Streamlit</p>
    <p style='font-size: 0.8rem; opacity: 0.7;'>Your data stays private • Algorithm + Ollama LLM Integration</p>
</div>
""", unsafe_allow_html=True)