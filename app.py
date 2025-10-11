import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Page Config
st.set_page_config(
    page_title="AI Career Compass",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Model Loading
# ==============================
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None
            ).to("cpu")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"TinyLlama model could not be loaded: {e}")
        return None, None

tokenizer, model = load_model()
TINYLLAMA_AVAILABLE = model is not None

# Career Recommendation Engine
class CareerRecommendationEngine:
    def __init__(self):
        self.careers = {
            # Software Development & Engineering
            "Software Engineer": {
                "min_coding": 6, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_teamwork": True,
                "keywords": ["coding", "logical", "technical"],
                "description": "Design, develop, and maintain software applications"
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
            "Mobile App Developer": {
                "min_coding": 7, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["coding", "Java", "Python", "hackathons"],
                "description": "Build applications for iOS and Android platforms"
            },
            "Game Developer": {
                "min_coding": 8, "min_logical": 7, "min_academic": 60,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["coding", "hackathons", "technical"],
                "description": "Create interactive video games and simulations"
            },
            "Embedded Systems Engineer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["coding", "logical", "technical"],
                "description": "Develop software for hardware devices and IoT"
            },
            
            # Data Science & AI/ML
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
            "Data Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["Python", "Data Science", "Cloud", "coding"],
                "description": "Build and maintain data pipelines and infrastructure"
            },
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
            "AI Research Scientist": {
                "min_coding": 8, "min_logical": 9, "min_academic": 85,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["AI", "ML", "Python", "olympiads"],
                "description": "Advance the field of artificial intelligence through research"
            },
            "Business Intelligence Analyst": {
                "min_coding": 5, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["Data Science", "logical", "Management"],
                "description": "Transform data into actionable business insights"
            },
            "Quantitative Analyst": {
                "min_coding": 7, "min_logical": 9, "min_academic": 80,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["logical", "Data Science", "Python", "olympiads"],
                "description": "Use mathematical models for financial analysis and trading"
            },
            "Data Analyst": {
                "min_coding": 5, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["Data Science", "Python", "logical"],
                "description": "Analyze data to support business decision-making"
            },
            
            # Cloud & Infrastructure
            "Cloud Architect": {
                "min_coding": 6, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_certifications": True,
                "keywords": ["Cloud", "AWS", "Azure", "Networking"],
                "description": "Design and manage cloud infrastructure solutions"
            },
            "DevOps Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_teamwork": True,
                "keywords": ["Cloud", "AWS", "Azure", "coding"],
                "description": "Automate and streamline development and operations"
            },
            "Site Reliability Engineer": {
                "min_coding": 7, "min_logical": 8, "min_academic": 65,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["Cloud", "coding", "technical", "logical"],
                "description": "Ensure system reliability and performance at scale"
            },
            "Cloud Solutions Architect": {
                "min_coding": 6, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_certifications": True,
                "keywords": ["Cloud", "AWS", "Azure", "technical"],
                "description": "Design enterprise-level cloud solutions and migrations"
            },
            "Infrastructure Engineer": {
                "min_coding": 6, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["Cloud", "Networking", "technical"],
                "description": "Build and maintain IT infrastructure systems"
            },
            
            # Cybersecurity
            "Cybersecurity Analyst": {
                "min_coding": 5, "min_logical": 8, "min_academic": 65,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["Cybersecurity", "Networking", "logical"],
                "description": "Protect systems and networks from cyber threats"
            },
            "Security Engineer": {
                "min_coding": 7, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["Cybersecurity", "coding", "technical"],
                "description": "Design and implement security systems and protocols"
            },
            "Penetration Tester": {
                "min_coding": 7, "min_logical": 8, "min_academic": 65,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["Cybersecurity", "Networking", "coding"],
                "description": "Test systems for vulnerabilities through ethical hacking"
            },
            "Security Architect": {
                "min_coding": 6, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_experience": True,
                "keywords": ["Cybersecurity", "Cloud", "technical"],
                "description": "Design comprehensive security architectures for organizations"
            },
            
            # Blockchain & Emerging Tech
            "Blockchain Developer": {
                "min_coding": 8, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["coding", "Cybersecurity", "logical"],
                "description": "Develop decentralized applications and smart contracts"
            },
            "Web3 Developer": {
                "min_coding": 8, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["Blockchain", "coding", "hackathons"],
                "description": "Build decentralized applications on blockchain platforms"
            },
            "Robotics Engineer": {
                "min_coding": 7, "min_logical": 8, "min_academic": 75,
                "tech_oriented": True, "requires_innovation": True,
                "keywords": ["coding", "AI", "logical", "olympiads"],
                "description": "Design and build intelligent robotic systems"
            },
            "IoT Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["coding", "Cloud", "technical"],
                "description": "Develop connected devices and IoT ecosystems"
            },
            "AR/VR Developer": {
                "min_coding": 8, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["coding", "AI", "hackathons"],
                "description": "Create immersive augmented and virtual reality experiences"
            },
            
            # Systems & Networking
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
                "description": "Design and maintain computer networks and infrastructure"
            },
            "Database Administrator": {
                "min_coding": 6, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["coding", "Data Science", "technical"],
                "description": "Manage and optimize database systems"
            },
            "Systems Administrator": {
                "min_coding": 5, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["technical", "Networking", "Cloud"],
                "description": "Maintain and troubleshoot IT systems and servers"
            },
            
            # Quality & Testing
            "Quality Assurance Engineer": {
                "min_coding": 5, "min_logical": 7, "min_academic": 60,
                "tech_oriented": True, "requires_attention": True,
                "keywords": ["coding", "logical", "technical"],
                "description": "Test software to ensure quality and reliability"
            },
            "Test Automation Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["coding", "Python", "technical"],
                "description": "Develop automated testing frameworks and scripts"
            },
            "Performance Engineer": {
                "min_coding": 7, "min_logical": 8, "min_academic": 65,
                "tech_oriented": True, "requires_analytical": True,
                "keywords": ["coding", "Cloud", "logical"],
                "description": "Optimize application performance and scalability"
            },
            
            # Research & Academia
            "Research Scientist": {
                "min_coding": 6, "min_logical": 9, "min_academic": 80,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["AI", "ML", "logical", "olympiads"],
                "description": "Conduct advanced research in specialized technical fields"
            },
            "Academic Researcher": {
                "min_coding": 5, "min_logical": 8, "min_academic": 85,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["olympiads", "logical", "talent_tests", "reading_writing"],
                "description": "Pursue academic research and publish scholarly findings"
            },
            "Computational Scientist": {
                "min_coding": 7, "min_logical": 9, "min_academic": 80,
                "tech_oriented": True, "requires_research": True,
                "keywords": ["Python", "logical", "Data Science"],
                "description": "Apply computational methods to solve scientific problems"
            },
            
            # Product & Management
            "Product Manager": {
                "min_coding": 3, "min_logical": 6, "min_academic": 65,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "public_speaking", "teamwork"],
                "description": "Define product strategy and lead development teams"
            },
            "Technical Product Manager": {
                "min_coding": 5, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_communication": True,
                "keywords": ["Management", "technical", "public_speaking"],
                "description": "Lead technical products with deep technical understanding"
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
                "description": "Advise organizations on technology strategy and implementation"
            },
            "Project Manager": {
                "min_coding": 3, "min_logical": 6, "min_academic": 65,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "teamwork", "public_speaking"],
                "description": "Plan and execute technology projects from start to finish"
            },
            "Scrum Master": {
                "min_coding": 3, "min_logical": 5, "min_academic": 60,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["Management", "teamwork", "public_speaking"],
                "description": "Facilitate agile development processes and team collaboration"
            },
            "Engineering Manager": {
                "min_coding": 6, "min_logical": 7, "min_academic": 70,
                "tech_oriented": True, "requires_leadership": True,
                "keywords": ["Management", "coding", "teamwork"],
                "description": "Lead and manage engineering teams and technical projects"
            },
            "Entrepreneur": {
                "min_coding": 4, "min_logical": 6, "min_academic": 50,
                "tech_oriented": False, "requires_risk_taking": True,
                "keywords": ["hackathons", "self_learning", "Management"],
                "description": "Build and scale your own technology business venture"
            },
            
            # Design & UX
            "UX Designer": {
                "min_coding": 4, "min_logical": 5, "min_academic": 55,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["technical", "creative", "user-focused"],
                "description": "Design intuitive and beautiful user experiences"
            },
            "UI Designer": {
                "min_coding": 3, "min_logical": 5, "min_academic": 55,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["creative", "technical"],
                "description": "Create visually appealing user interfaces"
            },
            "UX Researcher": {
                "min_coding": 2, "min_logical": 6, "min_academic": 60,
                "tech_oriented": False, "requires_analytical": True,
                "keywords": ["logical", "public_speaking", "teamwork"],
                "description": "Study user behavior to inform product design decisions"
            },
            "Product Designer": {
                "min_coding": 4, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["technical", "creative", "teamwork"],
                "description": "Create end-to-end product design solutions"
            },
            "Interaction Designer": {
                "min_coding": 3, "min_logical": 5, "min_academic": 60,
                "tech_oriented": True, "requires_creativity": True,
                "keywords": ["creative", "technical"],
                "description": "Design how users interact with digital products"
            },
            
            # Content & Communication
            "Technical Writer": {
                "min_coding": 3, "min_logical": 5, "min_academic": 65,
                "tech_oriented": True, "requires_writing": True,
                "keywords": ["reading_writing", "technical", "documentation"],
                "description": "Create clear technical documentation and user guides"
            },
            "Content Strategist": {
                "min_coding": 2, "min_logical": 5, "min_academic": 60,
                "tech_oriented": False, "requires_writing": True,
                "keywords": ["reading_writing", "public_speaking", "creative"],
                "description": "Plan and manage content strategy across digital platforms"
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
                "description": "Optimize websites for search engine visibility and ranking"
            },
            "Content Manager": {
                "min_coding": 2, "min_logical": 5, "min_academic": 60,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["reading_writing", "Management", "creative"],
                "description": "Oversee content creation and publication strategies"
            },
            
            # Specialized Roles
            "AI Ethics Consultant": {
                "min_coding": 4, "min_logical": 7, "min_academic": 70,
                "tech_oriented": False, "requires_ethics": True,
                "keywords": ["AI", "ML", "reading_writing", "public_speaking"],
                "description": "Ensure responsible and ethical AI development and deployment"
            },
            "Technical Support Engineer": {
                "min_coding": 4, "min_logical": 6, "min_academic": 55,
                "tech_oriented": True, "requires_communication": True,
                "keywords": ["technical", "public_speaking", "teamwork"],
                "description": "Help users resolve technical issues and problems"
            },
            "Sales Engineer": {
                "min_coding": 3, "min_logical": 5, "min_academic": 55,
                "tech_oriented": False, "requires_communication": True,
                "keywords": ["public_speaking", "technical", "Salary"],
                "description": "Bridge technical products with customer needs through sales"
            },
            "Solutions Architect": {
                "min_coding": 6, "min_logical": 8, "min_academic": 70,
                "tech_oriented": True, "requires_experience": True,
                "keywords": ["technical", "Cloud", "coding"],
                "description": "Design comprehensive technical solutions for complex problems"
            },
            "Platform Engineer": {
                "min_coding": 7, "min_logical": 7, "min_academic": 65,
                "tech_oriented": True, "requires_technical": True,
                "keywords": ["Cloud", "coding", "technical"],
                "description": "Build and maintain developer platforms and tools"
            },
            "Release Manager": {
                "min_coding": 5, "min_logical": 6, "min_academic": 60,
                "tech_oriented": True, "requires_coordination": True,
                "keywords": ["Management", "technical", "teamwork"],
                "description": "Coordinate software releases and deployment processes"
            },
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

# Modern Claude-inspired Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: #ffffff;
        padding: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8f9fa;
        padding: 4px;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border-radius: 6px;
        color: #4a5568;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0 1.5rem;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #1a202c;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    .stButton>button {
        background: #2b6cb0;
        color: white;
        font-weight: 500;
        font-size: 0.95rem;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        border: none;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: #2c5282;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: #f0fdf4;
        color: #166534;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #bbf7d0;
    }
    
    .career-card {
        background: #fafafa;
        border-left: 3px solid #2b6cb0;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .career-card:hover {
        background: #f5f5f5;
        transform: translateX(4px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1a202c;
        font-weight: 600;
    }
    
    label {
        color: #4a5568 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stRadio > label {
        color: #1a202c !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    .section-header {
        color: #1a202c;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .progress-indicator {
        background: #f8f9fa;
        border-left: 3px solid #2b6cb0;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #4a5568;
    }
    
    .stat-card {
        text-align: center;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .sidebar .sidebar-content {
        background: #fafafa;
    }
    
    /* Chat styling */
    .chat-message-user {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border-left: 3px solid #0284c7;
    }
    
    .chat-message-assistant {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border-left: 3px solid #64748b;
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
    st.session_state.selected_question = ""
    st.session_state.clear_input = False

# Header
st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem;'>AI Career Compass</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #64748b; margin-top: 0;'>Discover your perfect career path with AI-powered insights</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Dashboard")
    
    if st.session_state.prediction_made:
        st.success("Assessment Complete")
        st.markdown(f"**Top Match:** {st.session_state.recommended_careers[0][0]}")
        st.markdown(f"**Score:** {st.session_state.recommended_careers[0][1]:.1f}%")
    else:
        st.markdown("<div class='progress-indicator'>", unsafe_allow_html=True)
        st.markdown("**Profile:** " + ("Complete" if st.session_state.profile_filled else "Pending"))
        st.markdown("**Skills:** " + ("Complete" if st.session_state.skills_filled else "Pending"))
        st.markdown("**Preferences:** " + ("Complete" if st.session_state.preferences_filled else "Pending"))
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("• 60+ Career Options")
    st.markdown("• Smart Matching Algorithm")
    st.markdown("• AI Career Counselor")
    st.markdown("• Detailed Analytics")
    
    st.markdown("---")
    st.markdown("### Quick Tips")
    st.info("Fill all three tabs for best results")

# Main navigation
main_tab = st.radio("", ["Career Assessment", "AI Career Guide"], horizontal=True, label_visibility="collapsed")

if main_tab == "Career Assessment":
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Academic Profile", "Skills & Experience", "Career Preferences"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Academic Background</div>", unsafe_allow_html=True)
        st.markdown("Tell us about your academic performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenth = st.slider("10th Grade Percentage", 0, 100, 70, help="Your 10th standard percentage")
        with col2:
            twelfth = st.slider("12th Grade Percentage", 0, 100, 70, help="Your 12th standard percentage")
        with col3:
            ug = st.slider("Undergraduate Percentage", 0, 100, 70, help="Your current/expected UG percentage")
        
        if tenth > 0 or twelfth > 0 or ug > 0:
            st.session_state.profile_filled = True
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Technical Skills</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            logical = st.slider("Logical Reasoning", 0, 10, 5, help="Rate your logical thinking ability")
            coding = st.slider("Coding Skills", 0, 10, 5, help="Rate your programming proficiency")
        with col2:
            hackathons = st.selectbox("Participated in Hackathons?", ["No", "Yes"])
            hours_per_day = st.slider("Study/Work Hours per Day", 0, 12, 8)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Soft Skills</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Additional Achievements</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Career Preferences</div>", unsafe_allow_html=True)
        st.markdown("Help us understand your career goals")
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

    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    
    all_tabs_filled = st.session_state.profile_filled and st.session_state.skills_filled and st.session_state.preferences_filled
    
    if all_tabs_filled:
        st.success("All sections completed. Ready to discover your career path.")
    else:
        st.warning("Please complete all three tabs above for accurate recommendations")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Discover My Career Path", use_container_width=True, type="primary"):
            with st.spinner("Analyzing your profile..."):
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
                
                recommendations = engine.recommend_careers(profile, top_n=5)
                st.session_state.prediction_made = True
                st.session_state.recommended_careers = recommendations
                st.session_state.user_profile = profile
                st.balloons()

    # Display results
    if st.session_state.prediction_made:
        profile = st.session_state.user_profile
        
        st.markdown("<br><div class='success-box'>", unsafe_allow_html=True)
        st.markdown(f"**Your Top Career Match**")
        st.markdown(f"<h2 style='margin: 0.5rem 0; font-size: 2rem;'>{st.session_state.recommended_careers[0][0]}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.2rem; margin: 0.5rem 0;'>Match Score: {st.session_state.recommended_careers[0][1]:.1f}%</p>", unsafe_allow_html=True)
        career_desc = engine.careers[st.session_state.recommended_careers[0][0]]['description']
        st.markdown(f"<p style='font-size: 0.95rem;'>{career_desc}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Your Top 5 Career Matches</div>", unsafe_allow_html=True)
        for i, (career, score) in enumerate(st.session_state.recommended_careers, 1):
            color_map = {1: "#2b6cb0", 2: "#3182ce", 3: "#4299e1", 4: "#63b3ed", 5: "#90cdf4"}
            color = color_map.get(i, "#2b6cb0")
            career_desc = engine.careers[career]['description']
            st.markdown(f"""
            <div class='career-card' style='border-left-color: {color};'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <strong style='font-size: 1.1rem; color: #1a202c;'>{i}. {career}</strong>
                    <span style='color: {color}; font-weight: 600; font-size: 1rem;'>{score:.1f}%</span>
                </div>
                <p style='color: #64748b; font-size: 0.85rem; margin: 0.5rem 0;'>{career_desc}</p>
                <div style='background: linear-gradient(90deg, {color} {score}%, #e2e8f0 {score}%); 
                            height: 6px; border-radius: 3px; margin-top: 0.5rem;'></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analytics section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Your Profile Analytics</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        avg_academic = (profile['tenth'] + profile['twelfth'] + profile['ug']) / 3
        
        with col1:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
                <div style='font-size: 1.5rem; font-weight: 600; color: #2b6cb0;'>{round(avg_academic, 1)}%</div>
                <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>Academic Average</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>💻</div>
                <div style='font-size: 1.5rem; font-weight: 600; color: #2b6cb0;'>{round((profile['logical'] + profile['coding'])/2, 1)}/10</div>
                <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>Technical Skills</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🗣️</div>
                <div style='font-size: 1.5rem; font-weight: 600; color: #2b6cb0;'>{profile['public_speaking']}/10</div>
                <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>Communication</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>⏱️</div>
                <div style='font-size: 1.5rem; font-weight: 600; color: #2b6cb0;'>{profile['hours_per_day']} hrs</div>
                <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>Daily Commitment</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Over", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.profile_filled = False
                st.session_state.skills_filled = False
                st.session_state.preferences_filled = False
                st.rerun()
        with col2:
            if st.button("Get AI Guidance", use_container_width=True, type="primary"):
                st.rerun()

else:  # AI Career Guide tab
    
    # ==============================
    # Hugging Face Config
    # ==============================
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    HF_MODEL = "tiiuae/falcon-7b-instruct"
    HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    # ==============================
    # Premade Questions
    # ==============================
    PREMADE_QUESTIONS = [
        "What skills should I focus on for my top career recommendation?",
        "How can I improve my profile to be more competitive?",
        "What certifications would you recommend for my career path?",
        "Should I pursue higher education or gain work experience first?",
        "What are the typical salary ranges for my recommended careers?",
        "How can I transition from my current field to my recommended career?",
        "What networking strategies should I follow?",
        "What are the emerging trends in my recommended field?"
    ]

    # ==============================
    # TinyLlama Response
    # ==============================
    def get_tinyllama_response(context):
        if not TINYLLAMA_AVAILABLE:
            return False, "TinyLlama model not available"
        try:
            messages = [
                {"role": "system", "content": "You are a helpful career counselor providing concise, actionable advice."},
                {"role": "user", "content": context}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=350,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()

            return True, response
        except Exception as e:
            return False, f"TinyLlama failed: {str(e)}"

    # ==============================
    # Hugging Face Response
    # ==============================
    def get_hf_response(context):
        if not HF_API_TOKEN:
            return False, "HF API token not available"

        payload = {"inputs": context, "parameters": {"max_new_tokens": 300}}
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers=HF_HEADERS,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            output = response.json()
            if isinstance(output, list) and "generated_text" in output[0]:
                return True, output[0]["generated_text"].strip()
            elif isinstance(output, dict) and "error" in output:
                return False, f"Hugging Face API error: {output['error']}"
            else:
                return True, str(output)
        except Exception as e:
            return False, f"Hugging Face failed: {str(e)}"

    # ==============================
    # Fallback Response
    # ==============================
    def get_fallback_response(user_question, profile, careers):
        top_career = careers[0][0]
        avg_academic = (profile['tenth'] + profile['twelfth'] + profile['ug']) / 3
        question_lower = user_question.lower()

        if "skill" in question_lower:
            return f"For {top_career}, focus on: technical skills relevant to the field, problem-solving abilities, and industry-specific certifications. Your coding score of {profile['coding']}/10 is a good foundation."
        elif "certif" in question_lower:
            return f"Recommended certifications for {top_career}: Consider industry-standard certifications, online courses from reputable platforms, and practical project experience."
        elif "salary" in question_lower:
            return f"Salary ranges for {top_career} vary by location and experience. Entry-level positions typically start at competitive rates, with significant growth potential as you gain experience."
        elif "education" in question_lower or "experience" in question_lower:
            return f"With your academic average of {avg_academic:.1f}%, consider balancing education with practical experience. Internships and entry-level positions can provide valuable hands-on learning."
        else:
            return f"Based on your profile (Avg Academic: {avg_academic:.1f}%, Coding: {profile['coding']}/10), here are recommendations for {top_career}:\n\n• Focus on upskilling in relevant technical areas\n• Obtain industry-recognized certifications\n• Build a portfolio of practical projects\n• Network with professionals in the field\n\nYour {profile['worker_type']} approach and {profile['mgt_or_tech']} focus align well with this career path."

    # ==============================
    # Main AI Response
    # ==============================
    def get_ai_response(user_question, profile, careers):
        context = f"""User Profile:
    - Academic Avg: {(profile['tenth'] + profile['twelfth'] + profile['ug'])/3:.1f}%
    - Coding: {profile['coding']}/10
    - Logical Thinking: {profile['logical']}/10
    - Public Speaking: {profile['public_speaking']}/10
    - Worker Type: {profile['worker_type']}
    - Career Direction: {profile['mgt_or_tech']}
    - Study Hours/Day: {profile['hours_per_day']}
    - Priority: {profile['salary_or_work']}

    Top Career Recommendations:
    1. {careers[0][0]} ({careers[0][1]:.1f}% match)
    2. {careers[1][0]} ({careers[1][1]:.1f}% match)

    Question: {user_question}

    Provide concise, actionable career advice in 2-3 paragraphs."""

        if TINYLLAMA_AVAILABLE:
            success, response = get_tinyllama_response(context)
            if success:
                return response, "TinyLlama (Local)"

        if HF_API_TOKEN:
            success, response = get_hf_response(context)
            if success:
                return response, "Hugging Face API"

        return get_fallback_response(user_question, profile, careers), "Fallback Assistant"

    # ==============================
    # UI Helper
    # ==============================
    def render_chat_message(message):
        if message["role"] == "user":
            st.markdown(f"""
            <div class='chat-message-user'>
                <strong style='color: #0284c7;'>You</strong>
                <p style='color: #1a202c; margin: 0.5rem 0 0 0;'>{message['content']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-message-assistant'>
                <strong style='color: #475569;'>Career Advisor</strong>
                <p style='color: #1a202c; margin: 0.5rem 0 0 0;'>{message['content']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ==============================
    # Streamlit UI
    # ==============================
    st.title("💼 AI Career Counselor")

    status_col1, status_col2 = st.columns([3, 1])
    with status_col2:
        if TINYLLAMA_AVAILABLE:
            st.success("TinyLlama Ready ")
        else:
            st.warning("TinyLlama Unavailable ")

    # Quick Questions
    st.markdown("### Quick Questions")
    cols = st.columns(4)
    for idx, question in enumerate(PREMADE_QUESTIONS):
        col_idx = idx % 4
        with cols[col_idx]:
            if st.button(question[:25] + "...", key=f"premade_{idx}", use_container_width=True):
                st.session_state["user_input_area"] = question

    st.markdown("---")

    # Chat history
    st.markdown("### Conversation")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_input_area" not in st.session_state:
        st.session_state.user_input_area = ""

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            render_chat_message(message)
    else:
        st.info("Start a conversation by asking a question or clicking a quick question above 👆")

    # Chat input
    user_question = st.text_area(
        "Your Question",
        value=st.session_state.user_input_area,
        height=100,
        placeholder="Ask anything about your career...",
        key="user_input_area"
    )

    # Buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send", type="primary"):
            if user_question.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.spinner("Thinking..."):
                    response, source = get_ai_response(
                        user_question,
                        st.session_state.user_profile,
                        st.session_state.recommended_careers
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"{response}\n\n*Source: {source}*"}
                    )
                st.session_state.user_input_area = ""
                st.rerun()
            else:
                st.warning("Please enter a question first!")

    with col2:
        if st.button("Clear Chat "):
            st.session_state.chat_history = []
            st.session_state.user_input_area = ""