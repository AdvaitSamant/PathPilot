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

import streamlit as st
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# ==============================
# Model Loading
# ==============================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return tokenizer, model

try:
    tokenizer, model = load_model()
    TINYLLAMA_AVAILABLE = True
except Exception as e:
    st.warning(f"TinyLlama model could not be loaded: {e}")
    TINYLLAMA_AVAILABLE = False
    tokenizer, model = None, None


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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Career Assessment Quiz</div>", unsafe_allow_html=True)
    st.markdown("Complete this 30-question assessment. Your responses will help us understand your profile better.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize quiz state
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
        st.session_state.quiz_responses = {}
        st.session_state.current_question = 0

    if not st.session_state.quiz_started:
        if st.button("Start Assessment", use_container_width=True):
            st.session_state.quiz_started = True
            st.rerun()
    else:
        # Quiz questions with indirect assessment
        quiz_questions = [
            # Logical Reasoning (Q1-Q5)
            {
                "q": "If all roses are flowers and all flowers fade, which statement is true?",
                "options": ["All roses fade", "Some roses don't fade", "Flowers are roses", "Can't determine"],
                "indirect": "logical",
                "correct": 0
            },
            {
                "q": "A clock shows 3:15. What's the angle between hour and minute hands?",
                "options": ["7.5°", "15°", "30°", "45°"],
                "indirect": "logical",
                "correct": 0
            },
            {
                "q": "If you rearrange 'ANMIETD', you get a word that means?",
                "options": ["A place", "A feeling", "A profession", "An object"],
                "indirect": "logical",
                "correct": 2
            },
            {
                "q": "What comes next: 2, 6, 12, 20, 30, ?",
                "options": ["40", "42", "44", "45"],
                "indirect": "logical",
                "correct": 1
            },
            {
                "q": "Which one doesn't belong: Tiger, Lion, Leopard, Cheetah, Deer?",
                "options": ["Tiger", "Lion", "Cheetah", "Deer"],
                "indirect": "logical",
                "correct": 3
            },
            # Reading & Writing (Q6-Q9)
            {
                "q": "Choose the correct sentence:",
                "options": [
                    "He don't know nothing about it",
                    "He doesn't know anything about it",
                    "He don't know anything about it",
                    "He doesn't know nothing about it"
                ],
                "indirect": "reading_writing",
                "correct": 1
            },
            {
                "q": "Which word is closest in meaning to 'Meticulous'?",
                "options": ["Careless", "Thorough", "Quick", "Lazy"],
                "indirect": "reading_writing",
                "correct": 1
            },
            {
                "q": "Identify the error: 'The manager have decided to implement the new policy next month.'",
                "options": ["manager", "have", "implement", "No error"],
                "indirect": "reading_writing",
                "correct": 1
            },
            {
                "q": "The government's new initiative aims to promote renewable energy adoption. Solar panels are being subsidized, wind farms expanded, and citizens are incentivized to reduce carbon footprint. What is the main idea?",
                "options": ["Renewable energy costs too much", "Government is promoting clean energy adoption", "Solar panels are the only solution", "Citizens must buy solar panels"],
                "indirect": "reading_writing",
                "correct": 1
            },
            # Personality & Motivation (Q10-Q15)
            {
                "q": "After an 8-hour work day, you usually feel:",
                "options": ["Energized to socialize", "Tired and need alone time", "Ready for more interaction", "Exhausted but motivated"],
                "indirect": "introvert",
                "correct": 1
            },
            {
                "q": "When facing a difficult problem, your first instinct is to:",
                "options": ["Call a friend to discuss", "Analyze it step by step alone", "Seek expert advice immediately", "Brainstorm with others"],
                "indirect": "logical",
                "correct": 1
            },
            {
                "q": "A project deadline is tight. You would:",
                "options": ["Work just enough to pass", "Put in extra effort to excel", "Work steadily without rushing", "Delegate to team members"],
                "indirect": "mgt_or_tech",
                "correct": 1
            },
            {
                "q": "In a group presentation, you'd prefer to:",
                "options": ["Lead the entire presentation", "Support from the background", "Handle technical aspects", "Coordinate between team members"],
                "indirect": "public_speaking",
                "correct": 0
            },
            {
                "q": "Your ideal job emphasizes:",
                "options": ["High salary package", "Meaningful work", "Flexible hours", "Leadership opportunities"],
                "indirect": "salary_or_work",
                "correct": 1
            },
            {
                "q": "You solve problems more effectively by:",
                "options": ["Following established procedures", "Finding creative shortcuts", "Asking for guidance", "Trial and error"],
                "indirect": "worker_type",
                "correct": 1
            },
            # Academic Performance Indicators (Q16-Q23)
            {
                "q": "Your 10th standard academic performance was:",
                "options": ["Excellent (85%+)", "Very Good (75-84%)", "Good (65-74%)", "Average (below 65%)"],
                "indirect": "tenth",
                "correct": 0
            },
            {
                "q": "Your 12th standard academic performance was:",
                "options": ["Excellent (85%+)", "Very Good (75-84%)", "Good (65-74%)", "Average (below 65%)"],
                "indirect": "twelfth",
                "correct": 0
            },
            {
                "q": "Your current/expected undergraduate percentage is:",
                "options": ["Excellent (85%+)", "Very Good (75-84%)", "Good (65-74%)", "Average (below 65%)"],
                "indirect": "ug",
                "correct": 0
            },
            {
                "q": "How do you typically study?",
                "options": ["Last-minute cramming", "Regular daily study", "Irregular but focused", "Only for exams"],
                "indirect": "hours_per_day",
                "correct": 1
            },
            {
                "q": "Your approach to learning new concepts is:",
                "options": ["Wait for classroom teaching", "Self-study from multiple sources", "Ask peers and mentors", "Video tutorials only"],
                "indirect": "self_learning",
                "correct": 1
            },
            {
                "q": "Have you pursued additional learning beyond curriculum?",
                "options": ["Yes, multiple times", "Yes, occasionally", "Rarely", "Never"],
                "indirect": "extra_courses",
                "correct": 0
            },
            {
                "q": "During group projects, you:",
                "options": ["Do all the work yourself", "Contribute equally with team", "Let others lead", "Prefer working alone"],
                "indirect": "teamwork",
                "correct": 1
            },
            # Coding & Technical (Q21-Q24)
            {
                "q": "When writing code, your priority is:",
                "options": ["Getting output quickly", "Clean, maintainable code", "Minimal lines of code", "Following trends"],
                "indirect": "coding",
                "correct": 1
            },
            {
                "q": "You've participated in competitive programming or hackathons:",
                "options": ["Multiple times", "A few times", "Once or twice", "Never"],
                "indirect": "hackathons",
                "correct": 0
            },
            {
                "q": "Your debugging approach is usually:",
                "options": ["Random trial and error", "Systematic analysis", "Copy solutions online", "Ask for help immediately"],
                "indirect": "coding",
                "correct": 1
            },
            {
                "q": "Technical topics interest you because:",
                "options": ["Easy to earn money", "Genuine curiosity", "Required for job", "Friends are doing it"],
                "indirect": "mgt_or_tech",
                "correct": 1
            },
            # Achievements & Certifications (Q25-Q28)
            {
                "q": "You've participated in competitions like Olympiads or talent tests:",
                "options": ["Multiple times", "A few times", "Once", "Never"],
                "indirect": "olympiads",
                "correct": 0
            },
            {
                "q": "Professional certifications or online courses completed:",
                "options": ["3 or more", "1-2", "None yet", "Planning to start"],
                "indirect": "certifications",
                "correct": 0
            },
            {
                "q": "Can you work extended hours when needed?",
                "options": ["Yes, regularly", "Yes, occasionally", "Rarely", "No"],
                "indirect": "long_hours",
                "correct": 0
            },
            {
                "q": "Your memory for facts and figures is:",
                "options": ["Excellent recall", "Good generally", "Average", "Struggle to remember"],
                "indirect": "memory",
                "correct": 0
            }
        ]

        # Display progress
        progress = st.session_state.current_question / len(quiz_questions)
        st.progress(progress)
        st.markdown(f"**Question {st.session_state.current_question + 1}/{len(quiz_questions)}**")

        q_data = quiz_questions[st.session_state.current_question]
        st.markdown(f"### {q_data['q']}")
        
        answer = st.radio("", q_data['options'], label_visibility="collapsed", key=f"q_{st.session_state.current_question}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.current_question > 0:
                if st.button("← Previous"):
                    st.session_state.current_question -= 1
                    st.rerun()
        
        with col3:
            if st.session_state.current_question < len(quiz_questions) - 1:
                if st.button("Next →"):
                    st.session_state.quiz_responses[st.session_state.current_question] = q_data['options'].index(answer)
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                if st.button("Submit Assessment"):
                    st.session_state.quiz_responses[st.session_state.current_question] = q_data['options'].index(answer)
                    
                    # Calculate scores from responses
                    scoring_map = {
                        'tenth': [], 'twelfth': [], 'ug': [],
                        'logical': [], 'coding': [], 'hackathons': [],
                        'hours_per_day': [], 'public_speaking': [],
                        'memory': [], 'reading_writing': [], 'teamwork': [],
                        'self_learning': [], 'extra_courses': [],
                        'long_hours': [], 'talent_tests': [],
                        'olympiads': [], 'workshops': [], 'certifications': [],
                        'mgt_or_tech': [], 'salary_or_work': [],
                        'worker_type': [], 'introvert': []
                    }
                    
                    for q_idx, response_idx in st.session_state.quiz_responses.items():
                        q_data = quiz_questions[q_idx]
                        score = 1 if response_idx == q_data['correct'] else 0
                        scoring_map[q_data['indirect']].append(score)
                    
                    # Aggregate and convert to original format
                    tenth = min(100, int((sum(scoring_map['tenth']) / max(len(scoring_map['tenth']), 1)) * 95 + 5))
                    twelfth = min(100, int((sum(scoring_map['twelfth']) / max(len(scoring_map['twelfth']), 1)) * 95 + 5))
                    ug = min(100, int((sum(scoring_map['ug']) / max(len(scoring_map['ug']), 1)) * 95 + 5))
                    
                    logical = int((sum(scoring_map['logical']) / max(len(scoring_map['logical']), 1)) * 10)
                    coding = int((sum(scoring_map['coding']) / max(len(scoring_map['coding']), 1)) * 10)
                    public_speaking = int((sum(scoring_map['public_speaking']) / max(len(scoring_map['public_speaking']), 1)) * 10)
                    
                    hackathons = "Yes" if sum(scoring_map['hackathons']) > 0 else "No"
                    long_hours = "Yes" if sum(scoring_map['long_hours']) > 0 else "No"
                    olympiads = "Yes" if sum(scoring_map['olympiads']) > 0 else "No"
                    self_learning = "Yes" if sum(scoring_map['self_learning']) > 0 else "No"
                    extra_courses = "Yes" if sum(scoring_map['extra_courses']) > 0 else "No"
                    teamwork = "Yes" if sum(scoring_map['teamwork']) > 0 else "No"
                    talent_tests = "No"  # Not directly assessed in quiz
                    
                    memory = ["Poor", "Medium", "Excellent"][min(2, int(sum(scoring_map['memory']) / max(len(scoring_map['memory']), 1) * 2))]
                    reading_writing = ["Poor", "Medium", "Excellent"][min(2, int(sum(scoring_map['reading_writing']) / max(len(scoring_map['reading_writing']), 1) * 2))]
                    
                    mgt_or_tech = "Technical" if sum(scoring_map['mgt_or_tech']) > len(scoring_map['mgt_or_tech']) / 2 else "Management"
                    salary_or_work = "Work" if sum(scoring_map['salary_or_work']) > len(scoring_map['salary_or_work']) / 2 else "Salary"
                    worker_type = "Smart worker" if sum(scoring_map['worker_type']) > len(scoring_map['worker_type']) / 2 else "Hard worker"
                    introvert = "Yes" if sum(scoring_map['introvert']) > len(scoring_map['introvert']) / 2 else "No"
                    
                    hours_per_day = 8 + int((sum(scoring_map['hours_per_day']) / max(len(scoring_map['hours_per_day']), 1)) * 4)
                    
                    # Set session state for all fields
                    st.session_state.tenth = tenth
                    st.session_state.twelfth = twelfth
                    st.session_state.ug = ug
                    st.session_state.logical = logical
                    st.session_state.coding = coding
                    st.session_state.hackathons = hackathons
                    st.session_state.public_speaking = public_speaking
                    st.session_state.memory = memory
                    st.session_state.reading_writing = reading_writing
                    st.session_state.teamwork = teamwork
                    st.session_state.self_learning = self_learning
                    st.session_state.extra_courses = extra_courses
                    st.session_state.long_hours = long_hours
                    st.session_state.talent_tests = talent_tests
                    st.session_state.olympiads = olympiads
                    st.session_state.hours_per_day = hours_per_day
                    st.session_state.mgt_or_tech = mgt_or_tech
                    st.session_state.salary_or_work = salary_or_work
                    st.session_state.worker_type = worker_type
                    st.session_state.introvert_value = introvert
                    st.session_state.workshops = []  # Not directly assessed
                    st.session_state.certifications = []  # Not directly assessed
                    
                    st.session_state.profile_filled = True
                    st.session_state.skills_filled = True
                    st.session_state.preferences_filled = True
                    st.session_state.quiz_started = False
                    
                    st.success("Assessment Complete! Your profile has been generated.")
                    st.balloons()
                    st.rerun()

    # Prediction button - shown after quiz completion
    if st.session_state.profile_filled and st.session_state.skills_filled and st.session_state.preferences_filled:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("All sections completed. Ready to discover your career path.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Discover My Career Path", use_container_width=True, type="primary"):
                with st.spinner("Analyzing your profile..."):
                    profile = {
                        "tenth": st.session_state.tenth,
                        "twelfth": st.session_state.twelfth,
                        "ug": st.session_state.ug,
                        "logical": st.session_state.logical,
                        "coding": st.session_state.coding,
                        "hackathons": st.session_state.hackathons,
                        "public_speaking": st.session_state.public_speaking,
                        "memory": st.session_state.memory,
                        "reading_writing": st.session_state.reading_writing,
                        "self_learning": st.session_state.self_learning,
                        "teamwork": st.session_state.teamwork,
                        "extra_courses": st.session_state.extra_courses,
                        "long_hours": st.session_state.long_hours,
                        "talent_tests": st.session_state.talent_tests,
                        "olympiads": st.session_state.olympiads,
                        "hours_per_day": st.session_state.hours_per_day,
                        "mgt_or_tech": st.session_state.mgt_or_tech,
                        "salary_or_work": st.session_state.salary_or_work,
                        "worker_type": st.session_state.worker_type,
                        "introvert": st.session_state.introvert_value,
                        "workshops": st.session_state.workshops,
                        "certifications": st.session_state.certifications
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
    # PREMADE_QUESTIONS = [
    #     "What skills should I focus on for my top career recommendation?",
    #     "How can I improve my profile to be more competitive?",
    #     "What certifications would you recommend for my career path?",
    #     "Should I pursue higher education or gain work experience first?",
    #     "What are the typical salary ranges for my recommended careers?",
    #     "How can I transition from my current field to my recommended career?",
    #     "What networking strategies should I follow?",
    #     "What are the emerging trends in my recommended field?"
    # ]

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
    # Chat Rendering
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
    # Save Chat to PDF
    # ==============================
    def save_chat_to_pdf(chat_history):
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y = height - 50
        pdf.setFont("Helvetica", 11)

        for msg in chat_history:
            role = "You" if msg["role"] == "user" else "Career Advisor"
            text = f"{role}: {msg['content']}"
            for line in text.split("\n"):
                pdf.drawString(50, y, line)
                y -= 15
                if y < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 11)
                    y = height - 50
        pdf.save()
        buffer.seek(0)
        return buffer

    # ==============================
    # Streamlit UI
    # ==============================
    st.title("💼 AI Career Counselor")

    status_col1, status_col2 = st.columns([3, 1])
    with status_col2:
        if TINYLLAMA_AVAILABLE:
            st.success("TinyLlama Ready")
        else:
            st.warning("TinyLlama Unavailable")

    # Init states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input_area" not in st.session_state:
        st.session_state.user_input_area = ""

    # ==============================
    # Quick Questions
    # ==============================
    # st.markdown("### Quick Questions")
    # cols = st.columns(4)
    # for idx, question in enumerate(PREMADE_QUESTIONS):
    #     col_idx = idx % 4
    #     with cols[col_idx]:
    #         if st.button(question[:25] + "...", key=f"premade_{idx}", use_container_width=True):
    #             st.session_state.user_input_area = question

    # st.markdown("---")

    # ==============================
    # Conversation
    # ==============================
    st.markdown("### Conversation")
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            render_chat_message(message)
    else:
        st.info("Start a conversation by asking a question or clicking a quick question above 👆")

    # ==============================
    # Chat Input
    # ==============================
    user_question = st.text_area(
        "Your Question",
        value=st.session_state.user_input_area,
        height=100,
        placeholder="Ask anything about your career...",
        key="user_input_box"
    )

    # ==============================
    # Buttons
    # ==============================
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        send = st.button("Send", type="primary", use_container_width=True)
    with col2:
        clear = st.button("Clear Chat", use_container_width=True)
    with col3:
        save_pdf = st.button("Save Chat to PDF", use_container_width=True)

    # ==============================
    # Button Actions
    # ==============================
    if send and user_question.strip():
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

    if clear:
        st.session_state.chat_history = []
        st.session_state.user_input_area = ""

    if save_pdf and st.session_state.chat_history:
        pdf_buffer = save_chat_to_pdf(st.session_state.chat_history)
        st.download_button(
            label="📄 Download Chat PDF",
            data=pdf_buffer,
            file_name="career_chat.pdf",
            mime="application/pdf",
            use_container_width=True
        )