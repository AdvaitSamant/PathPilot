# PathPilot: AI-Driven Career Mentor

PathPilot is an intelligent career guidance platform that helps users identify suitable career paths in technology through AI-based analysis.  
It combines machine learning, behavioral assessment, and interactive quizzes to generate personalized career recommendations and insights.

---

## Overview

PathPilot allows users to:
- Take a short, category-based skill and personality quiz.  
- Receive a career prediction powered by a trained Random Forest model.  
- Chat with an AI mentor trained on their quiz data.  
- Generate a personalized PDF report with insights and recommendations.  
- Store user data, quiz scores, and predictions in a structured CSV file.

---

## Key Features

### AI Career Prediction
- Evaluates user responses to technical and behavioral questions.  
- Predicts the most compatible tech career path using machine learning.  
- Provides detailed analysis of each skill and personality category.

### AI Career Mentor
- Context-aware chatbot powered by OpenRouter API.  
- Offers real-time guidance, certification advice, and project ideas.  
- Customizes responses based on quiz results and stored user data.

### Automated Reporting
- Generates downloadable, professional PDF career reports.  
- Includes user information, category-wise scores, and personalized insights.  
- Fully compatible with `fpdf` and `fpdf2` libraries.

### Data Management
- Automatically stores all user data and predictions in `user_results.csv`.  
- Maintains session states for smooth navigation.  
- Processes data locally with no external storage or sharing.

---

## Technology Stack

| Component | Technology |
|------------|-------------|
| Frontend/UI | Streamlit |
| Backend/Logic | Python |
| Machine Learning | Scikit-learn, Joblib |
| Data Storage | CSV Files |
| AI Integration | OpenRouter API |
| Reporting | FPDF |
| Animation & Icons | Lottie, Lucide Icons |

---

## System Workflow

1. User provides personal details (name, age, location, hobbies, and goals).  
2. Quiz module loads 3 randomized questions per category from CSV files.  
3. User responses are scored automatically.  
4. Scores are fed into a trained Random Forest model for prediction.  
5. A career report is generated in PDF format.  
6. User data and results are stored in `user_results.csv`.  
7. The AI Career Mentor chatbot provides personalized career guidance.

---

## Project Structure

```
PathPilot/
│
├── app.py                     # Main Streamlit application
├── career_model_main.pkl      # Trained Random Forest model
├── label_encoder.pkl          # Label encoder for predicted career
├── quiz_data/                 # Folder containing quiz CSV files
├── user_results.csv           # Auto-generated user data log
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PathPilot.git
cd PathPilot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

Activate the environment:  
- Windows: `venv\Scripts\activate`  
- macOS/Linux: `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure Model Files Exist
Place the following files in the project root:
- `career_model_main.pkl`  
- `label_encoder.pkl`

### 5. Run the Application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Machine Learning Model

**Algorithm:** Random Forest Classifier  
**Input Features:** 14 skill and personality scores  
**Output:** Predicted career role (decoded using label encoder)

| Feature | Description |
|----------|-------------|
| Computer Architecture | Understanding of hardware systems |
| Programming Skills | Ability to write and structure code |
| Project Management | Efficiency in planning and execution |
| Communication Skills | Clarity and collaboration |
| Openness | Creativity and curiosity |
| Conscientiousness | Organization and reliability |
| Extraversion | Leadership and social interaction |
| Agreeableness | Cooperation and empathy |
| Emotional Range | Composure under pressure |
| Conversation | Verbal expressiveness |
| Openness to Change | Adaptability and innovation |
| Hedonism | Motivation through enjoyment |
| Self-enhancement | Drive for recognition and growth |
| Self-transcendence | Ethical and community focus |

---

## Data Logging Example

Each user’s data is stored in `user_results.csv` in this format:

| Timestamp | Full Name | Age | Gender | City | Country | Predicted Career | Programming Skills | Communication Skills | ... |
|------------|------------|-----|---------|------|----------|------------------|-------------------|----------------------|-----|
| 2025-11-08 12:45:21 | Advait Samant | 19 | Male | Pune | India | AI ML Specialist | 8 | 9 | ... |

---

## Security and Privacy

- All user data is processed locally within Streamlit.  
- No data is transmitted to external servers.  
- CSV files are overwritten safely to avoid duplication.  
- The OpenRouter API key is securely managed within the app.

---

## Future Improvements

- Add authentication and personalized dashboards.  
- Include career progress tracking and goal milestones.  
- Integrate resume and project evaluation modules.  
- Add multilingual quiz support.  
- Deploy data synchronization using cloud databases (Firebase, Supabase).

---

## Author

**Advait Samant**  
B.Tech Computer Science Engineering – MIT ADT University  
Aspiring Data Scientist and AI Engineer  

---

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software with appropriate credit.
