import streamlit as st
import pandas as pd
import joblib

# Load model
model_data = joblib.load('career_recommendation_model.pkl')
preprocessor = model_data['preprocessor']
rf_models_tuned = model_data['models']

# Recommendation function
def recommend_careers_tuned(input_data: dict):
    input_df = pd.DataFrame([input_data])
    input_processed = preprocessor.transform(input_df)
    
    recommendations = {}
    for role, model in rf_models_tuned.items():
        prob = model.predict_proba(input_processed)[0][1]
        recommendations[role] = prob
    
    ranked_roles = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return ranked_roles

# Streamlit app
st.title("🎓 Career Recommendation System")

# Qualification dropdown
qualifications = [
    "Arts - Information Technology - University of Sri Jayewardenepura",
    "Computer Science - University of Colombo School of Computing (UCSC)",
    "Computer Science - University of Jaffna",
    "Computer Science - University of Ruhuna",
    "Computer Science - Trincomalee Campus, Eastern University, Sri Lanka",
    "Physical Science - ICT - University of Kelaniya",
    "Physical Science - ICT - University of Sri Jayewardenepura",
    "Artificial Intelligence - University of Moratuwa",
    "Electronics and Computer Science - University of Kelaniya",
    "Information Systems - University of Colombo, School of Computing (UCSC)",
    "Information Systems - University of Sri Jayewardenepura",
    "Information Systems - Sabaragamuwa University of Sri Lanka",
    "Data Science - Sabaragamuwa University of Sri Lanaka",
    "Information Technology (IT) - University of Moratuwa",
    "Management and Information Technology (MIT) - University of Kelaniya",
    "Computer Science & Technology - Uva Wellassa University of Sri Lanka",
    "Information Communication Technology - University of Sri Jayewardenepura",
    "Information Communication Technology - University of Kelaniya",
    "Information Communication Technology - University of Vavuniya, Sri Lanka",
    "Information Communication Technology - University of Ruhuna",
    "Information Communication Technology - South Eastern University of Sri Lanka",
    "Information Communication Technology - Rajarata University of Sri Lanka",
    "Information Communication Technology - University of Colombo",
    "Information Communication Technology - Uva Wellassa University of Sri Lanka",
    "Information Communication Technology - Eastern University, Sri Lanka"
]

qualification = st.selectbox("🎓 Select Your Qualification", qualifications)

# Language proficiency multi-select
languages = st.multiselect(
    "🌐 Select Language Proficiency (You can select more than one)",
    ['English', 'Sinhala', 'Tamil']
)
language_combined = ", ".join(languages)

# Internship status
internship = st.selectbox("💼 Previous Internship Experience", ['None', 'Yes'])

# Skills multi-select
skills = st.multiselect(
    "🧠 Select Your Skills",
    [
        "Python", "Java", "SQL", "JavaScript", "TensorFlow", "Pandas", "Docker",
        "Kubernetes", "HTML/CSS", "Power BI", "Spark", "AWS", "Azure",
        "Linux", "Tableau", "React", "Node.js"
    ]
)

if st.button("🚀 Recommend Careers"):
    input_data = {
        'Qualification': qualification,
        'Language Proficiency': language_combined,
        'Previous Internships': internship,
        'Skills': ", ".join(skills)
    }
    results = recommend_careers_tuned(input_data)
    st.subheader("🔝 Top 5 Recommended Careers:")
    for role, score in results[:5]:
        st.write(f"**{role}** — Probability: {score:.2f}")
