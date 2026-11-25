import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import date, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="MehnatMeter AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

STUDENT_FILE = DATA_DIR / "students.csv"
MOCK_FILE = DATA_DIR / "mock_tests.csv"
REVISION_FILE = DATA_DIR / "revision_data.csv"
HISTORY_FILE = DATA_DIR / "study_history.csv"
MODEL_FILE = MODEL_DIR / "burnout_model.pkl"

FEATURES = [
    "study_hours",
    "sleep_hours",
    "breaks_per_day",
    "stress_level",
    "fatigue_level",
    "consecutive_study_days"
]

RISK_COLORS = {
    "Low": "#22C55E",
    "Medium": "#F59E0B",
    "High": "#EF4444"
}

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1120;
    color: #F1F5F9;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #CBD5E1;
}

[data-testid="stSidebar"] {
    background: #111B31;
    border-right: 1px solid #253451;
}

[data-testid="stSidebar"] h1 {
    color: #60A5FA !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #17243C, #1B2B49);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #304568;
}

[data-testid="stMetricValue"] {
    color: #60A5FA !important;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #6D28D9);
    color: white;
    border: none;
}

.stDownloadButton > button {
    background: #172B4D;
    color: #93C5FD;
    border: 1px solid #3B82F6;
    border-radius: 10px;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    background-color: #17243A !important;
    color: white !important;
    border-radius: 8px;
}

.stDataFrame {
    border: 1px solid #304568;
    border-radius: 10px;
}

hr {
    border-color: #304568;
}

div[data-testid="stExpander"] {
    background: #14213A;
    border: 1px solid #304568;
    border-radius: 12px;
}

.hero {
    background: linear-gradient(120deg, #172554, #312E81, #4C1D95);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid #4F46E5;
}

.hero h1 {
    font-size: 35px;
}

.hero p {
    font-size: 16px;
    color: #DBEAFE;
}

.info-card {
    background: #17243A;
    border: 1px solid #304568;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.info-card h3 {
    color: #60A5FA !important;
}

.small-note {
    color: #94A3B8;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data
def load_csv(path):

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_resource
def load_model():

    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    return None


students = load_csv(STUDENT_FILE)
mock_data = load_csv(MOCK_FILE)
revision_data = load_csv(REVISION_FILE)
history = load_csv(HISTORY_FILE)

model = load_model()


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def plot_layout(fig, height=400):

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


def predict_risk(values):

    if model is None:
        return None

    input_df = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    return model.predict(input_df)[0]


def risk_badge(risk):

    if risk == "High":
        st.error("🔴 HIGH RISK")

    elif risk == "Medium":
        st.warning("🟡 MEDIUM RISK")

    else:
        st.success("🟢 LOW RISK")


def save_history(values, risk):

    new_record = pd.DataFrame([{
        "date": str(date.today()),
        **values,
        "predicted_risk": risk
    }])

    if HISTORY_FILE.exists():

        old = pd.read_csv(HISTORY_FILE)

        new_record = pd.concat(
            [old, new_record],
            ignore_index=True
        )

    new_record.to_csv(
        HISTORY_FILE,
        index=False
    )

    st.cache_data.clear()


def get_recommendations(values, risk):

    suggestions = []

    if values["study_hours"] > 9:
        suggestions.append(
            "Your study hours are very high. Consider shorter, focused sessions."
        )

    if values["sleep_hours"] < 6:
        suggestions.append(
            "Your sleep duration is low. Try maintaining a consistent sleep routine."
        )

    if values["breaks_per_day"] < 2:
        suggestions.append(
            "Add regular breaks between study sessions."
        )

    if values["stress_level"] >= 7:
        suggestions.append(
            "Try light exercise, breathing exercises or relaxation breaks."
        )

    if values["fatigue_level"] >= 7:
        suggestions.append(
            "Reduce study intensity temporarily and review your workload."
        )

    if values["consecutive_study_days"] > 12:
        suggestions.append(
            "Consider scheduling a lighter day to recover."
        )

    if risk == "High":
        suggestions.append(
            "Your current inputs indicate elevated study-related risk. "
            "Review your routine and seek support if needed."
        )

    if not suggestions:
        suggestions.append(
            "Your current study pattern looks balanced. Continue monitoring progress."
        )

    return suggestions


def create_decay_score(row):

    gap = max(0, float(row["days_since_revision"]))
    score = min(100, max(0, float(row["quiz_score"])))

    gap_risk = min(gap / 30, 1) * 60
    score_risk = ((100 - score) / 100) * 40

    return round(gap_risk + score_risk, 1)


def priority_label(value):

    if value >= 65:
        return "High"

    elif value >= 35:
        return "Medium"

    return "Low"


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown("""
<h1>📚 MehnatMeter AI</h1>
<p style="color:#94A3B8;">Mehnat Kitni, Progress Kaisi?</p>
""", unsafe_allow_html=True)

pages = [
    "🏠 Dashboard",
    "🧠 Burnout Predictor",
    "🎯 What-If Simulator",
    "🤖 AI Study Coach",
    "📝 Study Tracker",
    "📊 Performance Analysis",
    "📚 Knowledge Retention",
    "🧩 Quiz Arena",
    "📅 Smart Exam Planner",
    "📈 Personal Analytics",
    "📄 My Report"
]
page = st.sidebar.radio(
    "NAVIGATION",
    pages
)

st.sidebar.divider()

st.sidebar.markdown("""
<div class="info-card">
<h3>🚀 Project Mission</h3>
<p>Study smarter, track progress and identify patterns before they affect your academic journey.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("MehnatMeter AI | Student Intelligence System")


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to MehnatMeter AI 🚀</h1>
        <p>Understand your study patterns. Improve your performance.
        Build a smarter study routine.</p>
    </div>
    """, unsafe_allow_html=True)

    if students.empty:

        st.error("Student dataset not found. Check data/students.csv")

    else:

        total = len(students)

        avg_study = students["study_hours"].mean()
        avg_sleep = students["sleep_hours"].mean()

        avg_stress = students["stress_level"].mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Students Analysed", total)
        c2.metric("Avg Study Hours", f"{avg_study:.1f} hrs")
        c3.metric("Avg Sleep", f"{avg_sleep:.1f} hrs")
        c4.metric("Avg Stress", f"{avg_stress:.1f}/10")

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("🔥 Burnout Risk Distribution")

            if "burnout_risk" in students.columns:

                counts = students["burnout_risk"].value_counts()

                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    hole=0.55,
                    color=counts.index,
                    color_discrete_map=RISK_COLORS
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )

        with right:

            st.subheader("📚 Study vs Sleep")

            fig = px.scatter(
                students,
                x="study_hours",
                y="sleep_hours",
                color="burnout_risk",
                color_discrete_map=RISK_COLORS,
                hover_data=["stress_level", "fatigue_level"]
                if "fatigue_level" in students.columns else None
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        st.subheader("📊 Dataset Overview")

        st.dataframe(
            students.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Explore the navigation menu to access study analytics, "
            "burnout screening, knowledge retention and learning behaviour."
        )


# ==========================================================
# BURNOUT PREDICTOR
# ==========================================================

elif page == "🧠 Burnout Predictor":

    st.title("🧠 Burnout Risk Predictor")

    st.write(
        "Enter your current study routine to estimate study-related risk."
    )

    if model is None:

        st.error("Model file not found: models/burnout_model.pkl")

    else:

        c1, c2 = st.columns(2)

        with c1:

            study = st.slider("Daily Study Hours", 0.0, 16.0, 6.0)
            sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
            breaks = st.slider("Breaks Per Day", 0, 10, 3)

        with c2:

            stress = st.slider("Stress Level", 1, 10, 5)
            fatigue = st.slider("Fatigue Level", 1, 10, 4)
            consecutive = st.slider(
                "Consecutive Study Days",
                0, 30, 5
            )

        if st.button("🔍 Predict My Risk"):

            values = {
                "study_hours": study,
                "sleep_hours": sleep,
                "breaks_per_day": breaks,
                "stress_level": stress,
                "fatigue_level": fatigue,
                "consecutive_study_days": consecutive
            }

            risk = predict_risk(values)

            st.divider()

            st.subheader("Your Prediction")

            risk_badge(risk)

            save_history(values, risk)

            st.subheader("💡 Recommendations")

            for item in get_recommendations(values, risk):
                st.info(item)

            st.caption(
                "This is an educational risk-screening tool, "
                "not a medical diagnosis."
            )


# ==========================================================
# WHAT IF SIMULATOR
# ==========================================================

elif page == "🎯 What-If Simulator":

    st.title("🎯 What-If Study Simulator")

    st.write(
        "Experiment with different study routines and compare predicted risk."
    )

    if model is None:

        st.error("Model unavailable.")

    else:

        st.subheader("Current Routine")

        c1, c2 = st.columns(2)

        with c1:

            current_study = st.slider(
                "Current Study Hours", 0.0, 16.0, 9.0
            )

            current_sleep = st.slider(
                "Current Sleep Hours", 0.0, 12.0, 5.0
            )

            current_breaks = st.slider(
                "Current Breaks", 0, 10, 1
            )

        with c2:

            current_stress = st.slider(
                "Current Stress", 1, 10, 8
            )

            current_fatigue = st.slider(
                "Current Fatigue", 1, 10, 8
            )

            current_days = st.slider(
                "Consecutive Days", 0, 30, 15
            )

        st.divider()

        st.subheader("Proposed Routine")

        new_study = st.slider(
            "New Study Hours", 0.0, 16.0, 6.0
        )

        new_sleep = st.slider(
            "New Sleep Hours", 0.0, 12.0, 8.0
        )

        new_breaks = st.slider(
            "New Breaks", 0, 10, 4
        )

        new_stress = st.slider(
            "New Stress", 1, 10, 4
        )

        new_fatigue = st.slider(
            "New Fatigue", 1, 10, 3
        )

        new_days = st.slider(
            "New Consecutive Days", 0, 30, 5
        )

        if st.button("Compare Study Routines"):

            old = {
                "study_hours": current_study,
                "sleep_hours": current_sleep,
                "breaks_per_day": current_breaks,
                "stress_level": current_stress,
                "fatigue_level": current_fatigue,
                "consecutive_study_days": current_days
            }

            new = {
                "study_hours": new_study,
                "sleep_hours": new_sleep,
                "breaks_per_day": new_breaks,
                "stress_level": new_stress,
                "fatigue_level": new_fatigue,
                "consecutive_study_days": new_days
            }

            old_risk = predict_risk(old)
            new_risk = predict_risk(new)

            a, b = st.columns(2)

            with a:
                st.subheader("Current Routine")
                risk_badge(old_risk)

            with b:
                st.subheader("Proposed Routine")
                risk_badge(new_risk)

            st.success(
                "Comparison complete. These are model estimates, "
                "not guaranteed outcomes."
            )


# ==========================================================
# AI STUDY COACH
# ==========================================================

elif page == "🤖 AI Study Coach":

    st.title("🤖 AI Study Coach")

    st.write(
        "Get recommendations based on your current study habits."
    )

    if model is None:

        st.error("Model unavailable.")

    else:

        study = st.number_input("Study Hours", 0.0, 16.0, 6.0)
        sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
        breaks = st.number_input("Breaks", 0, 10, 3)
        stress = st.slider("Stress", 1, 10, 5)
        fatigue = st.slider("Fatigue", 1, 10, 5)
        days = st.number_input("Consecutive Days", 0, 30, 5)

        if st.button("Generate My Study Plan"):

            values = {
                "study_hours": study,
                "sleep_hours": sleep,
                "breaks_per_day": breaks,
                "stress_level": stress,
                "fatigue_level": fatigue,
                "consecutive_study_days": days
            }

            risk = predict_risk(values)

            st.subheader("Study Pattern Assessment")
            risk_badge(risk)

            for suggestion in get_recommendations(values, risk):
                st.info(suggestion)

            st.subheader("Suggested Daily Structure")

            plan = pd.DataFrame({
                "Session": [
                    "Session 1",
                    "Session 2",
                    "Session 3",
                    "Session 4"
                ],
                "Activity": [
                    "Concept Learning",
                    "Practice Questions",
                    "Revision",
                    "Mock Test / Analysis"
                ],
                "Suggested Duration": [
                    "90 minutes",
                    "60 minutes",
                    "45 minutes",
                    "45 minutes"
                ]
            })

            st.dataframe(
                plan,
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# EARLY WARNING
# ==========================================================

elif page == "🚨 Early Warning":

    st.title("🚨 Academic Early Warning System")

    if students.empty:

        st.warning("Student data unavailable.")

    else:

        warning_data = students.copy()

        warning_data["Warning Score"] = (
            (warning_data["study_hours"] > 9).astype(int) * 2
            + (warning_data["sleep_hours"] < 6).astype(int) * 2
            + (warning_data["stress_level"] >= 7).astype(int) * 2
            + (warning_data["fatigue_level"] >= 7).astype(int) * 2
            + (warning_data["breaks_per_day"] < 2).astype(int)
        )

        warning_data["Attention Level"] = np.select(
            [
                warning_data["Warning Score"] >= 6,
                warning_data["Warning Score"] >= 3
            ],
            [
                "High",
                "Medium"
            ],
            default="Low"
        )

        counts = warning_data["Attention Level"].value_counts()

        c1, c2, c3 = st.columns(3)

        c1.metric("High Attention", counts.get("High", 0))
        c2.metric("Medium Attention", counts.get("Medium", 0))
        c3.metric("Low Attention", counts.get("Low", 0))

        fig = px.histogram(
            warning_data,
            x="Warning Score",
            color="Attention Level",
            color_discrete_map=RISK_COLORS
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        st.subheader("Students Requiring Attention")

        st.dataframe(
            warning_data.sort_values(
                "Warning Score",
                ascending=False
            ).head(20),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Warning Score is a rule-based indicator, not a clinical assessment."
        )


# ==========================================================
# STUDY TRACKER
# ==========================================================

elif page == "📝 Study Tracker":

    st.title("📝 Daily Study Tracker")

    st.write("Record your daily study routine.")

    c1, c2 = st.columns(2)

    with c1:

        study = st.number_input("Study Hours", 0.0, 16.0, 6.0)
        sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
        breaks = st.number_input("Breaks", 0, 10, 3)

    with c2:

        stress = st.slider("Stress Level", 1, 10, 5)
        fatigue = st.slider("Fatigue Level", 1, 10, 4)
        days = st.number_input("Consecutive Study Days", 0, 30, 5)

    if st.button("Save Today's Record"):

        if model is None:

            st.error("Model unavailable.")

        else:

            values = {
                "study_hours": study,
                "sleep_hours": sleep,
                "breaks_per_day": breaks,
                "stress_level": stress,
                "fatigue_level": fatigue,
                "consecutive_study_days": days
            }

            risk = predict_risk(values)

            save_history(values, risk)

            st.success("Your study record has been saved!")

            risk_badge(risk)

    st.divider()

    st.subheader("📚 Study History")

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        if "study_hours" in history.columns:

            fig = px.line(
                history,
                x="date",
                y="study_hours",
                markers=True,
                title="Daily Study Hours"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        st.download_button(
            "Download Study History",
            history.to_csv(index=False),
            file_name="study_history.csv",
            mime="text/csv"
        )

    else:

        st.info("No study records available yet.")


# ==========================================================
# PERFORMANCE ANALYSIS
# ==========================================================

elif page == "📊 Performance Analysis":

    st.title("📊 Student Performance Analysis")

    if mock_data.empty:

        st.warning("mock_tests.csv not found.")

    else:

        if "score" not in mock_data.columns:

            st.error("Score column not found.")

        else:

            avg_score = mock_data["score"].mean()
            best_score = mock_data["score"].max()
            lowest_score = mock_data["score"].min()

            c1, c2, c3 = st.columns(3)

            c1.metric("Average Score", f"{avg_score:.1f}%")
            c2.metric("Best Score", f"{best_score:.1f}%")
            c3.metric("Lowest Score", f"{lowest_score:.1f}%")

            st.divider()

            if "subject" in mock_data.columns:

                st.subheader("Subject-wise Performance")

                subject_scores = (
                    mock_data.groupby("subject")["score"]
                    .mean()
                    .reset_index()
                    .sort_values("score", ascending=False)
                )

                fig = px.bar(
                    subject_scores,
                    x="subject",
                    y="score",
                    color="score",
                    color_continuous_scale="Blues",
                    text_auto=".1f"
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )

            st.subheader("Score Distribution")

            fig = px.histogram(
                mock_data,
                x="score",
                nbins=20
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            if "test_no" in mock_data.columns:

                st.subheader("Test-wise Score Trend")

                trend = (
                    mock_data.groupby("test_no")["score"]
                    .mean()
                    .reset_index()
                )

                fig = px.line(
                    trend,
                    x="test_no",
                    y="score",
                    markers=True
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )

            st.dataframe(
                mock_data,
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# KNOWLEDGE RETENTION
# ==========================================================

elif page == "📚 Knowledge Retention":

    st.title("📚 Knowledge Retention Analysis")

    if revision_data.empty:

        st.warning("revision_data.csv not found.")

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Topics Analysed",
            len(revision_data)
        )

        c2.metric(
            "Average Quiz Score",
            f"{revision_data['quiz_score'].mean():.1f}%"
        )

        c3.metric(
            "Average Revision Gap",
            f"{revision_data['days_since_revision'].mean():.1f} days"
        )

        st.divider()

        if "subject" in revision_data.columns:

            subject_avg = (
                revision_data.groupby("subject")["quiz_score"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                subject_avg,
                x="subject",
                y="quiz_score",
                color="quiz_score",
                color_continuous_scale="Viridis"
            )

            st.subheader("Subject-wise Retention")

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        fig = px.scatter(
            revision_data,
            x="days_since_revision",
            y="quiz_score",
            color="subject" if "subject" in revision_data.columns else None
        )

        st.subheader("Revision Gap vs Quiz Score")

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        st.dataframe(
            revision_data,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# KNOWLEDGE DECAY PREDICTOR
# ==========================================================

elif page == "🧠 Knowledge Decay Predictor":

    st.title("🧠 Knowledge Decay Predictor")

    st.write(
        "Identify topics that may need revision based on revision gaps "
        "and quiz performance."
    )

    if revision_data.empty:

        st.warning("Revision dataset not available.")

    else:

        required = [
            "subject",
            "topic",
            "days_since_revision",
            "quiz_score"
        ]

        if not all(col in revision_data.columns for col in required):

            st.error("Required columns are missing from revision_data.csv.")

        else:

            retention = revision_data.copy()

            subjects_list = sorted(
                retention["subject"].dropna().unique()
            )

            selected_subject = st.selectbox(
                "Select Subject",
                subjects_list
            )

            subject_data = retention[
                retention["subject"] == selected_subject
            ].copy()

            subject_data["Decay Risk"] = subject_data.apply(
                create_decay_score,
                axis=1
            )

            subject_data["Revision Priority"] = (
                subject_data["Decay Risk"].apply(priority_label)
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Average Quiz Score",
                f"{subject_data['quiz_score'].mean():.1f}%"
            )

            c2.metric(
                "Average Revision Gap",
                f"{subject_data['days_since_revision'].mean():.1f} days"
            )

            c3.metric(
                "High Priority Topics",
                int(
                    (subject_data["Revision Priority"] == "High").sum()
                )
            )

            st.divider()

            st.subheader("Topic-wise Knowledge Risk")

            fig = px.bar(
                subject_data,
                x="topic",
                y="Decay Risk",
                color="Revision Priority",
                color_discrete_map=RISK_COLORS,
                text_auto=".1f"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            st.subheader("Revision Gap vs Quiz Performance")

            fig = px.scatter(
                subject_data,
                x="days_since_revision",
                y="quiz_score",
                color="Revision Priority",
                color_discrete_map=RISK_COLORS,
                hover_data=["topic"]
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            st.subheader("🎯 Smart Revision Recommendations")

            priority_data = subject_data.sort_values(
                "Decay Risk",
                ascending=False
            )

            for _, row in priority_data.iterrows():

                message = (
                    f"{row['topic']} | "
                    f"Quiz Score: {row['quiz_score']}% | "
                    f"Revision Gap: {row['days_since_revision']} days"
                )

                if row["Revision Priority"] == "High":
                    st.error("🔴 " + message)

                elif row["Revision Priority"] == "Medium":
                    st.warning("🟡 " + message)

                else:
                    st.success("🟢 " + message)

            st.download_button(
                "Download Knowledge Decay Report",
                priority_data.to_csv(index=False),
                file_name="knowledge_decay_report.csv",
                mime="text/csv"
            )

            st.caption(
                "Decay Risk is an illustrative heuristic, not a validated "
                "scientific measurement of memory decay."
            )


# ==========================================================
# STUDENT LEARNING DNA
# ==========================================================

elif page == "🧬 Student Learning DNA":

    st.title("🧬 Student Learning DNA")

    st.write(
        "Discover student learning behaviour patterns using "
        "K-Means Machine Learning."
    )

    if students.empty:

        st.warning("Student dataset unavailable.")

    else:

        dna = students.copy()

        if not mock_data.empty and "student_id" in mock_data.columns:

            scores = (
                mock_data.groupby("student_id")["score"]
                .mean()
                .reset_index()
                .rename(columns={"score": "average_score"})
            )

            dna = dna.merge(
                scores,
                on="student_id",
                how="left"
            )

        else:

            dna["average_score"] = 0

        if not revision_data.empty:

            revision = (
                revision_data.groupby("student_id")
                .agg(
                    avg_revision_gap=("days_since_revision", "mean"),
                    avg_quiz_score=("quiz_score", "mean")
                )
                .reset_index()
            )

            dna = dna.merge(
                revision,
                on="student_id",
                how="left"
            )

        else:

            dna["avg_revision_gap"] = 0
            dna["avg_quiz_score"] = 0

        cluster_features = [
            "study_hours",
            "sleep_hours",
            "stress_level",
            "fatigue_level",
            "average_score",
            "avg_revision_gap",
            "avg_quiz_score"
        ]

        dna[cluster_features] = dna[
            cluster_features
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        dna[cluster_features] = dna[
            cluster_features
        ].fillna(
            dna[cluster_features].median()
        ).fillna(0)

        if len(dna) < 4:

            st.warning("At least 4 student records are needed.")

        else:

            scaler = StandardScaler()

            scaled = scaler.fit_transform(
                dna[cluster_features]
            )

            kmeans = KMeans(
                n_clusters=4,
                random_state=42,
                n_init=10
            )

            dna["Cluster"] = kmeans.fit_predict(scaled)

            # Interpret clusters using their average characteristics
            profiles = dna.groupby("Cluster")[cluster_features].mean()

            median_score = dna["average_score"].median()
            median_study = dna["study_hours"].median()
            median_gap = dna["avg_revision_gap"].median()

            cluster_names = {}

            for cluster_id, row in profiles.iterrows():

                if (
                    row["average_score"] >= median_score
                    and row["study_hours"] <= median_study
                ):

                    label = "Efficient Learner"

                elif (
                    row["study_hours"] >= median_study
                    and row["average_score"] < median_score
                ):

                    label = "High-Effort Learner"

                elif row["avg_revision_gap"] >= median_gap:

                    label = "Revision-Needs Attention"

                elif row["stress_level"] >= 7:

                    label = "High-Stress Learner"

                else:

                    label = "Balanced Learner"

                cluster_names[cluster_id] = label

            dna["Learning Type"] = dna["Cluster"].map(
                cluster_names
            )

            st.success("Learning behaviour clusters generated!")

            c1, c2, c3 = st.columns(3)

            c1.metric("Students Analysed", len(dna))
            c2.metric("Learning Groups", dna["Cluster"].nunique())
            c3.metric(
                "Average Score",
                f"{dna['average_score'].mean():.1f}%"
            )

            st.divider()

            st.subheader("Learning DNA Distribution")

            counts = dna["Learning Type"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.55
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            st.subheader("Study Behaviour Map")

            fig = px.scatter(
                dna,
                x="study_hours",
                y="average_score",
                color="Learning Type",
                hover_data=[
                    "student_id",
                    "sleep_hours",
                    "stress_level"
                ]
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            st.subheader("Learning Profiles")

            st.dataframe(
                dna[
                    [
                        "student_id",
                        "study_hours",
                        "average_score",
                        "avg_revision_gap",
                        "avg_quiz_score",
                        "Learning Type"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Learning DNA Report",
                dna.to_csv(index=False),
                file_name="learning_dna.csv",
                mime="text/csv"
            )

            st.caption(
                "These are exploratory data-driven groups, not fixed "
                "personality types or scientifically validated learning styles."
            )


# ==========================================================
# EXPLAINABLE AI
# ==========================================================

elif page == "🔍 Explainable AI":

    st.title("🔍 Explainable AI")

    st.write(
        "Understand which factors influence the model's predictions."
    )

    if model is None:

        st.warning("Model unavailable.")

    else:

        actual_model = model

        if hasattr(model, "named_steps"):

            actual_model = list(
                model.named_steps.values()
            )[-1]

        if hasattr(actual_model, "feature_importances_"):

            importance = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": actual_model.feature_importances_
            }).sort_values(
                "Importance",
                ascending=False
            )

            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Feature importance is not available for this model type."
            )


# ==========================================================
# PERSONAL ANALYTICS
# ==========================================================

elif page == "📈 Personal Analytics":

    st.title("📈 Personal Study Analytics")

    if history.empty:

        st.info("Save study records to view your personal analytics.")

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Study Records",
            len(history)
        )

        c2.metric(
            "Average Study Hours",
            f"{history['study_hours'].mean():.1f}"
        )

        c3.metric(
            "Average Sleep",
            f"{history['sleep_hours'].mean():.1f}"
        )

        st.divider()

        fig = px.line(
            history,
            x="date",
            y="study_hours",
            markers=True,
            title="Study Hours Trend"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        fig = px.line(
            history,
            x="date",
            y="stress_level",
            markers=True,
            title="Stress Trend"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        if "predicted_risk" in history.columns:

            counts = history["predicted_risk"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map=RISK_COLORS,
                hole=0.5
            )

            st.subheader("Your Risk History")

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )


# ==========================================================
# SMART EXAM PLANNER
# ==========================================================

elif page == "📅 Smart Exam Planner":

    st.title("📅 Smart Exam Planner")

    st.write(
        "Create a revision schedule based on subject difficulty "
        "and available study hours."
    )

    subjects = [
        "DBMS",
        "Operating Systems",
        "Computer Networks",
        "Data Structures",
        "Algorithms",
        "TOC",
        "Discrete Mathematics"
    ]

    exam_date = st.date_input(
        "Exam Date",
        date.today() + timedelta(days=30),
        min_value=date.today()
    )

    hours = st.slider(
        "Daily Available Study Hours",
        1, 12, 5
    )

    selected_subjects = st.multiselect(
        "Select Subjects",
        subjects,
        default=subjects[:5]
    )

    if st.button("Generate Revision Plan"):

        if not selected_subjects:

            st.warning("Select at least one subject.")

        else:

            days_left = max(
                1,
                (exam_date - date.today()).days
            )

            total_hours = days_left * hours

            weights = np.ones(
                len(selected_subjects)
            )

            if not mock_data.empty:

                averages = mock_data.groupby("subject")["score"].mean()

                for i, subject in enumerate(selected_subjects):

                    if subject in averages.index:

                        weights[i] = max(
                            0.5,
                            (100 - averages[subject]) / 100
                        )

            allocations = (
                weights / weights.sum()
            ) * total_hours

            plan = pd.DataFrame({
                "Subject": selected_subjects,
                "Allocated Hours": np.round(allocations, 1)
            })

            st.success(
                f"Your plan covers {days_left} days "
                f"and approximately {total_hours} study hours."
            )

            st.dataframe(
                plan,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Study Plan",
                plan.to_csv(index=False),
                file_name="smart_exam_plan.csv",
                mime="text/csv"
            )


# ==========================================================
# WELLNESS SCORE
# ==========================================================

elif page == "💚 Wellness Score":

    st.title("💚 Study Wellness Indicator")

    st.write(
        "A simple routine-balance indicator based on your study inputs."
    )

    if students.empty:

        st.warning("Student dataset unavailable.")

    else:

        avg_study = students["study_hours"].mean()
        avg_sleep = students["sleep_hours"].mean()
        avg_stress = students["stress_level"].mean()
        avg_fatigue = students["fatigue_level"].mean()

        study_component = max(
            0,
            100 - abs(avg_study - 6) * 8
        )

        sleep_component = min(
            100,
            avg_sleep / 8 * 100
        )

        stress_component = max(
            0,
            100 - avg_stress * 10
        )

        fatigue_component = max(
            0,
            100 - avg_fatigue * 10
        )

        wellness = (
            study_component * 0.20
            + sleep_component * 0.30
            + stress_component * 0.25
            + fatigue_component * 0.25
        )

        st.metric(
            "Study Wellness Indicator",
            f"{wellness:.1f}/100"
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=wellness,
                title={"text": "Routine Balance"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#3B82F6"},
                    "steps": [
                        {"range": [0, 40], "color": "#7F1D1D"},
                        {"range": [40, 70], "color": "#78350F"},
                        {"range": [70, 100], "color": "#14532D"}
                    ]
                }
            )
        )

        st.plotly_chart(
            plot_layout(fig, 350),
            use_container_width=True
        )

        st.caption(
            "This is a custom educational indicator, not a validated "
            "health or wellness assessment."
        )


# ==========================================================
# MY REPORT
# ==========================================================

elif page == "📄 My Report":

    st.title("📄 My Study Report")

    st.write(
        "Download your study analytics and performance report."
    )

    if students.empty:

        st.warning("Student dataset unavailable.")

    else:

        st.subheader("Dataset Summary")

        summary = pd.DataFrame({
            "Metric": [
                "Total Students",
                "Average Study Hours",
                "Average Sleep Hours",
                "Average Stress",
                "Average Fatigue"
            ],
            "Value": [
                len(students),
                round(students["study_hours"].mean(), 2),
                round(students["sleep_hours"].mean(), 2),
                round(students["stress_level"].mean(), 2),
                round(students["fatigue_level"].mean(), 2)
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Student Dataset",
            students.to_csv(index=False),
            file_name="mehnatmeter_student_report.csv",
            mime="text/csv"
        )

        if not mock_data.empty:

            st.download_button(
                "📥 Download Performance Data",
                mock_data.to_csv(index=False),
                file_name="performance_report.csv",
                mime="text/csv"
            )

        if not revision_data.empty:

            st.download_button(
                "📥 Download Revision Data",
                revision_data.to_csv(index=False),
                file_name="revision_report.csv",
                mime="text/csv"
            )

        if not history.empty:

            st.download_button(
                "📥 Download Personal History",
                history.to_csv(index=False),
                file_name="personal_study_history.csv",
                mime="text/csv"
            )
# ==========================================================
# QUIZ ARENA
# ==========================================================

elif page == "🧩 Quiz Arena":

    from pathlib import Path
    from datetime import datetime
    import uuid
    import pandas as pd

    st.title("🧩 Quiz Arena")
    st.write("Challenge yourself and test your knowledge!")

    BASE_DIR = Path(__file__).resolve().parent
    QUESTIONS_FILE = BASE_DIR / "data" / "quiz_questions.csv"
    RESULTS_FILE = BASE_DIR / "data" / "quiz_results.csv"

    if not QUESTIONS_FILE.exists():
        st.error("Quiz questions file not found.")

    else:
        quiz_data = pd.read_csv(QUESTIONS_FILE)

        required_columns = [
            "subject", "topic", "question",
            "option_a", "option_b", "option_c",
            "option_d", "answer", "explanation"
        ]

        if not all(col in quiz_data.columns for col in required_columns):
            st.error("Quiz CSV columns are missing. Please check the questions file.")

        elif quiz_data.empty:
            st.warning("No quiz questions available.")

        else:
            subjects = sorted(
                quiz_data["subject"].dropna().astype(str).unique()
            )

            selected_subject = st.selectbox(
                "Select Subject",
                ["All Subjects"] + subjects
            )

            filtered = quiz_data.copy()

            if selected_subject != "All Subjects":
                filtered = filtered[
                    filtered["subject"].astype(str) == selected_subject
                ]

            topics = sorted(
                filtered["topic"].dropna().astype(str).unique()
            )

            selected_topic = st.selectbox(
                "Select Topic",
                ["All Topics"] + topics
            )

            if selected_topic != "All Topics":
                filtered = filtered[
                    filtered["topic"].astype(str) == selected_topic
                ]

            if filtered.empty:
                st.warning("No questions found.")

            else:
                st.info(f"Total Questions: {len(filtered)}")

                student_name = st.text_input(
                    "Student Name",
                    value="Student"
                )

                with st.form("mehnatmeter_quiz_form"):

                    answers = {}

                    for number, (index, row) in enumerate(
                        filtered.iterrows(), start=1
                    ):

                        st.markdown(
                            f"### Q{number}. {row['question']}"
                        )

                        options = [
                            f"A. {row['option_a']}",
                            f"B. {row['option_b']}",
                            f"C. {row['option_c']}",
                            f"D. {row['option_d']}"
                        ]

                        answers[index] = st.radio(
                            "Choose your answer:",
                            options,
                            index=None,
                            key=f"mehnat_quiz_{index}"
                        )

                        st.divider()

                    submitted = st.form_submit_button(
                        "Submit Quiz",
                        use_container_width=True
                    )

                if submitted:

                    score = 0
                    explanations = []

                    for index, row in filtered.iterrows():

                        selected = answers[index]
                        correct = str(row["answer"]).strip().upper()

                        selected_letter = (
                            selected[0] if selected else ""
                        )

                        if selected_letter == correct:
                            score += 1

                        explanations.append({
                            "question": row["question"],
                            "selected": selected or "Not answered",
                            "correct": correct,
                            "explanation": row["explanation"]
                        })

                    total = len(filtered)
                    accuracy = round(score / total * 100, 2)

                    result = {
                        "attempt_id": str(uuid.uuid4()),
                        "student_id": student_name or "Student",
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "subject": selected_subject,
                        "topic": selected_topic,
                        "total_questions": total,
                        "correct_answers": score,
                        "accuracy": accuracy,
                        "elapsed_seconds": 0
                    }

                    st.session_state["quiz_result"] = result
                    st.session_state["quiz_explanations"] = explanations

                    result_columns = list(result.keys())

                    try:
                        if RESULTS_FILE.exists():
                            old_results = pd.read_csv(RESULTS_FILE)
                        else:
                            old_results = pd.DataFrame(
                                columns=result_columns
                            )

                        all_columns = list(dict.fromkeys(
                            list(old_results.columns) + result_columns
                        ))

                        old_results = old_results.reindex(
                            columns=all_columns
                        )

                        new_row = pd.DataFrame(
                            [result]
                        ).reindex(columns=all_columns)

                        updated = pd.concat(
                            [old_results, new_row],
                            ignore_index=True
                        )

                        updated.to_csv(RESULTS_FILE, index=False)

                    except Exception as error:
                        st.error(f"Could not save result: {error}")

                if "quiz_result" in st.session_state:

                    result = st.session_state["quiz_result"]

                    st.subheader("🏆 Quiz Result")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Score",
                        f"{result['correct_answers']}/{result['total_questions']}"
                    )

                    col2.metric(
                        "Accuracy",
                        f"{result['accuracy']}%"
                    )

                    col3.metric(
                        "Wrong Answers",
                        result["total_questions"]
                        - result["correct_answers"]
                    )

                    if result["accuracy"] >= 80:
                        st.success("Excellent performance! 🎉")
                    elif result["accuracy"] >= 50:
                        st.info("Good effort! Keep practicing.")
                    else:
                        st.warning("Revise the topic and try again.")

                    st.subheader("📘 Answer Explanations")

                    for number, item in enumerate(
                        st.session_state["quiz_explanations"],
                        start=1
                    ):

                        with st.expander(f"Question {number}"):

                            st.write(item["question"])
                            st.write(f"Your answer: {item['selected']}")
                            st.write(f"Correct answer: {item['correct']}")
                            st.write(f"Explanation: {item['explanation']}")


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">

<h3 style="color:#60A5FA;">📚 MehnatMeter AI</h3>

<p style="color:#94A3B8;">
Smart Study Analytics • Performance Intelligence • Knowledge Retention
</p>

<p style="color:#64748B; font-size:12px;">
Built with Python, Streamlit, Scikit-learn and Plotly
</p>

</div>
""", unsafe_allow_html=True)