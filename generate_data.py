
import pandas as pd
import numpy as np
import os

np.random.seed(42)

os.makedirs("data", exist_ok=True)

# -------------------------------
# 1. STUDENT DATASET
# -------------------------------

n = 500

students = pd.DataFrame({
    "student_id": range(1, n + 1),
    "study_hours": np.random.uniform(1, 12, n).round(1),
    "sleep_hours": np.random.uniform(3, 9, n).round(1),
    "breaks_per_day": np.random.randint(0, 8, n),
    "stress_level": np.random.randint(1, 11, n),
    "fatigue_level": np.random.randint(1, 11, n),
    "consecutive_study_days": np.random.randint(1, 20, n)
})

# Illustrative labels for initial model development
risk_score = (
    (students["study_hours"] > 9).astype(int) * 2 +
    (students["sleep_hours"] < 5).astype(int) * 2 +
    (students["stress_level"] > 7).astype(int) * 2 +
    (students["fatigue_level"] > 7).astype(int) * 2 +
    (students["breaks_per_day"] < 2).astype(int) +
    (students["consecutive_study_days"] > 12).astype(int)
)

students["burnout_risk"] = np.where(
    risk_score >= 7, "High",
    np.where(risk_score >= 4, "Medium", "Low")
)

students.to_csv("data/students.csv", index=False)


# -------------------------------
# 2. MOCK TEST DATASET
# -------------------------------

subjects = ["DBMS", "Operating Systems", "Computer Networks",
            "Data Structures", "Algorithms"]

topics = {
    "DBMS": ["Normalization", "SQL", "Transactions"],
    "Operating Systems": ["Deadlock", "Scheduling", "Memory Management"],
    "Computer Networks": ["TCP/IP", "Routing", "OSI Model"],
    "Data Structures": ["Trees", "Graphs", "Arrays"],
    "Algorithms": ["Sorting", "Searching", "Dynamic Programming"]
}

mock_records = []

for student_id in range(1, n + 1):
    for test_no in range(1, 6):
        subject = np.random.choice(subjects)
        topic = np.random.choice(topics[subject])

        mock_records.append({
            "student_id": student_id,
            "test_no": test_no,
            "subject": subject,
            "topic": topic,
            "score": np.random.randint(25, 101),
            "test_date": pd.Timestamp("2026-10-01") +
                         pd.Timedelta(days=test_no * 5)
        })

mock_tests = pd.DataFrame(mock_records)
mock_tests.to_csv("data/mock_tests.csv", index=False)


# -------------------------------
# 3. REVISION DATASET
# -------------------------------

revision_records = []

for student_id in range(1, n + 1):
    for subject in subjects:
        for topic in topics[subject]:
            revision_records.append({
                "student_id": student_id,
                "subject": subject,
                "topic": topic,
                "days_since_revision": np.random.randint(1, 31),
                "quiz_score": np.random.randint(20, 101),
                "revision_count": np.random.randint(0, 6)
            })

revision_data = pd.DataFrame(revision_records)
revision_data.to_csv("data/revision_data.csv", index=False)

print("================================")
print("MEHNATMETER AI DATASETS CREATED!")
print("================================")

print("Students:", len(students))
print("Mock Tests:", len(mock_tests))
print("Revision Records:", len(revision_data))

print("\nAll CSV files saved inside data folder.")
