
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load datasets
students = pd.read_csv("data/students.csv")
mock = pd.read_csv("data/mock_tests.csv")
revision = pd.read_csv("data/revision_data.csv")

os.makedirs("results", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Burnout Risk Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=students, x="burnout_risk")
plt.title("Student Burnout Risk Distribution")
plt.xlabel("Burnout Risk")
plt.ylabel("Number of Students")
plt.savefig("results/burnout_distribution.png")
plt.show()

# 2. Study Hours vs Burnout
plt.figure(figsize=(8, 5))
sns.boxplot(data=students, x="burnout_risk", y="study_hours")
plt.title("Study Hours vs Burnout Risk")
plt.savefig("results/study_burnout.png")
plt.show()

# 3. Sleep vs Fatigue
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=students,
    x="sleep_hours",
    y="fatigue_level",
    hue="burnout_risk"
)
plt.title("Sleep Hours vs Fatigue")
plt.savefig("results/sleep_fatigue.png")
plt.show()

# 4. Mock Test Performance
plt.figure(figsize=(9, 5))
sns.histplot(mock["score"], bins=15, kde=True)
plt.title("Mock Test Score Distribution")
plt.xlabel("Test Score")
plt.savefig("results/mock_scores.png")
plt.show()

# 5. Subject-wise Performance
subject_scores = mock.groupby("subject")["score"].mean().sort_values()

plt.figure(figsize=(9, 5))
subject_scores.plot(kind="bar")
plt.title("Average Subject-wise Performance")
plt.ylabel("Average Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/subject_performance.png")
plt.show()

# 6. Revision Gap Analysis
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=revision,
    x="days_since_revision",
    y="quiz_score"
)
plt.title("Revision Gap vs Quiz Performance")
plt.xlabel("Days Since Last Revision")
plt.ylabel("Quiz Score")
plt.savefig("results/revision_analysis.png")
plt.show()

# Summary
print("\n========== MEHNATMETER AI EDA ==========")

print("\nStudent Dataset:")
print(students.describe())

print("\nBurnout Risk Counts:")
print(students["burnout_risk"].value_counts())

print("\nAverage Mock Test Score:")
print(mock["score"].mean())

print("\nSubject-wise Average Scores:")
print(subject_scores)

print("\nAverage Revision Quiz Score:")
print(revision["quiz_score"].mean())

print("\nEDA COMPLETED SUCCESSFULLY!")
