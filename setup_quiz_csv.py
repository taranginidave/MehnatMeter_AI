import os
import pandas as pd

# Create data folder if it does not exist
os.makedirs("data", exist_ok=True)

# -------------------------------
# QUIZ QUESTIONS
# -------------------------------

questions = [
    {
        "subject": "Python",
        "topic": "Basics",
        "question": "Which keyword is used to define a function in Python?",
        "option_a": "function",
        "option_b": "def",
        "option_c": "define",
        "option_d": "fun",
        "answer": "B",
        "explanation": "Python uses the def keyword to define a function."
    },
    {
        "subject": "Python",
        "topic": "Data Types",
        "question": "Which data type is used to store True or False?",
        "option_a": "int",
        "option_b": "str",
        "option_c": "bool",
        "option_d": "list",
        "answer": "C",
        "explanation": "The bool data type stores True or False values."
    },
    {
        "subject": "Python",
        "topic": "Lists",
        "question": "Which method adds an element to the end of a list?",
        "option_a": "append()",
        "option_b": "add()",
        "option_c": "insert_end()",
        "option_d": "push()",
        "answer": "A",
        "explanation": "The append() method adds an element at the end of a list."
    },
    {
        "subject": "DBMS",
        "topic": "SQL",
        "question": "Which SQL command is used to retrieve data?",
        "option_a": "INSERT",
        "option_b": "UPDATE",
        "option_c": "SELECT",
        "option_d": "DELETE",
        "answer": "C",
        "explanation": "SELECT is used to retrieve data from database tables."
    },
    {
        "subject": "DBMS",
        "topic": "Keys",
        "question": "Which key uniquely identifies each record in a table?",
        "option_a": "Foreign Key",
        "option_b": "Primary Key",
        "option_c": "Candidate Key",
        "option_d": "Composite Key",
        "answer": "B",
        "explanation": "A primary key uniquely identifies each record in a table."
    },
    {
        "subject": "Data Structures",
        "topic": "Stack",
        "question": "Which principle does a stack follow?",
        "option_a": "FIFO",
        "option_b": "LIFO",
        "option_c": "FILO only",
        "option_d": "Random order",
        "answer": "B",
        "explanation": "A stack follows Last In, First Out (LIFO)."
    },
    {
        "subject": "Data Structures",
        "topic": "Queue",
        "question": "Which principle does a queue follow?",
        "option_a": "LIFO",
        "option_b": "FIFO",
        "option_c": "Random order",
        "option_d": "Priority only",
        "answer": "B",
        "explanation": "A queue follows First In, First Out (FIFO)."
    },
    {
        "subject": "Computer Networks",
        "topic": "OSI Model",
        "question": "How many layers are present in the OSI model?",
        "option_a": "4",
        "option_b": "5",
        "option_c": "7",
        "option_d": "8",
        "answer": "C",
        "explanation": "The OSI model contains seven layers."
    },
    {
        "subject": "Operating Systems",
        "topic": "Processes",
        "question": "Which of these is a process scheduling algorithm?",
        "option_a": "FCFS",
        "option_b": "HTML",
        "option_c": "SQL",
        "option_d": "HTTP",
        "answer": "A",
        "explanation": "FCFS stands for First Come, First Served scheduling."
    },
    {
        "subject": "Machine Learning",
        "topic": "Basics",
        "question": "Which learning method uses labelled training data?",
        "option_a": "Unsupervised Learning",
        "option_b": "Supervised Learning",
        "option_c": "Reinforcement Learning only",
        "option_d": "Random Learning",
        "answer": "B",
        "explanation": "Supervised learning trains models using labelled data."
    }
]

questions_df = pd.DataFrame(questions)

questions_path = "data/quiz_questions.csv"
questions_df.to_csv(questions_path, index=False)

print(f"Added {len(questions_df)} questions to {questions_path}")


# -------------------------------
# QUIZ RESULTS CSV
# -------------------------------

results_columns = [
    "attempt_id",
    "student_id",
    "date",
    "subject",
    "topic",
    "total_questions",
    "correct_answers",
    "accuracy",
    "elapsed_seconds"
]

results_path = "data/quiz_results.csv"

# Create results file with headers if it does not exist
if not os.path.exists(results_path):
    pd.DataFrame(columns=results_columns).to_csv(
        results_path,
        index=False
    )
    print(f"Created results file: {results_path}")
else:
    print(f"Results file already exists: {results_path}")

print("\nCSV setup completed successfully!")