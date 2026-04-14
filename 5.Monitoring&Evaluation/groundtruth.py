import os
import pandas as pd
import random
import json
from tqdm import tqdm

# Load dataset
df = pd.read_csv("../clean_data/employee_data.csv")

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Question templates
TEMPLATES = {
    "easy": [
        "What is the job title of {name}?",
        "What department does {name} work in?",
        "What is the salary of {name}?",
        "Is {name} working full-time or part-time?"
    ],
    "medium": [
        "Who works as a {job_titles} in the {department}?",
        "Which employee earns {annual_salary} annually?",
        "Who works in the {department} with the title {job_titles}?",
        "Find the employee with hourly rate {hourly_rate}"
    ],
    "hard": [
        "Identify the employee who works in {department}, earns {annual_salary}, and has the title {job_titles}.",
        "Who is a {job_titles} working full-time in {department} earning {annual_salary}?",
        "Find the employee with hourly rate {hourly_rate} and typical hours {typical_hours} in {department}.",
        "Which employee matches: {job_titles}, {department}, salary {annual_salary}, full-time status {full_or_part_time}?"
    ]
}

PREFIXES = [
    "",
    "Can you tell me ",
    "I'm looking for ",
    "Do you know ",
    "Help me find ",
    "Please identify "
]

def generate_question(row, difficulty):
    template = random.choice(TEMPLATES[difficulty])
    prefix = random.choice(PREFIXES)

    try:
        return (prefix + template.format(**row)).strip()
    except KeyError:
        return None

def format_output(row):
    return {
        "name": row["name"],
        "job_titles": row["job_titles"],
        "department": row["department"],
        "full_or_part_time": row["full_or_part_time"],
        "salary_or_hourly": row["salary_or_hourly"],
        "annual_salary": row["annual_salary"],
        "typical_hours": row["typical_hours"],
        "hourly_rate": row["hourly_rate"]
    }

dataset = []

# Generate dataset
for _, row in tqdm(df.iterrows(), total=len(df)):
    row_dict = row.to_dict()

    for difficulty in ["easy", "medium", "hard"]:
        for _ in range(3):
            q = generate_question(row_dict, difficulty)
            if not q:
                continue

            example = {
                "inputs": {  
                    "question": q
                },
                "outputs": {  
                    "answer": format_output(row_dict)
                },
                "metadata": {
                    "difficulty": difficulty,
                    "source": "synthetic"
                }
            }

            dataset.append(example)

# Save as JSON (for LangSmith API)
try:
    os.makedirs('../groundtruth', exist_ok=True)

    with open("../groundtruth/groundtruth_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} examples successfully!")

except Exception as e:
    print("Save Failed:", e)