import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

client = OpenAI()


# Synthetic data generation
# -----------------------------
# Creates a controlled dataset to simulate restaurant demand conditions
def create_data(n=500):
    data = []

    for _ in range(n):
        day = np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        weather = np.random.choice(["good", "bad"])
        event = np.random.choice([0, 1])

        score = 0
        if day in ["Fri", "Sat", "Sun"]:
            score += 1
        if weather == "good":
            score += 1
        if event == 1:
            score += 1

        demand = 1 if score >= 2 else 0
        data.append([day, weather, event, demand])

    return pd.DataFrame(data, columns=["day", "weather", "event", "demand"])



# Data preprocessing
# -----------------------------
df = create_data()

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=["day", "weather"])

X = df.drop("demand", axis=1)
y = df["demand"]

# Train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Model training
# -----------------------------
# Logistic regression model for binary demand classification
model = LogisticRegression()
model.fit(X_train, y_train)


# Inference pipeline
# -----------------------------
# Transforms raw inputs into model-compatible feature format
def predict_demand(day, weather, event):
    input_data = pd.DataFrame([{
        "event": event,
        "day_Fri": 1 if day == "Fri" else 0,
        "day_Mon": 1 if day == "Mon" else 0,
        "day_Sat": 1 if day == "Sat" else 0,
        "day_Sun": 1 if day == "Sun" else 0,
        "day_Thu": 1 if day == "Thu" else 0,
        "day_Tue": 1 if day == "Tue" else 0,
        "day_Wed": 1 if day == "Wed" else 0,
        "weather_bad": 1 if weather == "bad" else 0,
        "weather_good": 1 if weather == "good" else 0,
    }])

    pred = model.predict(input_data)[0]
    return "high" if pred == 1 else "low"



# Rule-based scheduling engine
# -----------------------------
# Converts demand prediction into staffing allocation
def generate_schedule(demand):
    if demand == "low":
        return {"servers": 2, "kitchen": 2, "hosts": 1}
    return {"servers": 6, "kitchen": 4, "hosts": 2}


# LLM explanation layer
# -----------------------------
# Uses OpenAI API to generate natural language justification for decisions
def explain(demand, schedule):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a restaurant operations analyst. "
                    "Provide concise staffing justifications in a structured format."
                )
            },
            {
                "role": "user",
                "content": f"Demand: {demand}\nSchedule: {schedule}"
            }
        ]
    )

    return response.choices[0].message.content


# Full AI decision pipeline
# -----------------------------
def run_agent(day, weather, event):
    demand = predict_demand(day, weather, event)
    schedule = generate_schedule(demand)
    explanation = explain(demand, schedule)

    return {
        "demand": demand,
        "schedule": schedule,
        "explanation": explanation
    }



# Local execution test
# -----------------------------
result = run_agent("Sat", "good", 0)

print("\nDemand:", result["demand"])

print("\nSchedule:")
for role, count in result["schedule"].items():
    print(f"- {role}: {count}")

print("\nExplanation:")
print(result["explanation"])