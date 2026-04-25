import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

client = OpenAI()


# CREATE SYNTHETIC DATASET
# -----------------------------

def create_data(n=500):

    # Intitialize empyt list 
    data = []

# Loop n times to generate data 
    for _ in range(n):

        day = np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        weather = np.random.choice(["good", "bad"])
        event = np.random.choice([0, 1])

        # Truth Function
        score = 0
        if day in ["Fri", "Sat", "Sun"]:
            score += 1
        if weather == "good":
            score += 1
        if event == 1:
            score += 1

        # Target label     
        demand = 1 if score >= 2 else 0  # 1 = high demand
        
        # appends list to the data list
        data.append([day, weather, event, demand])

    # create a Dataframe from data
    return pd.DataFrame(data, columns=["day", "weather", "event", "demand"])


# PREP DATA FOR ML
# -----------------------------
# Generate Synthetic data 
df = create_data()

# Converts categorical (text) data into numbers so machine learning models can use it.
df = pd.get_dummies(df, columns=["day", "weather"])


# Seperate Features from target variable
# Remove the demand column from df 
X = df.drop("demand", axis=1)

# store the demand column in y
y = df["demand"]

# Split dataset into training and testing sets
# 80% of the data is used to train the model (X_train, y_train)
# 20% is used to test the model on unseen data (X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# TRAIN REAL ML MODEL
# -----------------------------

# Create a Logistic Regression model object 
# This model is used for binary classification (e.g., 0 = low demand, 1 = high demand)
# It will learn patterns from the training data (X_train, y_train) to predict demand 
model = LogisticRegression()

# .fit() Looks at all training examples (X_train , y_train pairs)
# Starts with random or default internal parameters (weights)
# Runs an optimization process to reduce error
# Adjusts its internal weights step-by-step until it reaches convergence 
model.fit(X_train, y_train)



# PREDICTION FUNCTION
# -----------------------------
# prediction function that formats raw inputs (day, weather, event)
# into the same one-hot encoded structure used during model training,
# then uses the trained model to predict demand (0 = low, 1 = high).
# Finally converts the numeric prediction into a human-readable label.
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


# STAFFING ENGINE
# -----------------------------
def generate_schedule(demand):

    if demand == "low":
        return {"servers": 2, "kitchen": 2, "hosts": 1}
    else:
        return {"servers": 6, "kitchen": 4, "hosts": 2}


# AI EXPLANATION LAYER
# -----------------------------
# Uses OpenAI Chat Completions API to generate a staffing explanation based on demand and schedule.
# Sends system + user prompts, then extracts the first response text from response.choices[0].message.content

def explain(demand, schedule):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a restaurant operations analyst explaining staffing decisions.
                Return your answer in this exact format:
                - <1 sentence>

                Staffing Justification:
                - Bullet 1
                - Bullet 2
                - Bullet 3
                Keep it concise. No long paragraphs.
                """
                
            },
            {
                "role": "user",
                "content": f"""
Demand prediction: {demand}
Schedule: {schedule}

Explain why this staffing decision makes sense.
"""
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# 7. FULL AGENT
# -----------------------------
def run_agent(day, weather, event):

    demand = predict_demand(day, weather, event)
    schedule = generate_schedule(demand)
    explanation = explain(demand, schedule)


    # Returns a dictionary 
    return {
        "demand": demand,
        "schedule": schedule,
        "explanation": explanation
    }


# TEST
# -----------------------------
# Problem : Python prints the raw dictionary representation (escaped strings like \n) print(run_agent(...))
# Solution: Print the value using dictionary access 
result = run_agent("Sat", "good", 0)


print("\n Demand:", result["demand"])

print("\n Schedule:")
# Iterate through the schedule dictionary and unpack each key-value pair
# k represents the role (e.g., servers, kitchen, hosts)
# v represents the number of staff assigned to that role
# This allows dynamic, readable formatting of the staffing plan
for k, v in result["schedule"].items():
    print(f"- {k}: {v}")

print("\n Explanation:")
print(result["explanation"])
