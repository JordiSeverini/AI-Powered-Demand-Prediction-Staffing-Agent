# AI-Powered Predictive Scheduling System for Restaurants

## Overview

This project is an AI-powered decision system that predicts restaurant demand and generates optimized staff schedules.

It demonstrates an AI-native workflow combining:
- Machine Learning prediction
- Rule-based decisioning
- LLM-powered explanations

The goal is to improve staffing efficiency, reduce operational costs, and enhance customer experience.

---

## Problem Statement

Restaurants often face unpredictable demand due to:
- Day-of-week patterns
- Weather conditions
- Local events

This leads to:
- Overstaffing and unnecessary costs
- Understaffing and reduced service quality
- Reactive rather than proactive scheduling decisions

---

## Solution

This system predicts demand and generates staffing recommendations through three stages:

### 1. Machine Learning Prediction Layer

A Logistic Regression model is trained on synthetic data to predict demand levels:
- Low
- High

Features used:
- Day of week
- Weather conditions
- Event presence

---

### 2. Decision Engine

A rule-based system converts predicted demand into staffing recommendations:
- Servers
- Kitchen staff
- Hosts

This ensures operational constraints are translated into actionable staffing plans.

---

### 3. LLM Explanation Layer

A large language model generates human-readable explanations of staffing decisions.

This helps translate technical outputs into business-friendly insights for stakeholders.

---

## Architecture

User Input  
↓  
Machine Learning Model (Demand Prediction)  
↓  
Rule-Based Staffing Engine  
↓  
LLM Explanation Layer  
↓  
Final Output (Schedule + Justification)

---

## Example Output

Input:
- Day: Saturday
- Weather: Good
- Event: False

Output:

```bash 
Demand: high

 Schedule:
- servers: 6
- kitchen: 4
- hosts: 2

 Explanation:
- The staffing levels align with predicted high demand, ensuring efficient service. 

Staffing Justification:
- Six servers provide adequate coverage for customer needs during peak hours.
- Four kitchen staff can handle a higher volume of orders without compromising food quality.
- Having two hosts optimizes guest flow and minimizes wait times at peak entry periods.
```
## Explanation

Demand is classified as high due to weekend traffic patterns. Staffing levels are increased to maintain service efficiency. Additional kitchen staff ensures timely order fulfillment and reduced wait times.

---

## Tech Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn (Logistic Regression)  
- OpenAI API (GPT-4o-mini)  
- Rule-based decision engine  

---
# Key Concepts Demonstrated
- Supervised machine learning
- Feature engineering (one-hot encoding)
- Hybrid AI system design
- Prompt engineering