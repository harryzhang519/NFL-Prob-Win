NFL Win Probability Web App

A Python-based web application that estimates NFL game win probabilities using historical team-level performance data.

Overview

This project builds a simple machine learning pipeline that engineers weekly team features from historical NFL data and uses a logistic regression model to estimate win probabilities. The model is exposed through a FastAPI backend for inference.

Features

Team-level feature engineering (EPA differential, success rate differential, Elo differential, home-field advantage)

Logistic regression model for win probability estimation

FastAPI backend for serving predictions

Data processing using Polars and pandas

Tech Stack

- Python

- FastAPI

- Uvicorn

- Polars

- pandas

- scikit-learn

Running the App Locally

1. Clone the repository

2. Install dependencies
   pip install -r requirements.txt
3. Start the server
   uvicorn app:app --reload
4. Access the API locally at http://127.0.0.1:8000

Notes

This project is intended as a learning-focused, applied data science and backend development exercise. Model performance and feature design can be improved with additional data, tuning, and validation.
