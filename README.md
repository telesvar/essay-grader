# IELTS Essay Grader

An AI-powered tool that automatically grades IELTS Task 2 essays using machine learning.

## About

This application uses a trained LightGBM model with BERT embeddings to analyze essays and predict IELTS band scores. The model was trained on a dataset of real IELTS essays with human-assigned scores.

## Features

- Instant essay grading
- Detailed band score descriptions
- Word count tracking
- Example essay loading

## Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Deployment

This app is deployed using Streamlit Cloud. You can access it at [your-app-url].

## Model Information

The grading system uses:
- BERT embeddings to understand essay content and structure
- TF-IDF features to analyze vocabulary usage
- Engineered features for grammar and syntax analysis
- LightGBM regression model for final score prediction
