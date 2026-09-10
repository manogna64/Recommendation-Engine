# Recommendation Engine

## Overview

The Recommendation Engine is a Python-based internship recommendation system that recommends suitable internships based on student profiles, skills, domain, experience level, and historical ratings.

The system combines **Collaborative Filtering** and **Content-Based Filtering** to generate recommendations and uses a **Hybrid Recommendation** approach to improve the overall recommendation process.

## Features

* Student profile processing
* Collaborative Filtering
* Content-Based Filtering
* Hybrid Recommendation
* Internship similarity calculation
* Recommendation score prediction
* Recommendation evaluation
* RMSE calculation
* Precision@3 and Recall@3 evaluation
* Adding new student profiles
* Generating recommendations for new students

## Recommendation Methods

### 1. Collaborative Filtering

Uses historical student-internship ratings to predict which internships may be suitable for a student.

### 2. Content-Based Filtering

Compares student skills and profile information with internship characteristics to identify similar opportunities.

### 3. Hybrid Recommendation

Combines collaborative filtering and content-based filtering scores to produce a final recommendation ranking.

## Example Student Profile

```text
Student: Alice
Skills: Python, Machine Learning, Data Science
Domain: AI
Experience Level: Intermediate
```

### Example Recommendations

```text
NLP Research Intern
Data Scientist
AI Intern
```

## Evaluation

The recommendation engine is evaluated using a leave-one-out approach over known ratings.

Example results:

```text
RMSE: 1.0
Precision@3: 0.212
Recall@3: 0.778
Evaluated Ratings: 11
```

## New Student Recommendation

The system can also accept a new student profile and generate recommendations.

Example:

```text
Student: Grace
Skills: Python, Computer Vision, OpenCV
Domain: AI
Experience Level: Intermediate
```

The system generates internship recommendations based on the new student's profile.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/recommendation-engine.git
cd recommendation-engine
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project requires:

```text
numpy
pandas
scikit-learn
```

## How to Run

Run the following command:

```bash
python internship_recommender.py
```

The program will display the student's profile, recommendations from different methods, evaluation metrics, and recommendations for a newly added student.

## Project Structure

```text
recommendation-engine/
│
├── internship_recommender.py
├── README.md
├── requirements.txt
└── screenshots/
```

## Future Improvements

* Add a larger internship and student dataset
* Implement a web-based user interface
* Add more recommendation algorithms
* Improve recommendation accuracy with larger datasets
* Connect the system to a database
* Provide personalized internship filtering and search

## Author

Developed as part of a virtual internship project.
