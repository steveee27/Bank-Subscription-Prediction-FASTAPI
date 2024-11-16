# Bank Subscription Prediction using Machine Learning and FASTAPI

This project predicts whether a customer is likely to subscribe to a long-term deposit based on their demographic and campaign-related data. It includes a machine learning pipeline for data preprocessing, model training, evaluation, and a RESTful API built with FastAPI for deployment.

---

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [API Endpoints](#api-endpoints)
- [Features](#features)
- [Model Evaluation Results](#model-evaluation-results)
- [License](#license)

---

## Overview

The **Bank Marketing Campaign Prediction** project is designed to help a bank focus its marketing efforts on customers who are most likely to subscribe to long-term deposits. It involves:
- Preprocessing campaign data with feature encoding and scaling.
- Training and evaluating machine learning models (Logistic Regression and Random Forest).
- Deploying the best-performing model through a FastAPI-based API.

### Dataset

The project uses the **Bank Marketing Dataset**, which includes customer demographic data, campaign-related information, and a target variable `y` that indicates whether the customer subscribed to a long-term deposit (`yes`/`no`). The dataset consists of 16 features and is located in the `data/bank-marketing.csv` file.

---

## Technologies Used

- **Python 3.10**
- **FastAPI** for building the RESTful API.
- **Scikit-learn** for preprocessing and machine learning.
- **Pandas** for data manipulation.
- **Uvicorn** for ASGI server.

---

## Project Structure

```plaintext
├── data/
│   └── bank-marketing.csv                # Dataset
├── models/
│   ├── logistic_classifier_best.pkl      # Trained Logistic Regression model
│   ├── robust_scaler.pkl                 # Scaler used during preprocessing
├── src/
│   ├── Training-Model.ipynb              # Notebook for training models
├── main.py                               # FastAPI application
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Pipenv or virtualenv for environment management
- `git` installed on your system

### Installation

1. **Clone the repository:**

    Clone the repository from GitHub to your local machine:
    ```bash
    git clone https://github.com/steveee27/Bank-Subscription-Prediction-FASTAPI.git
    cd Bank-Subscription-Prediction-FASTAPI
    ```

2. **Create and activate a virtual environment:**

    Create a virtual environment to isolate project dependencies:
    ```bash
    python -m venv env
    ```

    Activate the virtual environment:
    - On Linux/MacOS:
        ```bash
        source env/bin/activate
        ```
    - On Windows:
        ```bash
        env\Scripts\activate
        ```

3. **Install project dependencies:**

    Install all required dependencies specified in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the model and scaler files:**

    Ensure the following files are present in the `models/` directory:
    - `logistic_classifier_best.pkl` (the trained model)
    - `robust_scaler.pkl` (the scaler used for preprocessing)

    If the files are missing, refer to the `src/Training-Model.ipynb` notebook to retrain the model and generate these files.

---

### Running the API

1. **Start the FastAPI server:**

    Run the FastAPI server locally:
    ```bash
    uvicorn main:app --reload
    ```

2. **Access the API documentation:**

    Open your browser and navigate to:
    - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints

### **1. GET /**
- **Description**: A welcome endpoint to check if the API is running.
- **Method**: `GET`
- **Request**: No parameters required.
- **Response**:
    ```json
    {
      "message": "Welcome to the Bank Subscription Prediction API!"
    }
    ```

---

### **2. POST /predict**
- **Description**: Predicts whether a customer is likely to subscribe to a long-term deposit based on input data.
- **Method**: `POST`
- **Request Body**: Accepts a JSON object with the following fields:

    | **Field**         | **Type**   | **Description**                               | **Example**          |
    |-------------------|------------|-----------------------------------------------|----------------------|
    | `age`             | `integer`  | Age of the customer.                          | 30                   |
    | `job`             | `string`   | Type of job the customer has.                 | "technician"         |
    | `marital`         | `string`   | Marital status of the customer.               | "single"             |
    | `education`       | `string`   | Education level of the customer.              | "university.degree"  |
    | `default`         | `string`   | Whether the customer has credit in default.   | "no"                 |
    | `housing`         | `string`   | Whether the customer has a housing loan.      | "yes"                |
    | `loan`            | `string`   | Whether the customer has a personal loan.     | "no"                 |
    | `contact`         | `string`   | Contact communication type.                   | "cellular"           |
    | `month`           | `string`   | Last contact month of the year.               | "may"                |
    | `day_of_week`     | `string`   | Last contact day of the week.                 | "mon"                |
    | `duration`        | `integer`  | Last contact duration in seconds.             | 300                  |
    | `campaign`        | `integer`  | Number of contacts during this campaign.      | 1                    |
    | `pdays`           | `integer`  | Number of days since the client was last contacted. | 999             |
    | `previous`        | `integer`  | Number of contacts performed before this campaign. | 0                 |
    | `poutcome`        | `string`   | Outcome of the previous marketing campaign.   | "nonexistent"        |

- **Sample Request**:
    ```json
    {
      "age": 42,
      "job": "admin.",
      "marital": "single",
      "education": "university.degree",
      "default": "no",
      "housing": "yes",
      "loan": "yes",
      "contact": "telephone",
      "month": "may",
      "day_of_week": "wed",
      "duration": 938.0,
      "campaign": 1,
      "pdays": 999,
      "previous": 0,
      "poutcome": "nonexistent"
    }
    ```

- **Response**:
    - A JSON object indicating the prediction result (`yes` or `no`).
    ```json
    {
      "prediction": "yes"
    }
    ```

- **Sample cURL Command**:
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"age":42,"job":"admin.","marital":"single","education":"university.degree","default":"no","housing":"yes","loan":"yes","contact":"telephone","month":"may","day_of_week":"wed","duration":938,"campaign":1,"pdays":999,"previous":0,"poutcome":"nonexistent"}'
    ```

### Notes
- **Data Validation**: The API validates the input data. If any required field is missing or contains invalid data, the API will return an error response.
- **Default Ports**: The API runs on port `8000` by default. Modify this if necessary by updating the `uvicorn` command.

---

## Features

The project is built to provide a complete end-to-end solution for predicting customer subscription likelihood. Below are the core features:

- **Data Preprocessing**: 
  - Handles missing values to ensure data integrity.
  - Encodes categorical features (e.g., job, marital status) into numerical representations for machine learning compatibility.
  - Scales numerical features (e.g., age, duration, campaign) to normalize the data for better model performance.

- **Model Training**:
  - Implements two machine learning algorithms: Logistic Regression and Random Forest.
  - Tunes hyperparameters using GridSearchCV for optimal performance.
  - Evaluates models using precision, recall, F1-score, and accuracy to select the best-performing model.

- **RESTful API**:
  - Deploys the selected model using FastAPI to serve predictions in real-time.
  - Provides user-friendly API endpoints for integration with other systems.
  - Features automated input validation to ensure robust and reliable API interactions.

This combination of features ensures that the project is both technically robust and user-friendly, offering valuable insights and predictions to support marketing decisions.

---

## Model Evaluation Results

This project compares the performance of **Random Forest** and **Logistic Regression** models, tuned using GridSearchCV. Below are the hyperparameters and evaluation metrics for both **Class 0** and **Class 1**:

### Evaluation Metrics

| **Model**              | **Hyperparameters**                                                                                      | **Precision (Class 0)** | **Recall (Class 0)** | **F1-Score (Class 0)** | **Precision (Class 1)** | **Recall (Class 1)** | **F1-Score (Class 1)** | **Accuracy** |
|-------------------------|---------------------------------------------------------------------------------------------------------|-------------------------|----------------------|-------------------------|-------------------------|----------------------|-------------------------|--------------|
| **Random Forest**       | `{'criterion': 'gini', 'max_depth': None, 'min_samples_split': 5, 'n_estimators': 50}`                 | 0.93                    | 0.97                 | 0.95                    | 0.51                    | 0.29                 | 0.37                    | 91%          |
| **Logistic Regression** | `{'penalty': 'l2', 'C': 1, 'max_iter': 100}`                                                           | 0.94                    | 0.97                 | 0.96                    | 0.60                    | 0.37                 | 0.46                    | 92%          |

---

### Insights

1. **Class Imbalance**:
   - The dataset is highly imbalanced, with 1489 examples in **Class 0** (no subscription) and only 154 examples in **Class 1** (subscription).

2. **Random Forest**:
   - **Class 0**: High performance with precision (0.93), recall (0.97), and F1-Score (0.95).
   - **Class 1**: Struggles with recall (0.29) and F1-Score (0.37), indicating a high number of false negatives.

3. **Logistic Regression**:
   - **Class 0**: Slightly better metrics than Random Forest, with precision (0.94), recall (0.97), and F1-Score (0.96).
   - **Class 1**: Outperforms Random Forest with recall (0.37) and F1-Score (0.46), making it better suited for identifying potential subscribers.

4. **Overall Accuracy**:
   - Random Forest: 91%
   - Logistic Regression: 92%
   - While both models achieve high accuracy, this is largely due to the imbalance in the dataset, so accuracy alone is not sufficient to evaluate performance.

---

### Conclusion

**Logistic Regression** is selected as the final model due to its superior performance in predicting the minority class (**Class 1**) while maintaining high performance for the majority class (**Class 0**). This makes it more effective for identifying potential customers likely to subscribe, supporting better marketing decisions.

---

## License

This project is licensed under the [MIT License](./LICENSE).

You are free to use, modify, and distribute this project as long as proper attribution is given to the original author. See the `LICENSE` file for more details.

