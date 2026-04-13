# Hotel Booking Cancellation Prediction Using Machine Learning

## Executive Summary
This project develops a machine learning model to predict whether a hotel booking is likely to be canceled or not. The project is based on the Hotel Booking Demand dataset and applies data preprocessing, model development, model evaluation, and deployment using Streamlit. The final model helps support hotel reservation management by identifying bookings that are at higher risk of cancellation.

## Problem Statement
Hotels often face financial and operational challenges due to booking cancellations. A high number of cancellations can lead to revenue loss, inefficient room allocation, and poor planning. Therefore, it is important to build a predictive system that can identify whether a booking is likely to be canceled. This can help hotel managers make better decisions in pricing, planning, and reservation control.

## Domain
Logistics

## Project Objectives
- To preprocess and clean the hotel booking dataset.
- To identify relevant features affecting booking cancellations.
- To train and compare four machine learning classification models.
- To evaluate the models using suitable performance metrics.
- To select the best-performing model.
- To deploy the final model using Streamlit.

## Dataset Source
The dataset used in this project is the *Hotel Booking Demand* dataset. It contains booking information for both a city hotel and a resort hotel, including booking details such as lead time, arrival date, number of guests, room type, and customer type.

## Methodology
The methodology of this project consists of the following stages:

### 1. Data Preprocessing
The dataset was loaded and inspected to understand its structure. Missing values in selected columns were handled properly. Columns with excessive missing values, such as agent and company, were removed. Duplicate rows were also removed. In addition, columns that may cause data leakage, such as reservation_status and reservation_status_date, were dropped. Categorical variables were encoded into numerical form using one-hot encoding.

### 2. Model Development
Four classification models were trained and tested:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### 3. Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 4. Model Selection
Based on the evaluation results, *Random Forest* achieved the best overall performance and was selected as the final model.

### 5. Deployment
The selected Random Forest model was saved and deployed in a Streamlit web application. The application allows the user to select a booking record and predict whether it is likely to be canceled or not.

## Results
The performance of the trained models is summarized below:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.8419 | 0.7631 | 0.6112 | 0.6788 |
| Logistic Regression | 0.7914 | 0.6658 | 0.4749 | 0.5543 |
| Decision Tree | 0.7883 | 0.6107 | 0.6204 | 0.6155 |
| KNN | 0.7085 | 0.4553 | 0.3414 | 0.3902 |

The Random Forest model performed better than the other models in terms of overall accuracy, precision, and F1 score. Therefore, it was selected as the final model for deployment.

## Streamlit Application
The Streamlit application includes:
- A sidebar for selecting a booking row number
- Display of the selected booking data
- Prediction of booking cancellation status
- Probability of canceled and not canceled outcomes
## Acknowledgement

I would like to thank our lecturer for the guidance and support throughout this project. I also appreciate the effort and cooperation of all group members in completing this work successfully.

## Project Structure
```bash
finalproject/
│
├── app.py
├── best_model.pkl
├── requirements.txt
├── README.md
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
├── images/
│  ├── streamlit2_app.jpg
│  ├── confusion_matrix.jpg
│  ├── streamlit_app.jpg
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_development.ipynb
│   └── 3_model_testing.ipynb
│
├── models/
│   └── best_model.pkl
│
├── slides/
│   └── final_presentation.pptx
│
└── data/
    └── hotel_bookings.csv
