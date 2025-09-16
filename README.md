# Credit-Card-Fraud-Detection
This project focuses on detecting fraudulent credit card transactions using a hybrid approach that combines Machine Learning (XGBoost) with a Rule-Based Engine for more reliable predictions.
##Model Training:## Built on the well-known European Credit Card Fraud dataset (2013), where transaction features are anonymized into PCA components V1–V28, along with Time, Amount, and Class (fraud/legit).

##Hybrid Detection:##

XGBoost model learns hidden fraud patterns from PCA features.

Rule Engine flags risky cases based on thresholds like transaction amount, frequency, and time of day.

Final fraud score is a weighted blend of both methods (adjustable via Model Weight (α)).

Configurable Thresholds: Users can set Detection Threshold to control sensitivity.

##Modes of Use:#

Single transaction check

Bulk CSV upload for batch analysis (with progress tracking)

Synthetic transaction simulation with adjustable anomaly rate

##Technologies & Concepts Used##

Python, Streamlit for interactive web app

XGBoost for fraud classification

PCA-transformed features (V1–V28) from the original dataset

Explainability with SHAP (SHapley values)

Hybrid Model + Rule Engine for robust detection

Progress bars & batch processing for large-scale analysis

##Usage Guide:##

Model Weight (α): Controls balance between ML model and rule engine.

α = 1 → pure ML model

α = 0 → pure rule engine

Detection Threshold: Defines cutoff for classifying transactions as fraud.

Simulation: Useful for testing system performance under different anomaly rates.
git clone https://github.com/emanfatima-official/credit-card-fraud-detection.git
cd credit-card-fraud-detection

