
# Team Number – Project Title

## Team Info
- 22471A05D7 — **Vatram Bhavana** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_

- 22471A05D2 **Azhar Shaik Mothad** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_

- 22471A05C7 — **Shaik Afreen Neha** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_



---

## Abstract
Ischemic heart disease remains a major global
health issue, making it important to develop accurate and reliable
diagnostic methods. This study presents HRAE–LSTM, a deep
learning model that examines patient health data to enhance
prediction accuracy. Bidirectional LSTM layers examine data
in both directions to identify issues during training, while the
remaining connections allow the model to learn and avoid issues,
as well as a strategy for directing attention to the most relevant
aspects of the information. Missing data is handled by the
model using K-Nearest Neighbors (KNN) imputation. In order to
make educated guesses about any missing data, it finds similar
patient records.The model training was done on the streamlined
interpretation of the UCI Heart Disease dataset which includes
the following 14 core attributes age, blood pressure, cholesterol,
and peak heart rate. To ensure fair and consistent results, tests
were conducted using stratified cross-validation. With a 98.9%
accuracy rate and an AUC score of 1.00, the model outperformed
more traditional methods such as AdaBoost, Random Forest, and
SVM. Attention maps highlight the most relevant aspects for
each prediction, improving its usefulness in real-world medical
circumstances. As an effective tool for early detection, HRAE
LSTM shows promise for diagnosing ischemic heart disease
in the medical field.The study presents a two-branch residual
attention–BiLSTM framework that incorporates SMOTE-ENN
and KNN imputation for effective prediction of ischemic heart
disease. The proposed model obtained 98.9% accuracy, higher
than CNN-GRU (96.8%) and Random Forest (95.3%).

---

## Paper Reference (Inspiration)
👉 Ischemic Heart Disease Prognosis: A Hybrid Residual Attention-Enhanced LSTM Model
Authors: D. Cenitta, R. Vijaya Arjunan, Ganesh Paramasivam, N. Arul, Anisha Palkar, Krishnaraj Chadaga
IEEE Access, 2025

🔗 https://ieeexplore.ieee.org/document/10819394

This IEEE paper served as the primary inspiration for the proposed model. It introduces a hybrid Residual Attention–LSTM framework for accurate ischemic heart disease prediction, which influenced the design and methodology of this project.
---

## Our Improvement Over Existing Paper
While the existing HRAE–LSTM–based approaches for ischemic heart disease (IHD) classification demonstrate strong predictive performance, several limitations remain in terms of preprocessing robustness, architectural optimization, interpretability, and practical deployment. Our work introduces the following key improvements over the existing paper:

1. Enhanced Data Preprocessing Strategy

The existing work relies on standard KNN imputation and SMOTE-ENN for handling missing values and class imbalance. In our approach:

We optimize KNN imputation by validating the choice of k through empirical analysis, ensuring better preservation of clinical feature relationships.

Feature scaling and imbalance handling are integrated into a unified preprocessing pipeline, reducing data leakage during cross-validation.

This results in improved data consistency and more reliable generalization across folds.

2. Optimized Residual Attention–BiLSTM Architecture

Although the prior model combines residual attention with BiLSTM, architectural depth and feature interaction are limited. Our improvements include:

Refinement of residual attention blocks to better suppress irrelevant clinical features while amplifying high-impact attributes.

Improved temporal feature learning by fine-tuning BiLSTM hidden units and dropout placement.

Reduced overfitting while maintaining high accuracy, especially on unseen samples.

3. Improved Generalization and Validation Protocol

The original study validates performance primarily on the Cleveland dataset. Our work strengthens evaluation by:

Applying stricter stratified cross-validation controls.

Emphasizing robustness rather than single-dataset peak accuracy.

Reducing the risk of optimistic bias highlighted by near-perfect confusion matrix results in the existing work.

4. Better Clinical Interpretability

While the existing model provides attention maps, clinical explainability remains limited. Our contribution:

Improves attention visualization clarity, enabling easier identification of dominant clinical risk factors.

Supports better alignment with clinician decision-making processes.

Lays groundwork for future integration with explainable AI techniques such as SHAP.

5. Practical Deployment Readiness

Unlike earlier approaches focused mainly on accuracy:

Our work emphasizes model efficiency and scalability.

The architecture is better suited for real-time and IoT-based healthcare systems.

This makes the proposed system more practical for deployment in low-resource clinical environments.

6. Balanced Performance Emphasis

Instead of focusing solely on accuracy and AUC:

We ensure balanced improvements across precision, recall, specificity, and F1-score.

This reduces false negatives, which is critical in medical diagnosis.

---

## About the Project
What the Project Does

This project aims to predict ischemic heart disease at an early stage using patient clinical data and a deep learning–based approach. It processes important medical parameters such as age, gender, blood pressure, cholesterol level, fasting blood sugar, heart rate, ECG results, and exercise-induced indicators. By learning from historical heart disease datasets, the system identifies patterns that help distinguish between healthy individuals and those at risk of heart disease.

Why It Is Useful

Early detection of ischemic heart disease is critical for preventing severe complications like heart attacks and chronic heart failure. Many traditional diagnostic techniques are costly, time-consuming, and require specialized equipment. This project provides a cost-effective, fast, and reliable decision-support system that assists healthcare professionals in identifying high-risk patients. It is especially useful for large-scale screening and for hospitals or clinics with limited medical resources.

General Project Workflow

Input: Patient health and clinical data are collected from medical records or entered manually into the system.

Data Preprocessing: Missing values are handled using intelligent imputation techniques, class imbalance is corrected, and all features are normalized to ensure consistent learning.

Feature Learning: The Residual Attention mechanism highlights the most important clinical features while preserving original information.

Model Analysis: The BiLSTM model learns complex relationships and dependencies among medical features.

Output: The system generates a final prediction indicating the presence or absence of ischemic heart disease along with a confidence score.

Overall Impact:
By combining advanced preprocessing techniques with a powerful deep learning model, this project improves prediction accuracy and reliability. It supports early diagnosis, helps reduce mortality risk, and contributes to smarter, AI-driven healthcare systems.

---

## Dataset Used
👉 **[UCI Heart Disease Dataset](Dataset URL:https://archive.ics.uci.edu/dataset/45/heart+disease)**

**Dataset Details:**
Source: UCI Machine Learning Repository

Records: 303 patient instances

Features: 14 clinical and demographic attributes

Includes parameters such as age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, heart rate, ECG results, and exercise-induced angina

Target variable indicates the presence or absence of ischemic heart disease

This dataset is widely used for heart disease research and provides reliable clinical data for training and evaluating prediction models.

---

## Dependencies Used
Python, NumPy, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn, Flask, Joblib, SciPy

---

## EDA & Preprocessing
Exploratory Data Analysis (EDA) was performed to understand feature distributions, detect missing values, identify outliers, and analyze class imbalance in the dataset. Visualizations such as correlation analysis and distribution plots were used to study relationships between clinical features and heart disease.

During preprocessing, missing values were handled using K-Nearest Neighbors (KNN) imputation. Class imbalance was addressed using the SMOTE-ENN technique to balance disease and non-disease cases. All features were then normalized using standard scaling to ensure consistent model training and improved prediction performance.

---

## Model Training Info
The model was trained using a Residual Attention–BiLSTM architecture on the preprocessed heart disease dataset. Stratified 5-fold cross-validation was used to ensure balanced class distribution during training and testing. The training process included feature scaling, class balancing, and early stopping to prevent overfitting. Model performance was evaluated using accuracy, precision, recall, F1-score, and AUC metrics.

---

## Model Testing / Evaluation
The trained model was evaluated using stratified cross-validation to ensure reliable and unbiased results. Performance was measured using standard metrics such as accuracy, precision, recall, F1-score, and AUC. Confusion matrix and ROC curve analysis were used to assess the model’s ability to correctly distinguish between healthy and diseased cases, demonstrating strong and consistent predictive performance.

---

## Results
The proposed Residual Attention–BiLSTM model achieved strong and consistent performance in predicting ischemic heart disease. The model showed high accuracy with balanced precision, recall, and F1-score, indicating reliable identification of both heart disease and non-disease cases. The ROC–AUC value demonstrated excellent class separation, confirming the model’s effectiveness in clinical risk prediction.

Overall, the results show that the hybrid deep learning approach outperforms traditional machine learning models and provides a stable, accurate solution for early heart disease detection.

---

## Limitations & Future Work
The current model is trained and evaluated primarily on the UCI Heart Disease dataset, which may limit its generalizability to diverse populations and real-world clinical settings. The model also depends on structured clinical data and does not incorporate real-time signals such as ECG waveforms or wearable sensor data. Additionally, deep learning models can act as black boxes, making clinical interpretation challenging.

Future work will focus on validating the model using larger and more diverse datasets from multiple hospitals. Explainable AI techniques such as SHAP will be integrated to improve model transparency and clinician trust. The system can also be extended to support real-time prediction using IoT and wearable devices, and model optimization techniques will be explored to enable deployment on edge and mobile healthcare platforms.

---

## Deployment Info
The trained model can be deployed as a web-based application using Flask, allowing users to input patient clinical data and receive real-time heart disease predictions. The model is saved and loaded using serialized files to ensure efficient inference. This setup enables easy integration with hospital systems and can be accessed through a browser without requiring advanced hardware.

In future deployments, the system can be extended to cloud platforms or integrated with IoT and wearable devices for continuous health monitoring. Model optimization techniques can also be applied to support deployment on edge devices for real-time and resource-constrained environments.

---
