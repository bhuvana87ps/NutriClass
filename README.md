# 🥗 NutriClass: Food Classification Using Nutritional Data

## 📌 Project Overview
NutriClass is a **GUVI Mini Project** that I developed to demonstrate an **end-to-end Machine Learning workflow** using food nutritional data.  
The project focuses on how nutritional attributes such as calories, protein, carbohydrates, fat, and sugar can be used to **classify food items accurately**.

This project is designed to be **exam-ready, portfolio-ready, and live-evaluation ready**, following industry-style ML practices.

---
## ❓ Problem Statement

In real-world diet planning and nutrition monitoring, users often know their **nutritional targets** but not the **exact food** that satisfies those targets.

Manual identification:
- Is error-prone  
- Lacks consistency  
- Does not scale  

This project automates the process by learning a **one-to-one mapping between nutritional values and food names** using machine learning.

---

## 🧠 Business Assumptions & Design Logic

### Why Food Name as Target Column?
- Each food item represents a **distinct, allowed meal**
- Nutritional values act as a **fingerprint**
- In strict diet scenarios, **only one food is permitted**, not a list

---

### Why Classification Instead of Recommendation?

| Aspect | Recommendation System | NutriClass |
|------|----------------------|------------|
| Output | Top-N foods | Single exact food |
| Control | Flexible | Strict |
| Use Case | Casual diet | Medical / fitness diet |
| Error Tolerance | High | Very low |

This design ensures **diet compliance, automation, and precision**.

---

## 💼 Business Use Cases

- **Smart Dietary Applications**  
  Auto-select food based on nutritional targets  

- **Health Monitoring Tools**  
  Assist dieticians and nutritionists  

- **Food Logging Systems**  
  Automatically classify user-entered nutrition  

- **Educational Platforms**  
  Explain food–nutrition relationships using ML  

- **Meal & Grocery Planning Apps**  
  Suggest exact replacements within constraints  

---

## 📊 Dataset Description

- **Dataset Type:** Tabular  
- **Raw Data:** Synthetic and imbalanced (realistic scenario)  

### Features:
- Calories  
- Protein  
- Carbohydrates  
- Fat  
- Sugar  

- **Target Variable:** `Food_Name`

### Dataset Stages:
- Raw data for realism and imbalance handling  
- Processed data for modeling and deployment  

---

## 🔬 Project Methodology & Workflow

### 1️⃣ Data Understanding
- Studied class distribution and imbalance  
- Inspected nutrition ranges per food  
- Identified noisy and duplicate entries  

📌 **Notebook:**  
`01_data_understanding.ipynb`

---

### 2️⃣ Data Cleaning & Preprocessing

Handled:
- Missing values (imputation / removal)  
- Duplicate food records  
- Outliers using statistical thresholds  
- Feature scaling using `StandardScaler`  

Clean data stored separately to ensure **reproducibility**.

📌 **Notebook:**  
`02_data_cleaning.ipynb`

---

### 3️⃣ Exploratory Data Analysis (EDA)

- Distribution plots for nutritional features  
- Inter-class variation analysis  
- Feature correlation analysis  

📌 **Notebook:**  
`03_eda.ipynb`

---

### 4️⃣ Feature Engineering

- Label encoding for food names  
- PCA to understand dimensional contribution  
- Feature importance analysis for interpretability  

📌 **Notebook:**  
`04_feature_engineering.ipynb`

---

### 5️⃣ Unsupervised Learning (Analysis Support)

Although the final output is supervised classification, unsupervised learning was used to:
- Understand natural food groupings  
- Validate nutritional similarity patterns  
- Support business explanation during evaluation  

📌 **Techniques:**
- K-Means Clustering  
- Distance-based similarity analysis  

📌 **Notebook:**  
`05_unsupervised_learning.ipynb`

---

### 6️⃣ Supervised Learning & Model Training

Trained and compared multiple classifiers:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Machine  
- Gradient Boosting  
- XGBoost  

Used **cross-validation** and **GridSearchCV** for tuning.

📌 **Notebook:**  
`06_supervised_learning.ipynb`

---

## ⚙️ ML Pipelines & Engineering Design

To ensure **production readiness**, pipelines were created:

### 🔹 Preprocessing Pipeline
- Scaling  
- Encoding  
- Feature transformation  

📄 `pipelines/preprocessing_pipeline.py`

---

### 🔹 Model Pipelines
- Unified preprocessing + model flow  
- Ensures same logic for training & inference  

📄 `pipelines/model_pipelines.py`

---

### 🔹 Hyperparameter Optimization
- GridSearchCV used  
- Prevents overfitting  

📄 `pipelines/grid_search.py`

---


## 📈 Evaluation Metrics

Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics help analyze:
- Overall correctness  
- Class-wise misclassification  
- Model stability  

---

## 🖥️ Streamlit Application Design

The project includes a **multi-page Streamlit application** designed for clear separation of functionality and usability.

### 🔹 Pages Overview

| Page | Purpose |
|----|--------|
| Food Classifier | Predict exact food name |
| Diet Recommendation | Nutrition-based guidance |
| Pipeline Overview | Explain ML workflow |
| Raw Data Explorer | Inspect original dataset |


---

## ▶️ How to Run the Project

Follow these steps to run the project locally:

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd NutriClass
```

2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

The application will open in your browser at:
```bash
http://localhost:8501

```

> ⚠️ **Note:** GitHub Actions CI may fail due to Streamlit UI execution in a headless environment.  
> The application runs successfully in a local setup.


🗂️ Project Structure
```bash
NutriClass/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── synthetic_food_dataset_imbalanced.csv
│   └── processed/
│       └── clean_food_data.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_supervised_learning.ipynb
│
├── pipelines/
│   ├── preprocessing_pipeline.py
│   ├── model_pipelines.py
│   └── grid_search.py
│
├── models/
│   └── nutriclass_pipeline.pkl
│
└── pages/
    ├── 1_Food_Classifier.py
    ├── 2_Diet_Recommendation.py
    ├── 3_Pipeline_Overview.py
    └── 4_Raw_Data_Explorer.py
```

👤 Project Presentation & Author
```bash
Project Developed By:
Bhuvaneswari G
Web Developer & Data Science Learner
```
🔚 Conclusion

NutriClass showcases how a structured machine learning approach can be applied to the Food & Nutrition domain, delivering accurate classification, clear insights, and real-world applicability through a deployable application.
 

