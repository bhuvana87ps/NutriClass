# 🍽️ NutriClass: Food Classification Using Nutritional Data


## 📌 Project Overview

**NutriClass** is an end-to-end Machine Learning project that analyzes food nutrition data to:

- Perform robust feature engineering
- Discover hidden patterns using unsupervised learning
- Build, compare, and tune multiple supervised classification models
- Demonstrate real-world ML inference readiness
- Present results using Power BI and Streamlit dashboards

The project follows **industry-standard ML architecture**, separating **data processing, feature engineering, modeling, evaluation, and inference**.

---

## 🧠 Skills Takeaway

Through this project, the following technical skills were developed and demonstrated:

- Data preprocessing and cleaning
- Feature engineering for tabular nutrition data
- Handling numerical, categorical, and boolean features
- Dimensionality reduction using PCA
- Unsupervised learning (KMeans, DBSCAN)
- Supervised classification modeling
- Handling class imbalance using stratified sampling
- Model evaluation using accuracy, precision, recall, and F1-score
- Cross-validation and hyperparameter tuning (GridSearchCV)
- Machine learning inference
- Model performance visualization using Power BI
- Interactive ML inference using Streamlit

---

## 🎯 Objectives

- Transform raw nutrition data into model-compatible features  
- Explore inherent food groupings using clustering techniques  
- Predict food health classification using supervised ML  
- Compare multiple models using cross-validation and GridSearch  
- Build reusable preprocessing and modeling pipelines  

---

## 🗂️ Project Structure
```bash
NutriClass/
│
├── data/
│ ├── raw/
│ │ └── food_nutrition_raw.csv
│ └── processed/
│ ├── cleaned_food_data.csv
│ ├── X_features_inference_ready.csv
│ ├── X_features_pca.csv
│ └── labeled_food_data.csv
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_eda.ipynb
│ ├── 04_feature_engineering.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_supervised_learning.ipynb
│
├── pipelines/
│ ├── preprocessing_pipeline.py
│ ├── model_pipelines.py
│ └── grid_search.py
│
├── models/
│ ├── baseline_logistic.pkl
│ ├── best_random_forest.pkl
│ └── label_encoder.pkl
│
├── dashboards/
│ ├── power_bi/
│ │ └── model_results.pbix
│ └── streamlit/
│ ├── app.py
│ └── pages/
│ ├── inference.py
│ ├── model_comparison.py
│ └── pipeline_overview.py
│
├── README.md
└── requirements.txt

```
---

## 📘 Notebook Breakdown

---

### 1️⃣ Data Understanding  
**Notebook:** `01_data_understanding.ipynb`

- Loaded raw nutrition dataset
- Reviewed schema, data types, and feature meaning
- Verified dataset size and structure

**Output:** Clear understanding of numerical, categorical, and boolean variables.

---

### 2️⃣ Data Cleaning  
**Notebook:** `02_data_cleaning.ipynb`

- Handled missing values
- Removed duplicate records
- Standardized formats and values
- Ensured data consistency

**Output:** Cleaned dataset ready for analysis.

---

### 3️⃣ Exploratory Data Analysis (EDA)  
**Notebook:** `03_eda.ipynb`

- Univariate and multivariate analysis
- Outlier and skewness detection
- Distribution and variability analysis
- Feature behavior understanding

**Output:** Identified transformation and scaling requirements.

---

### 4️⃣ Feature Engineering  
**Notebook:** `04_feature_engineering.ipynb`

#### Key Transformations

- **Categorical Encoding**
  - One-Hot Encoding for `Meal_Type` and `Preparation_Method`
- **Boolean Encoding**
  - Converted `Is_Vegan`, `Is_Gluten_Free` to 0/1
- **Numerical Transformation**
  - Log transformation for skewed features (Calories, Sugar, Sodium)
  - StandardScaler for numerical standardization
- **Pipeline Design**
  - ColumnTransformer + Pipeline for reusable preprocessing

**Output:**  
An **inference-ready feature dataset** reusable across all ML models.

---

### 5️⃣ Unsupervised Learning  
**Notebook:** `05_unsupervised_learning.ipynb`

#### Objective
Discover natural groupings in food nutrition data **without labels**.

#### Techniques Used
- PCA for dimensionality reduction
- KMeans clustering
- DBSCAN clustering

#### Evaluation
- Elbow Method
- Silhouette Score
- PCA-based cluster visualization

**Output:**  
Nutrition-based food clusters and pattern insights.

---

### 6️⃣ Supervised Learning (Classification Modeling)  
**Notebook:** `06_supervised_learning.ipynb`

#### Label Creation
A rule-based health label was introduced:
- High Calories / Sugar / Sodium → `Unhealthy`
- Otherwise → `Healthy`

#### Models Trained
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting
- XGBoost (advanced)

#### Evaluation & Tuning
- 5-fold Cross-Validation
- Model comparison table
- GridSearchCV for hyperparameter tuning
- Classification report and confusion matrix

#### Inference
- Predictions demonstrated on unseen food data
- Same preprocessing + model pipeline applied

**Output:**  
A **complete supervised ML pipeline with inference capability**.

---

## 📊 Dashboards

### Power BI
- Model performance comparison
- Cross-validation results
- Confusion matrix visualization
- Feature importance analysis

### Streamlit
- User input for nutrition values
- Live ML inference
- Model selection and prediction output

---

## 🧠 Key Design Principles

- Feature engineering separated from modeling
- Pipelines used to avoid data leakage
- Unsupervised learning used for exploration
- Supervised learning used for prediction
- Cross-validation and GridSearch applied consistently

---

## 🏁 Conclusion

NutriClass demonstrates the **complete machine learning lifecycle**:

- Data Understanding
- Data Cleaning
- Feature Engineering
- Unsupervised Learning
- Supervised Classification
- Model Evaluation & Tuning
- Inference Demonstration
- Dashboard Visualization

## 🎓 GUVI Mini Project

This project was completed as a **GUVI Mini Project** and demonstrates an end-to-end machine learning pipeline with proper preprocessing, modeling, evaluation, and inference.

The project is **exam-ready, portfolio-ready.**.

---

## 🚀 Future Enhancements

- Model explainability using SHAP
- Deployment with FastAPI
- Automated retraining pipelines
- Feature store integration

---

## 👤 Author

**Bhuvana PS**
_Website Developer & Digital Advisor_
_Data Analytics & Machine Learning Practitioner_
