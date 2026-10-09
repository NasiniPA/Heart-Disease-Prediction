# 🫀 CardioPredict ML — Heart Disease Prediction System

A complete machine learning laboratory mini-project and clinical decision support system for predicting Coronary Artery Disease (CAD) using the **UCI Cleveland Heart Disease dataset (N=303)**.

---

## 🚀 Quickstart & Execution

### 1. Python Streamlit Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run clinical application
streamlit run app.py
```

### 2. Machine Learning Training Pipeline
```bash
# Train Logistic Regression, Random Forest & SVM, evaluate 5-fold CV & export figures
python train_model.py
```

### 3. Interactive Web Application (React + Vite)
```bash
npm install
npm run dev
```
Navigate to `http://localhost:3000`.

---

## 📊 Dataset & Course Outcome (CO5) Mapping

- **Dataset:** UCI Cleveland Heart Disease (303 patient records, 13 clinical biomarkers + 1 target class).
- **Target:** `0` = No Coronary Artery Disease (<50% luminal stenosis), `1` = Coronary Artery Disease present (≥50% diameter narrowing).
- **Course Outcome:** *"CO5: Solve real-life problems using appropriate machine learning models and evaluate performance measures."*
- **Algorithms Compared:**
  - **Logistic Regression (Standardized Pipeline):** Test Accuracy **85.25%**, Recall **88.24%**, Precision **83.33%**, ROC-AUC **0.912**.
  - **Random Forest (Ensemble):** Test Accuracy **83.61%**, Recall **85.29%**, ROC-AUC **0.895**.
  - **Support Vector Machine (RBF Kernel):** Test Accuracy **81.97%**, Recall **85.29%**, ROC-AUC **0.884**.
- **Clinical Prioritization:** **Recall (Sensitivity)** is prioritized above pure accuracy to minimize False Negatives (missing an acute coronary obstruction is potentially fatal).

---

## 📁 Repository Structure
```text
├── .vscode/
│   ├── launch.json         # F5 Debugger configurations (Streamlit, Python, Vite)
│   ├── tasks.json          # 1-click VS Code task runner
│   └── settings.json       # Workspace python settings
├── app.py                  # Streamlit clinical diagnosis web app (with auto-launcher)
├── train_model.py          # Machine learning pipeline training & 5-fold CV
├── dataset.csv             # Full UCI Cleveland dataset (303 records)
├── requirements.txt        # Python package dependencies
├── run.bat                 # 1-click launcher for Windows
├── run.sh                  # 1-click launcher for macOS/Linux
├── package.json            # Web app dependencies (React, Vite, Tailwind CSS)
└── src/
    ├── App.tsx             # Complete interactive clinical diagnosis & VS Code helper web app
    └── main.tsx            # Entry point
```
