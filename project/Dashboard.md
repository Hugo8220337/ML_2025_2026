Here is the comprehensive implementation plan for your **Machine Learning Dashboard**. This plan transforms your existing training scripts into a deployable "PowerBI-style" web application using **Streamlit**, satisfying the "Interface" and "Unsupervised Learning" requirements of your assessment1.

# ---

**📅 Implementation Plan: Fake News Detection Dashboard**

## **Phase 1: Dependencies & Environment**

**Goal:** Add necessary libraries for the web interface and news scraping.

1. Update project/requirements.txt  
   Add the following libraries to support the GUI and article extraction:  
   Plaintext  
   streamlit==1.40.0  
   newspaper3k==0.2.8      \# For extracting body text from news URLs  
   feedparser==6.0.10      \# For parsing Google News RSS feeds  
   joblib==1.3.2           \# For saving/loading trained models  
   scikit-learn==1.7.2     \# Ensure this matches your training env

## ---

**Phase 2: Backend & Model Persistence**

**Goal:** Modify your existing scripts to **save** trained models so the website can load them instantly without retraining.

### **2.1. Update topic\_classification.py**

* **Current State:** Runs EMS and prints JSON.  
* **Action:**  
  1. Accept arguments (C, max\_iter) to allow tuning from the UI.  
  2. Save the best model and vectorizer to project/files/models/.

Python  
\# Add to end of function  
import joblib  
import os  
os.makedirs('project/files/models', exist\_ok=True)  
joblib.dump(vectorizer, 'project/files/models/topic\_vectorizer.pkl')  
joblib.dump(best\_model, 'project/files/models/topic\_model.pkl')  
return accuracy\_score  \# Return metric to display in UI

### **2.2. Update setence\_detection.py & semantic\_consistency.py**

* **Action:** Similar to above, ensure vectorizer and model are saved as .pkl files after training.

### **2.3. Create project/inference.py**

* **Purpose:** A bridge script that loads models **once** and runs predictions.  
* **Key Logic:**  
  * Load all .pkl files in \_\_init\_\_.  
  * Define predict\_pipeline(headline, body) that runs:  
    1. Topic Model $\\rightarrow$ Returns Topic.  
    2. Stance Model $\\rightarrow$ Returns Agreement/Clickbait score.  
    3. Consistency Model $\\rightarrow$ Returns Consistency score.  
    4. **Aggregator:** Returns a final "Fake News Probability" (weighted average of above).

## ---

**Phase 3: Frontend Implementation (Streamlit)**

**Goal:** Build the 3-page structure.

### **3.1. Directory Structure**

Create the following folder structure inside project/:

Plaintext

project/  
├── app.py                   \# Landing Page  
├── inference.py             \# Prediction Logic (created in Phase 2\)  
└── pages/  
    ├── 1\_🔮\_Live\_Prediction.py  
    ├── 2\_⚙️\_Model\_Tuning.py  
    └── 3\_📈\_Performance\_&\_Data.py

### **3.2. Page 1: Live Prediction (1\_🔮\_Live\_Prediction.py)**

**Features:**

* **Input:** Text box for "Google News / Article URL".  
* **Logic:**  
  1. Use newspaper3k to download the article body from the URL.  
  2. Pass Headline \+ Body to inference.predict\_pipeline.  
* **Visuals:**  
  * 3 Key Metric Cards: "Topic", "Fake Probability", "Consistency Score".  
  * Expandable section "Show Extracted Text" (for debugging).

### **3.3. Page 2: Model Tuning (2\_⚙️\_Model\_Tuning.py)**

**Features:**

* **Sidebar Controls:** Sliders for Hyperparameters (e.g., C for Logistic Regression, n\_estimators for Random Forest).  
* **Action:** "Retrain Topic Model" button.  
* **Logic:**  
  * Calls topic\_classification(C=slider\_val, retrain=True).  
  * Updates the saved .pkl file.  
  * Shows a "Success: Accuracy changed from 85% $\\rightarrow$ 87%" message.

### **3.4. Page 3: Performance & Unsupervised (3\_📈\_Performance\_&\_Data.py)**

**Features:**

* **Supervised Metrics:** Load setence\_detection\_results.csv and display as a table/chart.  
* **Unsupervised Learning (Mandatory for Grade):**  
  * Run K-Means clustering on the news dataset.  
  * Use PCA/t-SNE to reduce to 2D.  
  * **Plot:** st.scatter\_chart showing how articles group together (e.g., "Politics" cluster vs. "Tech" cluster).  
  * *Justification:* "We used K-Means to validate if natural clusters align with our supervised labels."

## ---

**Phase 4: Docker Integration**

**Goal:** Ensure the app runs in the container.

1. Update project/dockerfile:  
   Change the command to launch Streamlit instead of just running main.py.  
   Dockerfile  
   \# ... previous install steps ...

   \# Create directory for models  
   RUN mkdir \-p project/files/models

   \# Expose Streamlit Port  
   EXPOSE 8501

   \# Run Streamlit  
   CMD \["streamlit", "run", "project/app.py", "--server.address=0.0.0.0"\]

## ---

**Phase 5: Execution Checklist**

1. \[ \] **Local Training:** Run python main.py locally *once* to generate the initial .pkl model files (so the app doesn't crash on first load).  
2. \[ \] **Docker Build:** docker-compose up \--build.  
3. \[ \] **Verify:** Open http://localhost:8501.  
4. \[ \] **Report:** Take screenshots of Page 3 (Clusters) and Page 2 (Tuning) for your PDF report2.  
