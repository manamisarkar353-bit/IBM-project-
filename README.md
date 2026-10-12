# 🍽️ Restaurant Market Analysis & Rating Prediction

A complete end-to-end data science project that analyses the Zomato restaurant market, identifies top cuisines and price trends, and predicts restaurant rating categories (Low / Medium / High) using machine learning.

---

## 📁 Dataset

- **Source:** [Zomato Restaurant Dataset](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data)
- **File:** `data/Zomato Restaurant Dataset.csv`
- **Size:** ~9,551 rows × 21 columns
- **Key columns:** Restaurant Name, City, Cuisines, Average Cost for Two, Price Range, Aggregate Rating, Votes, Has Table Booking, Has Online Delivery

---

## 🛠️ Technologies Used

| Layer | Library / Tool |
|-------|---------------|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (Random Forest, Gradient Boosting) |
| Visualisation | matplotlib, seaborn |
| Backend API | Flask |
| Frontend dashboard | Streamlit |
| Report generation | python-docx |
| Notebook | nbformat, Jupyter |
| Model persistence | joblib |

---

## 📂 Project Structure

```
Mnami_IBM/
├── data/
│   └── Zomato Restaurant Dataset.csv
├── backend/
│   └── app.py                          # Flask REST API
├── frontend/
│   └── ui.py                           # Streamlit dashboard
├── model/
│   ├── restaurant_model.pkl            # Saved best model
│   ├── model_meta.json                 # Metrics & metadata
│   ├── le_cuisine.pkl                  # Cuisine label encoder
│   ├── le_city.pkl                     # City label encoder
│   └── cleaned_data.csv               # Cleaned dataset
├── report_images/                      # EDA & model charts
├── train_model.py                      # Training script
├── requirements.txt
├── README.md
├── Mnami_RestaurantMarketAnalysis.ipynb
└── Mnami_ProjectReport.docx
```

---

## ⚙️ Setup & Run Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Generate EDA charts in `report_images/`
- Train Random Forest & Gradient Boosting classifiers
- Save the best model to `model/restaurant_model.pkl`
- Save metrics to `model/model_meta.json`

### 3. Start the Flask backend

```bash
python backend/app.py
```

API will be live at `http://127.0.0.1:5000`

**Available endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and endpoint list |
| `/health` | GET | Health check |
| `/predict` | POST | Predict rating category |
| `/model_info` | GET | Model metrics and feature importances |
| `/dataset_summary` | GET | Dataset KPIs |
| `/top_cuisines` | GET | Top cuisines with ratings |

**Prediction example:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"price_range":3,"cuisine":"North Indian","city":"New Delhi","votes":500,"has_table_booking":1,"has_online_delivery":1}'
```

### 4. Start the Streamlit frontend

```bash
streamlit run frontend/ui.py
```

Dashboard opens at `http://localhost:8501`

---

## 📊 Key Findings

1. **73.9% classification accuracy** – Gradient Boosting outperforms Random Forest marginally (F1: 0.7371 vs 0.7358).
2. **Votes (log-transformed)** and **Price Range** are the strongest predictors of rating category.
3. **Only ~18% of restaurants** achieve a High rating (≥4.0), making high-rated restaurants a clear differentiator.
4. **Luxury price range (4)** achieves the highest average rating – customers associate premium pricing with quality.
5. **North Indian and Chinese** cuisines dominate in volume, but niche cuisines (e.g., Japanese) achieve higher average ratings.
6. **Online delivery** shows a marginal but positive correlation with ratings in some markets.

---

## 📝 Report

See `Mnami_ProjectReport.docx` for the full project report including EDA charts, model evaluation, API documentation, and business insights.

---

## 📓 Notebook

`Mnami_RestaurantMarketAnalysis.ipynb` contains the full walkthrough from data loading to model evaluation with inline outputs.
