# Generated from: Mnami_RestaurantMarketAnalysis.ipynb
# Converted at: 2026-10-12T11:48:03.071Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🍽️ Restaurant Market Analysis & Rating Prediction
# **Dataset:** Zomato Restaurant Dataset  
# **Goal:** Analyse the restaurant market (ratings, prices, cuisines, locations) and predict rating categories (Low / Medium / High) from restaurant features.
# 
# ---
# 


# ## 1. Setup & Data Loading


import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

BASE_DIR   = os.path.dirname(os.path.abspath("__file__"))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Zomato Restaurant Dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
print(f"Shape: {df.shape}")
df.head(3)


# ## 2. Dataset Overview


print("=== Data Types ===")
print(df.dtypes)
print(f"\n=== Missing Values ===")
print(df.isnull().sum())
print(f"\nDuplicates: {df.duplicated().sum()}")
print(f"\n=== Descriptive Statistics ===")
df.describe()


# ## 3. Data Cleaning


# Remove duplicates
df = df.drop_duplicates()

# Remove unrated restaurants (rating = 0)
df = df[df["Aggregate rating"] > 0].copy()

# Remove rows with missing Cuisines
df = df[df["Cuisines"].notna()].copy()

# Strip whitespace from string columns
for col in ["City", "Cuisines", "Currency", "Has Table booking",
            "Has Online delivery", "Rating text"]:
    df[col] = df[col].astype(str).str.strip()

# Numeric casts
df["Average Cost for two"] = pd.to_numeric(df["Average Cost for two"], errors="coerce")
df["Votes"]                = pd.to_numeric(df["Votes"], errors="coerce")
df["Aggregate rating"]     = pd.to_numeric(df["Aggregate rating"], errors="coerce")
df["Price range"]          = pd.to_numeric(df["Price range"], errors="coerce")
df = df.dropna(subset=["Average Cost for two", "Votes", "Aggregate rating", "Price range"])

# Remove cost outliers (>99th percentile)
q99 = df["Average Cost for two"].quantile(0.99)
df  = df[df["Average Cost for two"] <= q99].copy()

print(f"Shape after cleaning: {df.shape}")
df.describe()


# ## 4. Exploratory Data Analysis


# ### 4.1 Rating Distribution


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Aggregate rating"].hist(bins=30, color="#3b82d4", edgecolor="white", ax=axes[0])
axes[0].set_title("Rating Distribution Histogram")
axes[0].set_xlabel("Aggregate Rating"); axes[0].set_ylabel("Count")

df["Rating text"].value_counts().plot(kind="bar", color="#3b82d4",
                                       edgecolor="white", ax=axes[1])
axes[1].set_title("Rating Text Counts")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ### 4.2 Price Range vs Rating


pr = df.groupby("Price range")["Aggregate rating"].mean().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
pr.plot(kind="bar", color=["#3b82d4","#7c5cd8","#27ae60","#f39c12"],
        edgecolor="white", ax=ax)
ax.set_title("Average Rating by Price Range")
ax.set_xlabel("Price Range (1=Budget → 4=Luxury)")
ax.set_ylabel("Avg Rating"); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout(); plt.show()
print(pr.to_string())


# ### 4.3 Top 15 Cuisines


df["Primary Cuisine"] = df["Cuisines"].apply(lambda x: x.split(",")[0].strip())
top_cuisines = df["Primary Cuisine"].value_counts().head(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top_cuisines.sort_values().plot(kind="barh", color="#3b82d4",
                                 edgecolor="white", ax=axes[0])
axes[0].set_title("Top 15 Cuisines by Count")
axes[0].set_xlabel("Count")

cuisine_rating = (df.groupby("Primary Cuisine")["Aggregate rating"]
                    .agg(["mean","count"]).query("count >= 20")
                    .sort_values("mean", ascending=False)
                    .head(15)["mean"].sort_values())
cuisine_rating.plot(kind="barh", color="#27ae60", edgecolor="white", ax=axes[1])
axes[1].set_title("Top 15 Cuisines by Avg Rating (min 20)")
axes[1].set_xlabel("Avg Rating")

plt.tight_layout(); plt.show()


# ### 4.4 City Analysis


city_stats = (df.groupby("City")["Aggregate rating"]
                .agg(["mean","count"])
                .query("count >= 10")
                .sort_values("mean", ascending=False)
                .head(15))

fig, ax = plt.subplots(figsize=(10, 5))
city_stats["mean"].sort_values().plot(kind="barh", color="#7c5cd8",
                                       edgecolor="white", ax=ax)
ax.set_title("Top 15 Cities by Average Rating (min 10 restaurants)")
ax.set_xlabel("Avg Rating")
plt.tight_layout(); plt.show()
city_stats.head(10)


# ### 4.5 Correlation Heatmap


numeric_cols = ["Average Cost for two","Price range","Aggregate rating",
                "Votes","Has Table booking","Has Online delivery"]
# Convert bool-like columns
df["Has Table booking"]   = (df["Has Table booking"]   == "Yes").astype(int)
df["Has Online delivery"] = (df["Has Online delivery"] == "Yes").astype(int)

fig, ax = plt.subplots(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap")
plt.tight_layout(); plt.show()
corr


# ## 5. Feature Engineering


# Target: Rating Category
def rating_category(r):
    if r < 3.5:   return "Low"
    elif r < 4.0: return "Medium"
    else:         return "High"
df["Rating Category"] = df["Aggregate rating"].apply(rating_category)

# Cuisine popularity (count of restaurants per primary cuisine)
cuisine_counts = df["Primary Cuisine"].value_counts()
df["Cuisine Popularity"] = df["Primary Cuisine"].map(cuisine_counts)

# City density
city_counts = df["City"].value_counts()
df["City Density"] = df["City"].map(city_counts)

# Log votes
df["Log Votes"] = np.log1p(df["Votes"])

# Label-encode cuisines and cities
le_cuisine = LabelEncoder()
le_city    = LabelEncoder()
df["Cuisine Encoded"] = le_cuisine.fit_transform(df["Primary Cuisine"])
df["City Encoded"]    = le_city.fit_transform(df["City"])

print("Rating category distribution:")
print(df["Rating Category"].value_counts())
print(f"\nFeature shapes: {df.shape}")

# Category distribution pie
fig, ax = plt.subplots(figsize=(5, 4))
palette = {"Low":"#e74c3c","Medium":"#f39c12","High":"#27ae60"}
vals = df["Rating Category"].value_counts().reindex(["Low","Medium","High"])
ax.pie(vals, labels=vals.index, autopct="%1.1f%%",
       colors=[palette[c] for c in vals.index], startangle=140)
ax.set_title("Rating Category Distribution")
plt.show()


# ## 6. Model Training & Evaluation


FEATURES = ["Price range","Log Votes","Cuisine Popularity","City Density",
            "Has Table booking","Has Online delivery",
            "Cuisine Encoded","City Encoded"]
TARGET   = "Rating Category"
CLASSES  = ["Low","Medium","High"]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ### 6.1 Random Forest


rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))


# ### 6.2 Gradient Boosting


gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                  max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_gb, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb, zero_division=0))


# ### 6.3 Model Comparison


rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
gb_f1 = f1_score(y_test, y_pred_gb, average='weighted')
best_model = rf if rf_f1 >= gb_f1 else gb
best_name  = "Random Forest" if rf_f1 >= gb_f1 else "Gradient Boosting"
y_pred_best = y_pred_rf if rf_f1 >= gb_f1 else y_pred_gb

metrics = ["Accuracy","Precision","Recall","F1"]
rf_vals = [accuracy_score(y_test,y_pred_rf),
           precision_score(y_test,y_pred_rf,average='weighted',zero_division=0),
           recall_score(y_test,y_pred_rf,average='weighted',zero_division=0),
           f1_score(y_test,y_pred_rf,average='weighted',zero_division=0)]
gb_vals = [accuracy_score(y_test,y_pred_gb),
           precision_score(y_test,y_pred_gb,average='weighted',zero_division=0),
           recall_score(y_test,y_pred_gb,average='weighted',zero_division=0),
           f1_score(y_test,y_pred_gb,average='weighted',zero_division=0)]

x = np.arange(len(metrics)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x-w/2, rf_vals, w, label="Random Forest",      color="#3b82d4", edgecolor="white")
ax.bar(x+w/2, gb_vals, w, label="Gradient Boosting",  color="#7c5cd8", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0,1.05); ax.set_title("Model Comparison"); ax.set_ylabel("Score"); ax.legend()
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.3f}",
                (bar.get_x()+bar.get_width()/2, bar.get_height()+0.005),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.show()
print(f"Best Model: {best_name}")


# ### 6.4 Confusion Matrix


cm = confusion_matrix(y_test, y_pred_best, labels=CLASSES)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_title(f"Confusion Matrix – {best_name}")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.show()


# ### 6.5 Feature Importance


fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8,4))
fi.plot(kind="barh", color="#3b82d4", edgecolor="white", ax=ax)
ax.set_title(f"Feature Importances – {best_name}")
ax.set_xlabel("Importance")
plt.tight_layout(); plt.show()
print(fi.sort_values(ascending=False).to_string())


# ## 7. Save Model & Metadata


joblib.dump(best_model,  os.path.join(MODEL_DIR, "restaurant_model.pkl"))
joblib.dump(le_cuisine,  os.path.join(MODEL_DIR, "le_cuisine.pkl"))
joblib.dump(le_city,     os.path.join(MODEL_DIR, "le_city.pkl"))
df.to_csv(os.path.join(MODEL_DIR, "cleaned_data.csv"), index=False)
print(f"Model saved: {MODEL_DIR}/restaurant_model.pkl")


# ## 8. Sample Prediction


# Example: North Indian restaurant, mid-range price, New Delhi, 500 votes
sample = {
    "price_range"        : 3,
    "cuisine"            : "North Indian",
    "city"               : "New Delhi",
    "votes"              : 500,
    "has_table_booking"  : 1,
    "has_online_delivery": 1
}

log_votes     = np.log1p(sample["votes"])
cuisine_pop   = int(cuisine_counts.get(sample["cuisine"], cuisine_counts.median()))
city_density  = int(city_counts.get(sample["city"], city_counts.median()))
cuisine_enc   = (int(le_cuisine.transform([sample["cuisine"]])[0])
                 if sample["cuisine"] in le_cuisine.classes_ else 0)
city_enc      = (int(le_city.transform([sample["city"]])[0])
                 if sample["city"] in le_city.classes_ else 0)

feat_vec = np.array([[
    sample["price_range"], log_votes, cuisine_pop, city_density,
    sample["has_table_booking"], sample["has_online_delivery"],
    cuisine_enc, city_enc
]])

pred   = best_model.predict(feat_vec)[0]
probs  = best_model.predict_proba(feat_vec)[0]
print(f"Input   : {sample}")
print(f"Predicted rating category: {pred}")
print(f"Probabilities:")
for cls, p in zip(best_model.classes_, probs):
    print(f"  {cls}: {p:.4f}")


# ## 9. Key Findings
# 
# 1. **Model accuracy: ~73.9%** using Gradient Boosting (best) / Random Forest
# 2. **Votes (log-transformed)** and **Price Range** are the strongest predictors
# 3. **~18% of restaurants** achieve a High rating (≥4.0)
# 4. **Luxury restaurants** (Price Range 4) achieve the highest average rating
# 5. **North Indian** cuisine dominates in volume; specialty cuisines achieve higher ratings
# 6. **Online delivery** shows a marginal positive correlation with ratings in certain markets
# 
# ---
# *Generated with Python | Zomato Restaurant Dataset*
#