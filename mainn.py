# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import os
import pandas as pd
import numpy as np
import re
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

print("READY: Libraries loaded.")


# =====================================================
# 2. LOAD DATASET
# =====================================================
DATA_PATH = "/mnt/data/Twitter_Emotion_Dataset1.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"❌ File tidak ditemukan: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("Dataset berhasil dimuat!")

# =====================================================
# 3. DETECT TEXT & LABEL COLUMN OTOMATIS
# =====================================================
def detect_cols(df):
    text_keywords = ["text", "tweet", "content", "message"]
    label_keywords = ["label", "sentiment", "emotion", "target", "class"]

    text_col = None
    label_col = None

    for col in df.columns:
        c = col.lower()
        if any(key in c for key in text_keywords):
            text_col = col
        if any(key in c for key in label_keywords):
            label_col = col

    if text_col is None:
        raise ValueError("❌ Kolom teks tidak ditemukan! Cek nama kolom dataset.")
    if label_col is None:
        raise ValueError("❌ Kolom label tidak ditemukan! Cek nama kolom dataset.")

    return text_col, label_col


text_col, label_col = detect_cols(df)
print("TEXT COLUMN  =", text_col)
print("LABEL COLUMN =", label_col)


# =====================================================
# 4. TEXT CLEANING & PREPROCESSING
# =====================================================
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_series(series):
    result = []
    for text in series:
        t = clean_text(text)
        t = " ".join([lemm.lemmatize(w) for w in t.split() if w not in stop_words])
        result.append(t)
    return result


print("Cleaning text...")
df["clean_text"] = preprocess_series(df[text_col])


# =====================================================
# 5. LABEL ENCODING
# =====================================================
labels = sorted(df[label_col].unique())
label2idx = {lab: i for i, lab in enumerate(labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

df["label_idx"] = df[label_col].map(label2idx)

print("Label mapping:", label2idx)


# =====================================================
# 6. TF-IDF VECTORIZE
# =====================================================
X = df["clean_text"].values
y = df["label_idx"].values

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_vect = tfidf.fit_transform(X)

print("Vectorized shape:", X_vect.shape)


# =====================================================
# 7. TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# =====================================================
# 8. SMOTE OVERSAMPLING (IMBALANCE HANDLING)
# =====================================================
try:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("SMOTE applied:", X_train_res.shape)
except:
    print("SMOTE failed → Using original training data.")
    X_train_res, y_train_res = X_train, y_train


# =====================================================
# 9. TRAIN 2 MODELS
# =====================================================
model_a = LogisticRegression(max_iter=1000)
model_b = RandomForestClassifier(n_estimators=200)

print("Training Model A (Logistic Regression)...")
model_a.fit(X_train_res, y_train_res)

print("Training Model B (Random Forest)...")
model_b.fit(X_train_res, y_train_res)


# =====================================================
# 10. ENSEMBLE VOTING CLASSIFIER
# =====================================================
ensemble = VotingClassifier(
    estimators=[("lr", model_a), ("rf", model_b)],
    voting="soft"
)

print("Training Ensemble...")
ensemble.fit(X_train_res, y_train_res)


# =====================================================
# 11. EVALUATION
# =====================================================
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy = {acc:.4f}")
    return acc

print("\n=== MODEL ACCURACIES ===")
acc_a = evaluate(model_a, X_test, y_test, "Model A (LR)")
acc_b = evaluate(model_b, X_test, y_test, "Model B (RF)")
acc_ens = evaluate(ensemble, X_test, y_test, "Ensemble Voting")

print("\nClassification Report (Ensemble):")
print(classification_report(y_test, ensemble.predict(X_test), target_names=labels))


# =====================================================
# 12. SAVE MODELS
# =====================================================
os.makedirs("models", exist_ok=True)

joblib.dump(tfidf, "models/vectorizer.joblib")
joblib.dump(model_a, "models/model_a.joblib")
joblib.dump(model_b, "models/model_b.joblib")
joblib.dump(ensemble, "models/ensemble.joblib")

with open("models/label_map.json", "w") as f:
    json.dump({"label2idx": label2idx, "idx2label": idx2label}, f)

print("\nModels saved inside /models folder.")


