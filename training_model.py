import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("data/student_records.csv")

# Feature matrix and target
X = df[["attendance", "assignment_score", "exam_score"]]
y = df["risk_level"]

# Encode target labels (Low=0, Medium=1, High=2)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save encoder for decoding predictions later
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/label_encoder.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "models/rf_model.pkl")

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "models/xgb_model.pkl")

# Evaluate
y_pred = rf_model.predict(X_test)
print("Random Forest Performance:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Performance:\n", classification_report(y_test, y_pred_xgb, target_names=encoder.classes_))
