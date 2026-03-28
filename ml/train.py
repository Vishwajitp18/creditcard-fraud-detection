import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle imbalance (fraud is rare)
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# Model (XGBoost)
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("../model/model.pkl", "wb"))

print("✅ Model trained and saved!")

# -------------------------------
# 📊 Feature Importance
# -------------------------------
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n🔝 Top 10 Important Features:")
print(importance_df.head(10))

# Plot (top 10)
plt.figure()
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xlabel("Importance")
plt.title("Top 10 Features (XGBoost)")
plt.gca().invert_yaxis()
plt.show()