import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------------
# 1. Load Dataset
# ------------------------
train = pd.read_csv("data/Constraint_Train.csv")
test = pd.read_csv("data/Constraint_Test.csv")
val = pd.read_csv("data/Constraint_Val.csv")

# Combine train + val for better performance
data = pd.concat([train, val], axis=0)

data = data[['tweet', 'label']]
data['tweet'] = data['tweet'].astype(str)

# ------------------------
# 2. Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['tweet'], data['label'], test_size=0.2, random_state=42
)

# ------------------------
# 3. Vectorization
# ------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------
# 4. Train Model
# ------------------------
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# ------------------------
# 5. Evaluation
# ------------------------
pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# ------------------------
# 6. Save the model
# ------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved as fake_news_model.pkl")
print("Vectorizer saved as vectorizer.pkl")
