📰 Fake News Detection (Machine Learning + Streamlit)
A simple and effective Fake News Detection System built using Python, Machine Learning (TF-IDF + Logistic Regression), and a clean Streamlit UI.
The model is trained on the Constraint 2021 COVID-19 Fake News Dataset and classifies news as:
🔴 Fake News
🟢 Real News
🚀 Live Demo
Try the app here:
👉 https://chitra231025-fake-news-detection.streamlit.app/
📌 Features
✔ Detects fake vs real news in real-time
✔ Simple & clean Streamlit UI
✔ Machine Learning pipeline (TF-IDF + Logistic Regression)
✔ Trained on COVID-19 misinformation dataset
✔ Completely deployable on Streamlit Cloud
✔ Works on any device via browser
📂 Project Structure
fake-news-detection/
│── app.py                   # Streamlit app
│── fake_news.py             # ML training script
│── predict.py               # CLI prediction script
│── fake_news_model.pkl      # Saved ML model
│── vectorizer.pkl           # TF-IDF vectorizer
│── requirements.txt
│── README.md
│
└── data/
    ├── Constraint_Train.csv
    ├── Constraint_Test.csv
    ├── Constraint_Val.csv
🧠 How It Works
The dataset is cleaned and prepared
Text is converted to numerical features using TF-IDF Vectorizer
Logistic Regression is trained on labeled COVID tweets
Model predicts "fake" or "real"
Streamlit UI displays the result instantly
⚠️ Note:
The dataset contains only COVID-related misinformation.
The model is optimized for COVID fake news, not general fake news.
🛠️ Installation
Clone the repository
git clone https://github.com/chitra231025/fake-news-detection.git
cd fake-news-detection
Install dependencies
pip install -r requirements.txt
▶️ Run the Streamlit App
streamlit run app.py
📊 Model Performance
Your ML model achieved:
Accuracy: ~92%
Performs best on:
False COVID treatments
Rumor-style tweets
WhatsApp-type misinformation
Government/health-related claims
📦 Requirements
These are included in requirements.txt:
streamlit
scikit-learn
pandas
numpy
joblib
🖥 Preview Screenshots
(Add your own screenshots here)
screenshots/
│── home.png
│── result_fake.png
│── result_real.png
🧑‍💻 Author
Chitra
⭐ Future Enhancements
Train on a general fake news dataset
Upgrade to BERT-based model
Add probability visualization
Deploy on HuggingFace Spaces
API endpoint for integration
