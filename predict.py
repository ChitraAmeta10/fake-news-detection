import joblib

print("Loading model...\n")

try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    print("❌ ERROR: Model files not found or corrupted.")
    print("Run 'python3 fake_news.py' first to generate the files.")
    exit()

print("Fake News Detection Model Loaded!")
print("Type any news sentence. Type 'exit' to stop.\n")

while True:
    text = input("Enter news text: ")

    if text.strip().lower() == "exit":
        print("Exiting...")
        break

    if len(text.strip()) == 0:
        print("Please type something.\n")
        continue

    x = vectorizer.transform([text])
    pred = model.predict(x)[0]   # this returns "fake" or "real"

    if pred == "fake":
        print("🔴 Fake News\n")
    else:
        print("🟢 Real News\n")
