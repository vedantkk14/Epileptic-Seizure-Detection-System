from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")
svm_model = joblib.load("model/svm_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(file, header=None)

                if df.shape != (1, 178):
                    prediction_text = f"Error: CSV must have exactly 1 row and 178 columns, got {df.shape}"
                else:
                    # Apply scaler → PCA → prediction
                    scaled = scaler.transform(df)
                    pca_features = pca.transform(scaled)
                    pred = svm_model.predict(pca_features)[0]
                    prob = svm_model.predict_proba(pca_features)[0][1]

                    if pred == 1:
                        prediction_text = f"⚠️ Likely seizure. Probability: {prob:.2%}"
                    else:
                        prediction_text = f"✅ Unlikely seizure. Probability: {prob:.2%}"
            except Exception as e:
                prediction_text = f"Error processing file: {e}"
        else:
            prediction_text = "Please upload a CSV file."

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
