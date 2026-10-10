from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib  import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bank_churn_gb_enn.pkl"

model = joblib.load(MODEL_PATH)

@app.route('/')

def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    customer_data = {
        "CreditScore": int(request.form["CreditScore"]),
        "Geography": request.form["Geography"],
        "Gender": request.form["Gender"],
        "Age": int(request.form["Age"]),
        "Tenure": int(request.form["Tenure"]),
        "Balance": float(request.form["Balance"]),
        "NumOfProducts": int(request.form["NumOfProducts"]),
        "HasCrCard": int(request.form["HasCrCard"]),
        "IsActiveMember": int(request.form["IsActiveMember"]),
        "EstimatedSalary": float(request.form["EstimatedSalary"]),
        "Satisfaction Score": int(
            request.form["Satisfaction Score"]
        ),
        "Card Type": request.form["Card Type"],
        "Point Earned": int(request.form["Point Earned"])
    }

    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]
    probabilities = model.predict_proba(customer_df)[0]
    churn_probability = probabilities[
        list(model.classes_).index(1)
    ]

    if prediction == 1:
        result = "The model predicts that this customer may churn."
    else:
        result = "The model predicts that this customer may stay."

    percentage = round(churn_probability * 100, 2)

    return render_template(
        "result.html",
        result=result,
        percentage=percentage
    )

if __name__ == "__main__":
    app.run(debug=True)