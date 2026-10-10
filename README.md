# Bank Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to leave the bank. The project includes a trained model and a Flask web interface where users can enter customer details and receive a prediction with an estimated churn probability.

## Project Features

- Web form for entering customer profile, financial, and engagement details.
- Churn prediction using a trained Gradient Boosting model.
- Imbalance handling with Random Over-Sampling and Edited Nearest Neighbours (ENN).
- Result page showing the predicted outcome and estimated churn probability.

## Model Development

The dataset contains 10,000 customer records. The target column is `Exited` (`1` = churned, `0` = stayed). The target is imbalanced, with approximately 20.38% churned customers and 79.62% customers who stayed.

The final workflow uses a Gradient Boosting model with Random Over-Sampling and ENN. Customer identifiers (`RowNumber`, `CustomerId`, `Surname`) and the `Complain` column are excluded; `Complain` closely tracks the target and can cause data leakage.

Approximate test-set results for the selected model:

| Metric | Score |
|---|---:|
| Accuracy | 83% |
| Churn precision | 57% |
| Churn recall | 72% |
| Churn F1-score | 64% |
| ROC-AUC | 0.876 |

The model prioritizes finding more customers who may churn. Its recall is higher than the original Gradient Boosting model, though its overall accuracy and precision are lower. These results are estimates from this dataset and do not guarantee real-world performance.

## Input Features

The app collects these 13 features:

- Credit score, geography, gender, age, and tenure
- Account balance and estimated salary
- Number of products, credit-card ownership, and active-member status
- Satisfaction score, card type, and points earned

## Tech Stack

- **Language:** Python
- **Data and ML:** pandas, scikit-learn, imbalanced-learn
- **Model persistence:** joblib
- **Web app:** Flask, HTML, CSS

## Project Structure

```text
customer_churn/
├── app.py
├── models/
│   └── bank_churn_gb_enn.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
└── README.md
```

## Run Locally

1. Open a terminal in the project directory.
2. Install the required packages:

   ```bash
   python -m pip install flask pandas joblib scikit-learn imbalanced-learn
   ```

3. Confirm that the trained model exists at `models/bank_churn_gb_enn.pkl`.
4. Start the Flask app:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000/` in your browser.

## How It Works

1. The user enters customer details in the web form.
2. Flask converts the submitted values into a one-row pandas DataFrame.
3. The saved preprocessing-and-model pipeline generates a churn prediction and churn probability.
4. The result page displays the outcome and estimated probability.

## Limitations

- Predictions depend on the dataset and training process used.
- Churn probability is a model estimate, not a certainty.
- The Flask development server is intended for local development, not production deployment.

## Future Improvements

- Add stronger server-side input validation and error handling.
- Evaluate the model on newer or external data.
- Deploy the app using a production-ready hosting setup.
