from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data/Prostate_Cancer.csv')
df.drop(columns=['id', 'diagnosis_result'], inplace=True)  # remove label/id
features = df.columns.tolist()


@app.route('/', methods=['GET'])
def index():
    data_preview = pd.read_csv('data/Prostate_Cancer.csv').head(10).to_html(classes='table table-striped')
    return render_template('index.html', features=features, data_preview=data_preview)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get selected model from frontend
        selected_model = request.form['model']  # e.g., 'logreg', 'knn', etc.
        model_path = f'weights/{selected_model}_model.pkl'

        # Load selected model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        with open('weights/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Get input values from form
        input_values = [float(request.form[f]) for f in features]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        result = 'Malignant' if prediction == 1 else 'Benign'

        # Additional evaluations (ROC + Confusion Matrix)
        data = pd.read_csv('data/Prostate_Cancer.csv')
        y_true = data['diagnosis_result'].map({'M': 1, 'B': 0})
        X_all = data.drop(columns=['id', 'diagnosis_result'])
        X_scaled = scaler.transform(X_all)
        y_score = model.predict_proba(X_scaled)[:, 1]
        y_pred = model.predict(X_scaled)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('static/roc_curve.png')
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('static/conf_matrix.png')
        plt.close()

        return render_template('result.html', result=result, model_name=selected_model.upper())

    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/prediction_score')
def prediction_score():
    # Ensure images are in the static folder
    confusion_matrix_file = 'confusion_matrices.png'
    comparison_chart_file = 'model_comparison_with_values.png'

    # Serve the images as static files
    return render_template('prediction_score.html',
                           confusion_matrix=confusion_matrix_file,
                           comparison_chart=comparison_chart_file)

if __name__ == '__main__':
    app.run(debug=True)
