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
        # Logistic Regression Sigmoid Curve
        if selected_model == 'logreg':
            from scipy.special import expit
            x_vals = np.linspace(-10, 10, 100)
            y_vals = expit(x_vals)
            plt.figure()
            plt.plot(x_vals, y_vals, label='Sigmoid Function')
            plt.title('Logistic Regression Sigmoid Curve')
            plt.xlabel('Logit')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.savefig('static/logreg_sigmoid.png')
            plt.close()

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
        if selected_model in ['logreg', 'knn', 'svm']:  # Add 'svm' if model is trained similarly
            from matplotlib.colors import ListedColormap
            from sklearn.decomposition import PCA

            # Reduce dimensions for plotting
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            model.fit(X_pca, y_true)  # Fit on PCA-reduced data

            h = .02
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure()
            cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
            cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
            plt.contourf(xx, yy, Z, cmap=cmap_light)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap=cmap_bold, edgecolor='k', s=20)
            plt.title(f'{selected_model.upper()} Decision Boundary (PCA-reduced)')
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.tight_layout()
            plt.savefig(f'static/{selected_model}_decision_boundary.png')
            plt.close()


        if selected_model in ['rf', 'logreg']:  # 'rf' for random forest
            try:
                importances = model.feature_importances_ if selected_model == 'rf' else np.abs(model.coef_[0])
                sorted_idx = np.argsort(importances)[::-1]
                feature_names = X_all.columns[sorted_idx]
                importances_sorted = importances[sorted_idx]

                plt.figure(figsize=(8, 5))
                sns.barplot(x=importances_sorted, y=feature_names, palette='coolwarm')
                plt.title(f'{selected_model.upper()} Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(f'static/{selected_model}_feature_importance.png')
                plt.close()
            except:
                pass  # In case model doesn’t support feature_importances_
        # Decision Tree Structure
        if selected_model == 'dt':
            from sklearn import tree
            plt.figure(figsize=(20, 10))
            tree.plot_tree(model,
                           feature_names=X_all.columns,
                           class_names=['Benign', 'Malignant'],
                           filled=True, rounded=True)
            plt.title('Decision Tree Structure')
            plt.tight_layout()
            plt.savefig('static/dt_structure.png')
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
