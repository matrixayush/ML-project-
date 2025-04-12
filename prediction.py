import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# Create a directory to save the images if it doesn't exist
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Load the test data
X_test = pd.read_csv('weights/X_test.csv')
y_test = pd.read_csv('weights/y_test.csv')

# Load the scaler
with open('weights/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the models
with open('weights/logreg_model.pkl', 'rb') as f:
    logreg = pickle.load(f)

with open('weights/knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('weights/dt_model.pkl', 'rb') as f:
    dt = pickle.load(f)

with open('weights/nb_model.pkl', 'rb') as f:
    nb = pickle.load(f)

with open('weights/rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Get predictions from each model
logreg_pred = logreg.predict(X_test_scaled)
knn_pred = knn.predict(X_test_scaled)
dt_pred = dt.predict(X_test_scaled)
nb_pred = nb.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)

# Generate confusion matrices for each model
logreg_cm = confusion_matrix(y_test, logreg_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Plot confusion matrices as heatmaps and save them
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Logistic Regression Confusion Matrix
sns.heatmap(logreg_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Logistic Regression')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# KNN Confusion Matrix
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('KNN')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Decision Tree Confusion Matrix
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2], cbar=False)
axes[0, 2].set_title('Decision Tree')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('Actual')

# Naive Bayes Confusion Matrix
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Naive Bayes')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# Random Forest Confusion Matrix
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('Random Forest')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

# Remove empty subplot (2,2)
axes[1, 2].axis('off')

plt.tight_layout()
confusion_matrix_file = os.path.join(output_dir, 'confusion_matrices.png')
plt.savefig(confusion_matrix_file)
plt.close()

# Calculate accuracy for comparison
logreg_acc = accuracy_score(y_test, logreg_pred)
knn_acc = accuracy_score(y_test, knn_pred)
dt_acc = accuracy_score(y_test, dt_pred)
nb_acc = accuracy_score(y_test, nb_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# Calculate Precision, Recall, F1-Score for each model
logreg_prec = precision_score(y_test, logreg_pred)
knn_prec = precision_score(y_test, knn_pred)
dt_prec = precision_score(y_test, dt_pred)
nb_prec = precision_score(y_test, nb_pred)
rf_prec = precision_score(y_test, rf_pred)

logreg_rec = recall_score(y_test, logreg_pred)
knn_rec = recall_score(y_test, knn_pred)
dt_rec = recall_score(y_test, dt_pred)
nb_rec = recall_score(y_test, nb_pred)
rf_rec = recall_score(y_test, rf_pred)

logreg_f1 = f1_score(y_test, logreg_pred)
knn_f1 = f1_score(y_test, knn_pred)
dt_f1 = f1_score(y_test, dt_pred)
nb_f1 = f1_score(y_test, nb_pred)
rf_f1 = f1_score(y_test, rf_pred)

# Print classification reports
print("Logistic Regression Classification Report:\n", classification_report(y_test, logreg_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Create a comparison bar chart of accuracies, precision, recall, and F1-scores
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
logreg_values = [logreg_acc, logreg_prec, logreg_rec, logreg_f1]
knn_values = [knn_acc, knn_prec, knn_rec, knn_f1]
dt_values = [dt_acc, dt_prec, dt_rec, dt_f1]
nb_values = [nb_acc, nb_prec, nb_rec, nb_f1]
rf_values = [rf_acc, rf_prec, rf_rec, rf_f1]

# Plotting the comparison
x = range(len(metrics))

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.15  # Bar width
logreg_bars = ax.bar(x, logreg_values, width, label='Logistic Regression')
knn_bars = ax.bar([p + width for p in x], knn_values, width, label='KNN')
dt_bars = ax.bar([p + width*2 for p in x], dt_values, width, label='Decision Tree')
nb_bars = ax.bar([p + width*3 for p in x], nb_values, width, label='Naive Bayes')
rf_bars = ax.bar([p + width*4 for p in x], rf_values, width, label='Random Forest')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison (Accuracy, Precision, Recall, F1-Score)')
ax.set_xticks([p + width*2 for p in x])
ax.set_xticklabels(metrics)
ax.legend()

# Add values on top of the bars
for bars in [logreg_bars, knn_bars, dt_bars, nb_bars, rf_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
comparison_chart_file = os.path.join(output_dir, 'model_comparison_with_values.png')
plt.savefig(comparison_chart_file)
plt.close()

print(f"Confusion matrices and comparison graph saved in '{output_dir}' directory.")
