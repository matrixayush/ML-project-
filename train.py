import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
df = pd.read_csv('data/Prostate_Cancer.csv')
X = df.drop(columns=['id', 'diagnosis_result'])
y = df['diagnosis_result'].map({'B': 0, 'M': 1})

# Split into train/test with 50% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler
os.makedirs('weights', exist_ok=True)
with open('weights/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save test data
X_test.to_csv('weights/X_test.csv', index=False)
y_test.to_csv('weights/y_test.csv', index=False)

# Train and save Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
with open('weights/logreg_model.pkl', 'wb') as f:
    pickle.dump(logreg, f)

# Train and save KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
with open('weights/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Train and save Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
with open('weights/dt_model.pkl', 'wb') as f:
    pickle.dump(dt, f)

# Train and save Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
with open('weights/nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)

# Train and save Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)
with open('weights/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("All weights and scaler saved successfully!")
