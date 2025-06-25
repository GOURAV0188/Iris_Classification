import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('Iris.csv')
print(df.head())
print(df.info())



# Drop ID column if present
df.drop(['Id'], axis=1, inplace=True)

# Encode species labels
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])


# Pairplot
sns.pairplot(df, hue='Species')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
