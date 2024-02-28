import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# a) Read the Mobile price dataset using the Pandas module
df = pd.read_csv('C:/Users/HP/Documents/python(ML)/mobile_prices.csv')

# b) print the 1st five rows
print(df.head())

# c) Basic statistical computations on the data set or distribution of data
print(df.describe())

# d) the columns and their data types
print(df.dtypes)

# 2) Detect null values in the dataset if there is any null values replaced it with mode value
print(df.isnull().sum())
df = df.fillna(df.mode().iloc[0])

# Explore the data set using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.show()

# 4) Spit the data in to test and train
X = df.drop('price_range', axis=1)  # Assuming the target variable column is named 'target_variable_column'
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# h) Fit in to the model Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict the model
y_pred = model.predict(X_test)

# Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)
