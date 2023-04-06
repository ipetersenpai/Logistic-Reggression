import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add your testing size set here and it must be float
testing_set = 0.3 # the percentage value is 30% out of 100% total of data.

# Load Fisher Iris dataset
iris_df = pd.read_csv('iris.csv')

# Convert to a Pandas DataFrame for easier data manipulation
iris_df = sns.load_dataset("iris")

# Split the dataset into features (X) and target (y)
X = iris_df.drop("species", axis=1)
y = iris_df["species"]

# Convert target labels to numeric values
species_to_numeric = {"setosa": 0, "versicolor": 1, "virginica": 2}
y = y.map(species_to_numeric)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_set, random_state=42)

# Create a logistic regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Predict the target labels for the test data
y_pred = logreg.predict(X_test)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a scatter plot with the logistic regression line
sns.scatterplot(x=X_test["sepal_width"], y=y_test, hue=y_pred, palette="viridis")
sns.regplot(x=X_test["sepal_width"], y=y_pred, logistic=True, scatter=False, color="red", line_kws={"linewidth": 2})
plt.xlabel("Sepal Width")
plt.ylabel("Species")
plt.title("Logistic Regression on Fisher Iris Dataset")
plt.show()
