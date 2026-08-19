# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset containing sepal length, sepal width, petal length and petal width.

2.Split the dataset into training and testing data.

3.Train an SGDClassifierusing the training data.

4.Predict the species of a new Iris flower and calculate the model accuracy.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SIVAPRASATH B
RegisterNumber:  212225230268
*/

# Iris Species Prediction using SGD Classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier
model = SGDClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Predict species for a new Iris flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction[0]])

```

## Output:

<img width="527" height="121" alt="image" src="https://github.com/user-attachments/assets/ddb79cba-ce26-469e-8db1-ca94eeea59b3" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
