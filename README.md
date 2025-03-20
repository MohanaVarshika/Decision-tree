# Decision-tree

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: BEESETTY MOHANA VARSHIKA

**INTERN ID**: CT12WKFH

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**: JANUARY 10TH,2025 TO APRIL 10TH,2025

**MENTOR NAME**: NEELA SANTHOSH

**DESCRIPTION**: 

A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It is a tree-like model where data is split into branches based on specific conditions, making it easy to interpret and visualize. Decision trees are widely used because of their simplicity, effectiveness, and ability to handle both numerical and categorical data.
=>A decision tree consists of the following components:
Root Node: This is the top-most node of the tree, representing the entire dataset. It is the starting point of the decision-making process.
Decision Nodes: These are intermediate nodes that split the data based on a feature and its threshold value.
Leaf Nodes: These are the terminal nodes that contain the final output or decision (class labels in classification tasks or values in regression tasks).
Branches: These are the connections between nodes, representing the flow of decision-making.
=>A decision tree works by recursively splitting the dataset based on certain conditions. The key steps in building a decision tree are:
Feature Selection: Identify the best feature to split the data at each step. This is done using criteria like Gini Index, Entropy (Information Gain), or Mean Squared Error (MSE) for regression tasks.
Splitting: The dataset is split into subsets based on the selected feature. Each subset becomes a child node.
Stopping Criteria: The recursion continues until a predefined stopping criterion is met. This could be:
A node containing samples of only one class (pure node).
The maximum depth of the tree is reached.
A minimum number of samples per leaf node is met.
Prediction: Once the tree is built, it is used for prediction by traversing from the root node to a leaf node based on feature values.
=>Applications of Decision Trees
Medical Diagnosis: Used in disease prediction based on symptoms and test results.
Finance and Banking: Used for credit risk analysis and loan approvals.
Marketing: Helps in customer segmentation and product recommendations.
Fraud Detection: Identifies fraudulent transactions in real-time.
Manufacturing: Used in quality control and defect detection.

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Optional: Predicting for a single example
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
prediction = clf.predict(sample)
print(f"Predicted Class: {iris.target_names[prediction[0]]}")   

EXPLANATION:
Key Components of the Code
Loading the Dataset:

The load_iris() function loads the famous Iris dataset, which consists of 150 samples from three classes (Setosa, Versicolor, and Virginica) with four features (sepal length, sepal width, petal length, petal width).
Splitting the Dataset:

train_test_split() is used to divide the dataset into 70% training data and 30% test data, ensuring the model is trained on a portion of the data and tested on unseen data.
Initializing and Training the Decision Tree:

A DecisionTreeClassifier is initialized with the Gini impurity criterion and a max_depth of 3, restricting tree growth to prevent overfitting.
The classifier is trained using clf.fit(X_train, y_train).
Making Predictions:

The trained model predicts class labels for X_test, and accuracy is computed using accuracy_score().
Visualizing the Decision Tree:

tree.plot_tree() generates a graphical representation of the decision tree, showing the decision splits based on feature values.
Single Example Prediction:

A sample flower measurement ([5.1, 3.5, 1.4, 0.2]) is used to predict the flower species.

# OUTPUT

![Image](https://github.com/user-attachments/assets/6f010d56-3ae0-419e-8406-313c5d8f71d0)
![Image](https://github.com/user-attachments/assets/cd6c94d9-e4b0-4f71-b0db-e5116337076a)


