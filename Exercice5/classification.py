import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, r2_score

from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Loading data
inputs = np.load('../data/classification/inputs.npy')
labels = np.load('../data/classification/labels.npy')

ss = StandardScaler()
inputs = ss.fit_transform(inputs)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np.ravel(labels)

classification_models = {
    "SVC": SVC(),
    "LinearSVC": LinearSVC(C=1),
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "Perceptron" : Perceptron(),
    "SGD" : SGDClassifier(),
    "DecisionTree" : DecisionTreeClassifier()
}

classification_scores = {}

def classification_testing(classifier):
    print(f"Testing {classifier[0]}...")
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    model = classifier[1]
    print("Fitting...")
    model.fit(X_train, y_train)
    print("Predicting...")
    print("Scoring...")
    scores = cross_val_score(model, inputs, labels, cv=10, scoring="accuracy")
    score = scores.mean()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print(f"{classifier[0]} score : {score}")
    classification_scores[classifier[0]] = score

for key, value in classification_models.items():
    classification_testing((key, value))
    
for i, classifier in enumerate(classification_scores.items()):
    print(f"{classifier[0]} score : {classifier[1]}")
    plt.bar(i, classifier[1], label=classifier[0])
plt.legend()
plt.show()