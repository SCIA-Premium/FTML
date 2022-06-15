from statistics import mode
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def regression():
    inputs = np.load('../data/regression/inputs.npy')
    labels = np.load('../data/regression/labels.npy')

    #labelEncoder = LabelEncoder()
    #labels_transformed = labelEncoder.fit_transform(labels)
    scaler = MinMaxScaler()
    labels_transformed = scaler.fit_transform(labels)

    print(labels_transformed)


    
    print(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    #print(labels)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    #scores = cross_val_score(model, inputs, labels, cv=5)
    #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


regression()