
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 

"""
Naive Bayes Classifier for Categorical Data

    Implementation Steps:

        * Count class frequencies → priors

        * Count feature value frequencies per class → likelihoods

        * Use Laplace smoothing for zeros

        * Multiply prior × likelihoods

        * Choose the class with highest score

"""
class Nbayes:
    
    def __init__(self, smoothing=1):
        self.smoothing = smoothing
        self.trained = False

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        """
        X_train is a NumPy array shaped like this:
                (number of samples) × (number of features)
            
            where : 
            
                n = number of training samples (rows)

                d = number of features per sample (columns)"""

        n_samples, n_features = X_train.shape

        # Get all class labels
        self.classes = np.unique(y_train)

        # Count labels
        """
        Example

        If your y_train is: ["yes", "no", "yes", "yes", "no"] 
        
        Then,Counter(y_train) will give you: 
        {
            "yes": 3,
            "no": 2
        }
        So it creates a dictionary-like object:

        the keys are the class labels,

        the values are the number of occurrences of each class.
        """
        label_counts = Counter(y_train)

        # self.class_priors = {}  # create an empty dictionary
        # for c in self.classes:  # loop over each class
        #     self.class_priors[c] = label_counts[c] / len(y_train)  # compute prior probability

        self.class_priors = {
            c: label_counts[c] / len(y_train) for c in self.classes
        }

        """
        Determine possible values of each feature (important for Laplace smoothing)

        example : 
          [0]  outlook     : overcast, rainy, sunny
          [1]  temperature : hot, cold, mild
          [2]  humidity    : high, normal 
          [3]  windy       : false, true
        """
        self.feature_values = {}    # create an empty dictionary
        for j in range(n_features): # loop over each feature/column
            self.feature_values[j] = np.unique(X_train[:, j])  # X_train[:, j] : Take all rows, but only column j.


        # self.feature_values = {
        #     j: np.unique(X_train[:, j]) for j in range(d)
        # }

        # For each feature and each class, compute P(feature=value | class)
        self.cond_probs = {}  # dict: (feature_index, value, class) -> probability

        # Build probability tables
        for j in range(n_features):  # say we are working with feature 1
            for c in self.classes:   # that is in class "YES"
                X_class_c = X_train[y_train == c, j] # select all rows belonging to class c  
                value_counts = Counter(X_class_c)

                # For Laplace smoothing
                V = len(self.feature_values[j]) # V is the number of possible values of feature  (ex V_outlook = 3)

                for v in self.feature_values[j]:
                    # P(X_j = v | class=c) with Laplace smoothing
                    prob = (value_counts[v] + self.smoothing) / (len(X_class_c) + V)
                    self.cond_probs[(j, v, c)] = prob

        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise ValueError("Model not trained")

        posteriors = {}
        
        for c in self.classes:
            # Start with the prior
            posterior = self.class_priors[c]
            
            # Multiply likelihoods
            for j in range(len(X_test)):
                value = X_test[j]
                
                # Handle missing values
                if value not in self.feature_values[j]:
                    # Choose uniform probability or ignore the feature
                    prob = 1 / len(self.feature_values[j])
                else:
                    prob = self.cond_probs[(j, value, c)]

                posterior *= prob 
                
            posteriors[c] = posterior
        
        return posteriors

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model not trained")
        y_predict = self.predict(X_test)
         # return accuracy, fraction of correct predictions:
        return (np.array(y_test) == np.array(y_predict)).sum() / len(y_test)

# --- Load the dataset ---

# WEATHER DATASET 
# data_file = "weather.data"

# X = []
# y = []

# with open(data_file, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         if line.startswith("#") or line == "":
#             continue  # skip comments and empty lines
#         parts = line.split()
#         X.append(parts[:-1])  # all columns except last
#         y.append(parts[-1])   # last column is the label

# BREAST CANCER DATASET
breast_cancer = fetch_ucirepo(id=14)

X = breast_cancer.data.features
y = breast_cancer.data.targets.iloc[:, 0]   # convert to 1D array

# --- Fix missing values using mode ---
X = X.fillna(X.mode().iloc[0])

# Convert to numpy arrays for the classifier
X = np.array(X)
y = np.array(y)


# --- Split dataset into training and testing sets ---
# test_size = 0.3 means 30% of the data is used for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.45, random_state=42, shuffle=True
)

# --- Train the classifier ---
clf = Nbayes(smoothing=1)
clf.fit(X_train, y_train)

# --- Make predictions on the test set ---
y_pred = []
for sample in X_test:
    posteriors = clf.predict(sample)
    predicted_class = max(posteriors, key=posteriors.get)
    y_pred.append(predicted_class)

y_pred = np.array(y_pred)

# --- Calculate accuracy ---
accuracy = (y_pred == y_test).sum() / len(y_test)
print(f"Accuracy on the test data: {accuracy*100:.2f}%")
