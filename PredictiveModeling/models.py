"""
These are snippets from the research code I used to experiment with different models, hyperparameter tuning and cross
validation.
"""
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.ensemble import GradientBoostingClassifier

RANDOM_STATE = 1


def get_data_from_svmlight(svmlight_file):
    """
    input: features and label stored in the svmlight_file
    output: X_train, Y_train
    """

    data_train = load_svmlight_file(svmlight_file, n_features=3190)
    X_train = data_train[0]
    Y_train = data_train[1]
    return X_train, Y_train


def logistic_regression_pred(X_train, Y_train, X_test):
    """
    Trains the logistic regression classifier and performs inference on the test examples.
    Input: training features, training labels, test features.
    """
    classifier = LogisticRegression(random_state=RANDOM_STATE)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


def svm_pred(X_train, Y_train, X_test):
    """
    Trains the SVM classifier and performs inference on the test examples.
    Input: training features, training labels, test features.
    """
    classifier = LinearSVC(random_state=RANDOM_STATE)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


def decisionTree_pred(X_train, Y_train, X_test):
    """
    Trains the decision tree classifier and performs inference on the test examples.
    Input: training features, training labels, test features.
    """
    classifier = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


def gradientBoost_pred(X_train, Y_train, X_test):
    """
    Trains the gradient boosting classifier and performs inference on the test examples.
    Input: training features, training labels, test features.
    """
    classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.17, max_depth=6)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    return y_pred


def classification_metrics(Y_pred, Y_true):
    """
    Calculates model performance metrics.
    Input: predicted labels, actual labels.
    Output: accuracy, area under the curve, precision, recall and f1 score.
    """
    accuracy = accuracy_score(Y_true, Y_pred)

    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
    a_u_c = auc(fpr, tpr)

    precision = precision_score(Y_true, Y_pred)

    recall = recall_score(Y_true, Y_pred)

    f1score = f1_score(Y_true, Y_pred)

    return accuracy, a_u_c, precision, recall, f1score


def display_metrics(classifierName, Y_pred, Y_true):
    """
    Displays the calculated model performance metrics.
    Input: Name of classifier, predicted labels, actual labels.
    """
    print("______________________________________________")
    print(("Classifier: " + classifierName))
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred, Y_true)
    print(("Accuracy: " + str(acc)))
    print(("AUC: " + str(auc_)))
    print(("Precision: " + str(precision)))
    print(("Recall: " + str(recall)))
    print(("F1-score: " + str(f1score)))
    print("______________________________________________")
    print("")


def main():
    X_train, Y_train = get_data_from_svmlight("../output/features_svmlight.train")
    X_test, Y_test = get_data_from_svmlight("../data/features_svmlight.validate")

    display_metrics("Logistic Regression", logistic_regression_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("SVM", svm_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("Decision Tree", decisionTree_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("Gradient Boosting", gradientBoost_pred(X_train, Y_train, X_test), Y_test)


if __name__ == "__main__":
    main()
