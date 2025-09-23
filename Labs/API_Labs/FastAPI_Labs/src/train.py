from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from data import load_data, split_data

def fit_models(X_train, y_train):
    """
    Train Decision Tree, Random Forest, and Logistic Regression classifiers, saving all models to files.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, "../model/iris_model.pkl")

    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=12)
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, "../model/iris_model_randomforest.pkl")

    lr_classifier = LogisticRegression(max_iter=200, random_state=12)
    lr_classifier.fit(X_train, y_train)
    joblib.dump(lr_classifier, "../model/iris_model_logreg.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_models(X_train, y_train)
