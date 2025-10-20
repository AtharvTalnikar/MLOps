from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


if __name__ == "__main__":
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a simple and robust classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Persist both scaler and model for consistent inference
    joblib.dump({"scaler": scaler, "model": clf}, "iris_rf.joblib")
    print("Sklearn model trained and saved to iris_rf.joblib")

