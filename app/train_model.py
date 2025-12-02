# app/train_model.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


def train_and_save_model():
    # Cargamos dataset de ejemplo
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split simple (no nos complicamos)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo simple
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Guardar modelo
    model_path = Path(__file__).parent / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")


if __name__ == "__main__":
    train_and_save_model()
