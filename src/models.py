
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train_lda_classifier(X_train_features, y_train):
    """
    Train an LDA classifier on CSP features.
    """
    X_train_features = np.asarray(X_train_features)
    y_train = np.asarray(y_train)

    if X_train_features.ndim != 2:
        raise ValueError("Training features must be a 2D array.")

    if len(X_train_features) != len(y_train):
        raise ValueError("Features and labels must have equal lengths.")

    if not np.isfinite(X_train_features).all():
        raise ValueError("Training features contain NaN or infinity.")

    if len(np.unique(y_train)) < 2:
        raise ValueError("LDA requires at least two classes.")

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(X_train_features, y_train)

    return model


def predict_classes(model, X_features):
    """
    Predict motor imagery classes from CSP features.
    """
    X_features = np.asarray(X_features)

    if X_features.ndim != 2:
        raise ValueError("Features must be a 2D array.")

    if not np.isfinite(X_features).all():
        raise ValueError("Features contain NaN or infinity.")

    return model.predict(X_features)
