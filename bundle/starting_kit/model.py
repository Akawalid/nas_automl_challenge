import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class Model:
    def __init__(self):
        """
        KNN regressor for volumetric density (rho) prediction.
        Predicts rho at each point from (x, y, z, nx, ny, nz, Minf, AoA, pi).
        k is tuned following the paper: best k found around 6-9 for surface fields.
        Here k=7 as a reasonable default for the volumetric case.
        """
        self.regressor = KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            algorithm='auto',
            n_jobs=-1
        )

    def fit(self, X, y):
        """
        Train the model.
        Args:
            X: Training data matrix of shape (num_samples, 9), type np.ndarray.
               Columns: x, y, z, nx, ny, nz, Minf, AoA, pi
            y: Training target vector of shape (num_samples,), type np.ndarray.
               Values: adimensional volumetric density rho
        """
        self.regressor.fit(X, y)

    def predict(self, X):
        """
        Predict rho values.
        Args:
            X: Data matrix of shape (num_samples, 9), type np.ndarray.
        Returns:
            y: Predicted rho vector of shape (num_samples,), type np.ndarray.
        """
        return self.regressor.predict(X)