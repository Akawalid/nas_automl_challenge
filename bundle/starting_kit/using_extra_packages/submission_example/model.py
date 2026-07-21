import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent / "python_packages"))
import lightgbm as lgb


class Model:
    def __init__(self):
        """
        LightGBM regressor for volumetric density (rho) prediction.
        Predicts rho at each point from (x, y, z, nx, ny, nz, Minf, AoA, pi).
        """
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            n_jobs=-1, 
            random_state=42
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
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict rho values.
        Args:
            X: Data matrix of shape (num_samples, 9), type np.ndarray.
        Returns:
            y: Predicted rho vector of shape (num_samples,), type np.ndarray.
        """
        return self.model.predict(X)