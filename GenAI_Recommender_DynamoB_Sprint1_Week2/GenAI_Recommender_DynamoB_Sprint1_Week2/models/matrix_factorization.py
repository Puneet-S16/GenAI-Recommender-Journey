import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=self.n_components)
        self.user_item_matrix = None
        self.predicted_matrix = None
        self.user_means = None

    def fit(self, df):
        # Create user-item matrix
        self.user_item_matrix = df.pivot_table(
            index='user_id',
            columns='product_id',
            values='score',
            fill_value=0
        )

        #normalize by user means
        self.user_means = self.user_item_matrix.mean(axis=1)
        matrix_centered = self.user_item_matrix.sub(self.user_means, axis=0)
        #apply SVD
        u_features = self.model.fit_transform(matrix_centered)
        vt_features = self.model.components_
        #reconstruct the matrix
        reconstructed = np.dot(u_features, vt_features)

        # Add means back
        reconstructed = reconstructed + self.user_means.values.reshape(-1, 1)

        # Convert to DataFrame
        self.predicted_matrix = pd.DataFrame(
            reconstructed,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_item_matrix.index:
            return []

        user_predictions = self.predicted_matrix.loc[user_id]

        # Remove already interacted items
        already_interacted = self.user_item_matrix.loc[user_id]
        user_predictions = user_predictions[already_interacted == 0]

        return (
            user_predictions
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )