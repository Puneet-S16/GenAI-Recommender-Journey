import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None

    def fit(self, df):
        self.user_item_matrix = df.pivot_table(
            index='user_id',
            columns='product_id',
            values='score',
            fill_value=0
        )

        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

    def recommend_user_based(self, user_id, top_n=10):
        if user_id not in self.user_item_matrix.index:
            return []

        user_idx = self.user_item_matrix.index.get_loc(user_id)

        sim_scores = list(enumerate(self.user_similarity[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        similar_users = [i[0] for i in sim_scores[1:6]]

        recommendations = set()
        for u in similar_users:
            items = self.user_item_matrix.iloc[u]
            liked = items[items > 0].index
            recommendations.update(liked)

        return list(recommendations)[:top_n]

    def recommend_item_based(self, product_id, top_n=10):
        if product_id not in self.user_item_matrix.columns:
            return []

        item_idx = self.user_item_matrix.columns.get_loc(product_id)

        sim_scores = list(enumerate(self.item_similarity[item_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        return [
            int(self.user_item_matrix.columns[i[0]])
            for i in sim_scores[1:top_n+1]
        ]