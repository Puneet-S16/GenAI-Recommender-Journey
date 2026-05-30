class PopularityModel:
    def __init__(self):
        self.popular_items = None

    def fit(self, df):
        self.popular_items = (
            df.groupby('product_id')['score']
            .sum()
            .sort_values(ascending=False)
        )

    def recommend(self, top_n=10):
        return self.popular_items.head(top_n).index.tolist()