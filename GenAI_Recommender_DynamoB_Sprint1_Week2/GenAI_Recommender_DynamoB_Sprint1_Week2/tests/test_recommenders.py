import pandas as pd
from models.popularity_model import PopularityModel

def test_popularity():
    df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "product_id": [101, 102, 101],
        "score": [5, 4, 5]
    })

    model = PopularityModel()
    model.fit(df)

    recs = model.recommend(2)
    assert len(recs) == 2