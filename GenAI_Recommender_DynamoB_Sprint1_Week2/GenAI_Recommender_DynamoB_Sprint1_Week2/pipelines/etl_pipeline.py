import pandas as pd

def load_data():
    products = pd.read_parquet("products_clean.parquet")
    behavior = pd.read_parquet("user_behavior_clean.parquet")
    return products, behavior

def validate_data(products, behavior):
    assert 'product_id' in products.columns
    assert 'user_id' in behavior.columns
    assert 'event_type' in behavior.columns
    print("Data validated")

def create_interaction_score(df):
    df = df.copy()
    df['score'] = 0
    
    df.loc[df['event_type'] == 'view', 'score'] = 3
    df.loc[df['event_type'] == 'wishlist', 'score'] = 5
    df.loc[df['event_type'] == 'cart', 'score'] = 7
    df.loc[df['event_type'] == 'purchase', 'score'] = 20
    
    return df[['user_id', 'product_id', 'score']]

if __name__ == "__main__":
    #Loads the data
    prod, beh = load_data()
    validate_data(prod, beh)
    scored_df = create_interaction_score(beh)
    print("Pipeline executed successfully!")
    print(scored_df.head())
    print(beh['event_type'].unique())