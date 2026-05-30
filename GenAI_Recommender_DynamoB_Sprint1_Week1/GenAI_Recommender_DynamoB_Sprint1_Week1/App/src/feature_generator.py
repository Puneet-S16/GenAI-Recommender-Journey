import pandas as pd
products = pd.read_parquet("../data/processed/products_clean.parquet")
behavior = pd.read_parquet("../data/processed/user_behavior_clean.parquet")
purchases = behavior[behavior['event_type'] == 'purchase'].copy()
purchases = purchases.merge(
    products[['product_id', 'price']],
    on='product_id',
    how='left'
)
purchases['timestamp'] = pd.to_datetime(purchases['timestamp'])
reference_date = pd.to_datetime('2023-12-31')
recency = purchases.groupby('user_id')['timestamp'].max().reset_index()
recency['recency'] = (reference_date - recency['timestamp']).dt.days

recency = recency[['user_id', 'recency']]
frequency = purchases.groupby('user_id').size().reset_index(name='frequency')
monetary = purchases.groupby('user_id')['price_x'].sum().reset_index()
monetary.rename(columns={'price_x': 'monetary'}, inplace=True)
rfm = recency.merge(frequency, on='user_id')
rfm = rfm.merge(monetary, on='user_id')
all_users = behavior[['user_id']].drop_duplicates()

rfm = all_users.merge(rfm, on='user_id', how='left')

rfm.fillna({
    'recency': 999,
    'frequency': 0,
    'monetary': 0
}, inplace=True)

print(rfm.head())
print(rfm.describe())

rfm.to_parquet("../data/processed/user_features.parquet", index=False)