import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(42)
np.random.seed(42)
NUM_PRODUCTS = 500
NUM_USERS = 2000

categories = ['Sofa', 'Chair', 'Table', 'Bed', 'Wardrobe']
materials_map = {
    'Sofa': ['Fabric', 'Leather'],
    'Chair': ['Wood', 'Metal'],
    'Table': ['Wood', 'Glass'],
    'Bed': ['Wood'],
    'Wardrobe': ['Wood']
}
colors = ['Black', 'White', 'Grey', 'Brown', 'Beige', 'Blue']
styles = ['Modern', 'Vintage', 'Minimalist', 'Industrial']
#gen products 
def generate_products(n):
    data = []

    for i in range(n):
        category = random.choice(categories)
        material = random.choice(materials_map[category])

        price = {
            'Chair': random.randint(50, 200),
            'Table': random.randint(150, 600),
            'Sofa': random.randint(500, 1500),
            'Bed': random.randint(400, 1200),
            'Wardrobe': random.randint(300, 1000)
        }[category]

        data.append({
            "product_id": i + 1,
            "name": f"{fake.word().capitalize()} {category}",
            "category": category,
            "material": material,
            "color": random.choice(colors),
            "style": random.choice(styles),
            "price": price,
            "rating": round(random.uniform(2.5, 5.0), 1),
            "description": fake.sentence()
        })

    return pd.DataFrame(data)

#gen users
def generate_users(n):
    users = []

    for i in range(n):
        users.append({
            "user_id": i + 1,
            "budget_level": random.choice(['low', 'mid', 'high'])
        })

    return pd.DataFrame(users)

#user behavior calculation hai 
def generate_behavior(users_df, products_df):
    records = []
    event_types = ['view', 'cart', 'wishlist', 'purchase']

    for _ in range(len(users_df) * 15):
        user = users_df.sample(1).iloc[0]
        product = products_df.sample(1).iloc[0]

        event = random.choices(
            event_types,
            weights=[0.7, 0.15, 0.1, 0.05]
        )[0]

        timestamp = datetime.now() - timedelta(days=random.randint(1, 365))

        records.append({
            "user_id": user['user_id'],
            "product_id": product['product_id'],
            "event_type": event,
            "timestamp": timestamp
        })

    return pd.DataFrame(records)
#run
def run():
    products = generate_products(NUM_PRODUCTS)
    users = generate_users(NUM_USERS)
    behavior = generate_behavior(users, products)

    products.to_csv("data/raw/products.csv", index=False)
    behavior.to_csv("data/raw/user_behavior.csv", index=False)

    print("Data Generated")

if __name__ == "__main__":
    run()