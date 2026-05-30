from pipelines.etl_pipeline import load_data, validate_data, create_interaction_score
from models.popularity_model import PopularityModel
from models.collaborative_filter import CollaborativeFiltering
from models.matrix_factorization import SVDRecommender as ALSModel

# Load
products, behavior = load_data()
validate_data(products, behavior)

# Create scores
interaction_df = create_interaction_score(behavior)

#popularity 
pop = PopularityModel()
pop.fit(interaction_df)
pop_recs = pop.recommend()
print("Popularity:", pop_recs)

#collaborative filtering - user-based and item-based
cf = CollaborativeFiltering()
cf.fit(interaction_df)

user_id = interaction_df['user_id'].iloc[0]

print("User-CF:", cf.recommend_user_based(user_id))
print("Item-CF:", cf.recommend_item_based(pop_recs[0]))

#matrix factorization 
als = ALSModel()
als.fit(interaction_df)

print("ALS:", als.recommend(user_id))

#final output
def show_products(ids):
    return products[products['product_id'].isin(ids)][['product_id', 'name']]

print("CF Recommendations:")
print(show_products(cf.recommend_user_based(user_id)))

print("ALS Recommendations:")
print(show_products(als.recommend(user_id)))

print("Final Output:")
print(show_products(pop_recs))

