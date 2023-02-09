# ---------------------------------- cosine and knn configs ---------------------------------- #

COSINE_NEIGHBORS = 30
cosine_model = "23090"
source_name_model = "paraphrase-multilingual-mpnet-base-v2"
category_model = "all-MiniLM-L6-v2"
country = "EGY"
non_english=True
# ------------------------------ pipeline config ----------------------------- #

is_inference = False

threshold_dict = {"NZL": 0.0001, "GLP": 0.0001, "ZAF": 0.0001, "AUS": 0.0001,"BEL":0.0001}

cols_to_block = [
    "sourceNames", "latitude", "longitude", "houseNumber", "streets", "cities",
    "brands", "postalCode", "category"
]

columns_keep = [
    "locality", "clusterId", "placeId", "officialName","sourceNames","subCategory", "category", "latitude",
    "longitude", "houseNumber", "streets", "cities", "postalCode", "Id",
    "brands", "phoneNumbers", "supplier","email","internet"
]

# -------------------------- directory configuration ------------------------- #
root_dir = "/workspace/clustering/"
model_dir = "/workspace/clustering/models/"
raw_dir = "/workspace/clustering/data/raw/"
input_dir = "/workspace/clustering/data/input/"
gt_dir = "/workspace/clustering/data/ground_truth/"
embed_dir = "/workspace/clustering/embeddings/"
output_stage1 = "/workspace/clustering/outputs/stage1/"
output_stage2 = "/workspace/clustering/outputs/stage2/"
final_result_dir = "/workspace/clustering/outputs/results/"

# ----------------------- model features and parameters ---------------------- #

lgb_level1_params = {
    "learning_rate": 0.05,
    "num_leaves": 500,
    "reg_alpha": 1,
    "reg_lambda": 10,
    "min_child_samples": 1000,
    "min_split_gain": 0.01,
    "min_child_weight": 0.01,
    "path_smooth": 0.1
}

L2_features = [
    'all_similarities', 'latdiff', 'londiff', 'manhattan', 'euclidean',
    'haversine', 'sourceNames_jaro', 'sourceNames_set_ratio',
    'sourceNames_WRatio', 'sourceNames_ratio', 'sourceNames_qratio',
    'sourceNames_nleven', 'houseNumber_nleven', 'houseNumber_jaro',
    'category_jaro', 'category_set_ratio', 'category_WRatio', 'category_ratio',
    'category_qratio', 'category_nleven', 'streets_nleven', 'streets_jaro',
    'cities_nleven', 'cities_jaro', 'postalCode_nleven', 'postalCode_jaro',
    'brands_nleven', 'brands_jaro', 'phoneNumbers_lcs', 'supplier_match'
]

L3_features = [
    'all_similarities', 'name_similarities', 'category_similarities',
    'cities_similarities', 'streets_similarities', 'latdiff', 'londiff',
    'manhattan', 'euclidean', 'haversine', 'sourceNames_jaro',
    'sourceNames_set_ratio', 'sourceNames_WRatio', 'sourceNames_ratio',
    'sourceNames_qratio', 'sourceNames_nleven', 'sourceNames_Dice',
    'sourceNames_lcs', 'sourceNames_davies', 'sourceNames_soundex',
    'sourceNames_metaphone', 'houseNumber_nleven', 'houseNumber_jaro',
    'category_jaro', 'category_set_ratio', 'category_WRatio', 'category_ratio',
    'category_qratio', 'category_nleven', 'category_Dice', 'category_lcs',
    'category_davies', 'category_soundex', 'category_metaphone',
    'streets_jaro', 'streets_set_ratio', 'streets_WRatio', 'streets_ratio',
    'streets_qratio', 'streets_nleven', 'streets_Dice', 'streets_lcs',
    'streets_davies', 'streets_soundex', 'streets_metaphone', 'cities_jaro',
    'cities_set_ratio', 'cities_WRatio', 'cities_ratio', 'cities_qratio',
    'cities_nleven', 'cities_Dice', 'cities_lcs', 'cities_davies',
    'cities_soundex', 'cities_metaphone', 'postalCode_nleven',
    'postalCode_jaro', 'brands_nleven', 'brands_jaro', 'phoneNumbers_lcs',
    'phoneNumbers_nleven', 'phoneNumbers_jaro', 'supplier_match'
]


class CFG:
    wandb = True
    apex = True
    print_freq = 400
    num_workers = 4
    out_dir = "/workspace/clustering/models/"
    batch_size = 64
    max_len = 128
    seed = 42
    text_cols = [
        'sourceNames', 'category', 'houseNumber', 'streets', 'cities','postalCode']
    numeric_cols = ['haversine', 'latdiff', 'manhattan', 'euclidean', 'all_similarities', 'name_similarities', 'category_similarities','cities_similarities', 'streets_similarities']
