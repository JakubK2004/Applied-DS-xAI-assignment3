import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error

stemmer = SnowballStemmer('english')

# Data loading
DATA_DIR = Path(__file__).parent.parent / "data"

df_train = pd.read_csv(DATA_DIR / 'train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR / 'test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv(DATA_DIR / 'attributes.csv')
df_pro_desc = pd.read_csv(DATA_DIR / 'product_descriptions.csv')

num_train = df_train.shape[0]

# Helpers
def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())


def build_features(stem=True):
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
	df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

	if stem:
		df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
		df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
		df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

	df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
	df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
	df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
	df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
	df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1)

	train_part = df_all.iloc[:num_train]
	test_part = df_all.iloc[num_train:]

	y = train_part['relevance'].values
	X = train_part.drop(['id', 'relevance'], axis=1).values
	X_test = test_part.drop(['id', 'relevance'], axis=1).values
	id_test = test_part['id']

	return X, y, X_test, id_test


def make_model():
	rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
	return BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)


# --- With stemming (baseline) ---
X, y, X_test, id_test = build_features(stem=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = make_model()
clf.fit(X_train, y_train)
rmse_stem = np.sqrt(mean_squared_error(y_val, clf.predict(X_val)))
print(f"RMSE (with stemming):    {rmse_stem:.4f}")

# --- Without stemming ---
X_ns, y_ns, X_test_ns, _ = build_features(stem=False)
X_train_ns, X_val_ns, y_train_ns, y_val_ns = train_test_split(X_ns, y_ns, test_size=0.2, random_state=42)

clf_ns = make_model()
clf_ns.fit(X_train_ns, y_train_ns)
rmse_no_stem = np.sqrt(mean_squared_error(y_val_ns, clf_ns.predict(X_val_ns)))
print(f"RMSE (without stemming): {rmse_no_stem:.4f}")

# Submission using stemmed model
y_test_pred = clf.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_test_pred}).to_csv('submission.csv', index=False)
