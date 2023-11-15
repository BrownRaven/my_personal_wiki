# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetRegressor
from dataloader import fetch_YEAR
SEED = 42
np.random.seed(SEED)
# %%
CATEGORICAL_COLUMNS = ['neighbourhood_group', 'room_type']
DROP_COLUMNS = ['id', 'name', 'host_id',
                'host_name', 'last_review', 'neighbourhood']
assert len(set(CATEGORICAL_COLUMNS).intersection(set(DROP_COLUMNS))) == 0
# %%
data = pd.read_csv('/workspace/data/AB_NYC_2019.csv')
# %%
df = data.drop(DROP_COLUMNS, axis=1)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0.)
df = pd.concat(
    [df, pd.get_dummies(df[CATEGORICAL_COLUMNS], dtype=np.float64)], axis=1)
df = df.drop(CATEGORICAL_COLUMNS, axis=1)
# %%
y = df[['price']]
X = df.drop(['price'], axis=1)
scalar_X = StandardScaler()
X = scalar_X.fit_transform(X)
scalar_y = StandardScaler()
y = scalar_y.fit_transform(y)
# %%
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=SEED)
model = TabNetRegressor(seed=SEED)
model.fit(train_X, train_y, eval_set=[(test_X, test_y)])

# %%
pred = model.predict(test_X)
# %%
scalar_y.inverse_transform(pred)
# %%
data = fetch_YEAR('/workspace/data/YEAR')
train_y, valid_y, test_y = data['y_train'], data['y_valid'], data['y_test']
train_X, valid_X, test_X = data['X_train'], data['X_valid'], data['X_test']

mu, std = train_y.mean(), train_y.std()
def normalize(x): return ((x - mu) / std).astype(np.float32)


train_y, valid_y, test_y = map(normalize, [train_y, valid_y, test_y])

model = TabNetRegressor(seed=SEED, device_name='cuda')
model.fit(train_X, train_y.reshape(-1, 1),
          eval_set=[(valid_X, valid_y.reshape(-1, 1))])
# %%
