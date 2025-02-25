# import all required module
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# check rmse, mean, standard deviation
def check_scores(scores):
    print('Scores : ', scores)
    print('Mean : ', scores.mean())
    print('Standard Deviation : ', scores.std())

DataFrame = pd.read_csv('housing.csv')

# slit dataset
split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4)
for train_index, test_index in split_data.split(DataFrame, DataFrame['CHAS']):
    train_set = DataFrame.loc[train_index]
    test_set = DataFrame.loc[test_index]

x_train, y_train = train_set.drop('MEDV', axis=1), train_set['MEDV']
x_test, y_test = test_set.drop('MEDV', axis=1), test_set['MEDV']

# create pipeline
# fill median instead of missing value(imputer) and scale features
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
train = my_pipeline.fit_transform(x_train)
print(train.shape)

# Create model
model = RandomForestRegressor()
model.fit(train, y_train)

# evaluate model
scores = cross_val_score(model, train, y_train, scoring='neg_mean_squared_error', cv=10)
# root mean squared error
rmse_scroes = np.sqrt(-scores)
check_scores(rmse_scroes)

# model testing
test = my_pipeline.transform(x_test)
prediction = model.predict(test)
print(prediction, '\n\n', list(y_test))