import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# Importing and reading the data
teams = pd.read_csv("teams.csv")

# Keeping only columns of interest
    # -> We'll use athletes and previous medals to predict future medals
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Encoding country and team string values 
le = LabelEncoder()
teams["country"] = le.fit_transform(teams["country"])
teams["team"] = le.fit_transform(teams["country"])
teams.corr()["medals"]

# Visualizing relationships
sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)
teams.plot.hist(y="medals")
teams[teams.isnull().any(axis=1)]
teams = teams.dropna()

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train[predictors], train[target])

predictions = reg.predict(test[predictors])
test["predictions"] = predictions

test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()

# Error analysis in depth
error = mean_absolute_error(test["medals"], test["predictions"])
errors = (test["medals"] - test["predictions"]).abs()
error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()
