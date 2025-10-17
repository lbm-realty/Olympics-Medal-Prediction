import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importing and reading the data
teams = pd.read_csv("teams.csv")

# Keeping only columns of interest
    # -> We'll use athletes and previous medals to predict future medals
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Encoding country and team string values 
le = LabelEncoder()
teams["country"] = le.fit_transform(teams["country"])
teams["team"] = le.fit_transform(teams["country"])

# Visualizing relationships
sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)


# teams.corr()["medals"]

