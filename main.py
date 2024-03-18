import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Clique_master import Clique as clique
from Clique_master import Cluster

def loadData (name: str = "Traffic_Crashes.csv") -> pd.DataFrame:

    #Load the data via the name.
    data = pd.read_csv('Traffic_Crashes.csv')

    # Cleaning data via dropping columns will null values > 60% and saving it.

    # checking null values
    sorted_na = data.isna().sum()/data.shape[0]
    to_drop = sorted_na[sorted_na>0.6].sort_values(ascending=False).index
    data.drop(to_drop, inplace=True, axis=1)

    # Drop the record ID (no significant information), crash date (Duplicate information, not more significant than Crash_hour / Crash_Day_of_Week / Crash_Month),
    # and drop index columns.
    data.drop(columns=["CRASH_RECORD_ID", "CRASH_DATE"], inplace=True)
    data.drop(columns=data.columns[data.columns.str.contains('unnamed',case=False)], inplace=True)

    print("Done cleaning!")
    return data

data = loadData()

# data.to_csv("Traffic_Crashes_Clean.csv")

np_data = data.to_numpy()

# print(type(np_data))
print(data.nunique())
print(data.iloc[0])
exit(2)

clusters = clique.run_clique(np_data, 3, 0.1)

print(clusters)

# print(data.shape)


# print(data.iloc[0])

# clusters = clique.run_clique(data, 16, 0.2)

# print(clusters)