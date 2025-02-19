import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df_comp = pd.read_csv("cleaned_data_v2/cleaned_companies.csv")
df_di = pd.read_csv("cleaned_data_v2/cleaned_dealInvestor.csv")
df_deals = pd.read_csv("cleaned_data_v2/cleaned_deals.csv") 
df_invs = pd.read_csv("cleaned_data_v2/cleaned_deals.csv")
df_eco = pd.read_csv("cleaned_data_v2/cleaned_ecosystem.csv") 


# df_deals.to_csv("cleaned_deals.csv", index=False)
# df_invs.to_csv("cleaned_investors.csv", index=False)
# df_comp.to_csv("cleaned_companies.csv", index=False)


# creates a new dataframe of categories (sectors) with associated total investments and number of deals per year
cat_trends = df_deals.groupby(['primaryTag', 'year']).agg(
    totalInv=('amount', 'sum'), 
    numDeals=('amount', 'count')
).reset_index()

# calculates growth rate for both number of deals and investment amount
cat_trends['invGrowth'] = cat_trends.groupby('primaryTag')['totalInv'].pct_change() * 100
cat_trends['dealGrowth'] = cat_trends.groupby('primaryTag')['numDeals'].pct_change() * 100
cat_trends = cat_trends.dropna(subset=['invGrowth']) # drop the first years where theres no growth rate indicators
print(cat_trends)

# lag features (previous year investments & deals)
cat_trends['prevInv'] = cat_trends.groupby('primaryTag')['totalInv'].shift(1)
cat_trends['prevDeals'] = cat_trends.groupby('primaryTag')['numDeals'].shift(1)

X = cat_trends[['year', 'prevInv', 'prevDeals']]
y = cat_trends['totalInv']

# train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mod = RandomForestRegressor(n_estimators=100, random_state=42)
mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(mae)

future = [2025, 2026, 2027]
predictions = []
for sector in cat_trends['primaryTag'].unique():
    last = cat_trends[cat_trends['primaryTag'] == sector].iloc[-1] 
    prev_inv = last['totalInv']
    prev_deals = last['numDeals']
    
    for year in future:
        X_future = [[year, prev_inv, prev_deals]]
        future_pred = mod.predict(X_future)[0]
        predictions.append((sector, year, future_pred))
        prev_inv = future_pred

future_df = pd.DataFrame(predictions, columns=['primaryTag', 'year', 'predInv'])
print(future_df.sort_values(by='predInv', ascending=False).head(5))
