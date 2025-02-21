import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df_deals = pd.read_csv("cleaned_data_v2/cleaned_deals.csv") 

# creates a new dataframe of categories (sectors) with associated total investments and number of deals per year
cat_trends = df_deals.groupby(['primaryTag', 'year']).agg(
    totalInv=('amount', 'sum'), 
    numDeals=('amount', 'count')
).reset_index()

# calculates growth rate for both number of deals and investment amount
cat_trends['invGrowth'] = cat_trends.groupby('primaryTag')['totalInv'].pct_change()
cat_trends['dealGrowth'] = cat_trends.groupby('primaryTag')['numDeals'].pct_change()

cat_trends.replace([np.inf, -np.inf], np.nan, inplace=True)
cat_trends = cat_trends.dropna(subset=['invGrowth', 'dealGrowth']) # drop the first years where theres no growth rate indicators
cat_trends = cat_trends[cat_trends['year'].isin([2021, 2022, 2023, 2024])] # grab a subset of the recent four years
# target of high growth sectors is a sector with more than 50% investment growth and no decrease of deals from the prev year
cat_trends['hiGrowth'] = ((cat_trends['invGrowth'] > 0.5) & (cat_trends['dealGrowth'] >= 0)).astype(int)
print(cat_trends)
cat_trends['hiGrowth'].value_counts().plot(kind='bar', color=['blue', 'orange'], title='Class Distribution')
plt.show()

X = cat_trends[['totalInv', 'numDeals', 'invGrowth', 'dealGrowth']]
y = cat_trends['hiGrowth']

# text split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# test different models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(gamma="auto")
}
results = {}

# train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

best_model = max(results, key=results.get)
print(f"Best model: {best_model} with accuracy {results[best_model]:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0.5, 1.0])
plt.show()

# model - random forest classification
mod = RandomForestClassifier(n_estimators=100, random_state=42)
mod.fit(X_train, y_train)
pred = []
# uses model to predict whether each sector (category) will be high growth or low growth in 2025
for sector in cat_trends['primaryTag'].unique():
    
    last_row = cat_trends[cat_trends['primaryTag'] == sector].iloc[-1]
    prev_inv = last_row['totalInv']
    prev_deals = last_row['numDeals']
    prev_inv_growth = last_row['invGrowth']
    prev_deals_growth = last_row['dealGrowth']
    
    X_future = pd.DataFrame([[prev_inv, prev_deals, prev_inv_growth, prev_deals_growth]],
                            columns=['totalInv', 'numDeals', 'invGrowth', 'dealGrowth'])
    X_future = X_future.to_numpy()
    hi_or_lo = mod.predict(X_future)[0]
    pred.append((sector, 2025, hi_or_lo))

future_df = pd.DataFrame(pred, columns=['primaryTag', 'year', 'hiGrowth'])
#sectors that are likely to be high growth in 2025
hg_sectors = future_df[future_df['hiGrowth'] == 1]
print(hg_sectors)


# model - random forest regression, growth rate prediction
cat_trends = df_deals.groupby(['primaryTag', 'year']).agg(
    totalInv=('amount', 'sum'), 
    numDeals=('amount', 'count')
).reset_index()

# calculates growth rate for both number of deals and investment amount
cat_trends['invGrowth'] = cat_trends.groupby('primaryTag')['totalInv'].pct_change()
cat_trends['dealGrowth'] = cat_trends.groupby('primaryTag')['numDeals'].pct_change()
cat_trends.replace([np.inf, -np.inf], np.nan, inplace=True)
cat_trends = cat_trends.dropna(subset=['invGrowth', 'dealGrowth']) # drop the first years where theres no growth rate indicators

# lag features (previous year investments & deals)
cat_trends['prevInv'] = cat_trends.groupby('primaryTag')['totalInv'].shift(1)
cat_trends['prevDeals'] = cat_trends.groupby('primaryTag')['numDeals'].shift(1)

X = cat_trends[['year', 'prevInv', 'prevDeals']]
y = cat_trends['totalInv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mod = RandomForestRegressor(n_estimators=100, random_state=42)
mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(mae)
# 198173046.42679608
range = cat_trends['totalInv'].max() - cat_trends['totalInv'].min()
print(range)
# 2481721091

# predicts the investment amount of each sector for 2025
predictions = []
for sector in cat_trends['primaryTag'].unique():
    last = cat_trends[cat_trends['primaryTag'] == sector].iloc[-1] 
    prev_inv = last['totalInv']
    prev_deals = last['numDeals']
    
    X_future = pd.DataFrame([[2025, prev_inv, prev_deals]], columns=['year', 'prevInv', 'prevDeals'])
    future_pred = mod.predict(X_future)[0]
    predictions.append((sector, 2025, future_pred))
    prev_inv = future_pred

future_df = pd.DataFrame(predictions, columns=['primaryTag', 'year', 'predInv'])
# top 5 predicted sectors in 2025
print(future_df.sort_values(by='predInv', ascending=False).head(5))
