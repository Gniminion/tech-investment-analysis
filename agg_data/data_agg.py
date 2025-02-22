
import pandas as pd

# ***** HOW WE AGGREGATED THE DATA WE NEEDED FOR VISUALISATION ****** # 

df_comp = pd.read_csv("cleaned_data_v2/cleaned_companies.csv") # remove path names and change to og data name once were done
df_di = pd.read_csv("cleaned_data_v2/cleaned_dealInvestor.csv")
df_deals = pd.read_csv("cleaned_data_v2/cleaned_deals.csv") 
df_invs = pd.read_csv("cleaned_data_v2/cleaned_investor.csv")
df_eco = pd.read_csv("cleaned_data_v2/cleaned_ecosystem.csv")  

# ------------------------------------ DATA CLEANING ------------------------------------ 
# add any code we had to make our cleaned datas here

# ------------------------------------ REGIONAL INSIGHTS ------------------------------------
# top investment categories
cats = df_deals.groupby('primaryTag').agg({'amount': 'sum'}).reset_index()
cats = cats.sort_values(by='amount', ascending=False)
cats = cats.head(10) # top 10
cats.to_csv('agg_data/top10cats.csv')

# average deal size
avg_deal = df_deals.groupby('ecosystemName')['amount'].mean().reset_index()
avg_deal.to_csv('agg_data/avg_deal.csv')

# total investment vol and category
cat_pref = df_deals.groupby(['ecosystemName', 'primaryTag'])['amount'].sum().reset_index()
cat_pref = cat_pref.sort_values(by='amount', ascending=False)
top25 = df_deals.groupby('primaryTag')['amount'].sum().sort_values(ascending=False)
top25 = top25.head(25).index
cat_pref = cat_pref[cat_pref['primaryTag'].isin(top25)]
cat_pref.to_csv('agg_data/cat_pref.csv')

# map visualisation by key headquarter regions
# longitude and latitude of major regions of interest
hq_loc = {
    'toronto': [43.6511, -79.3470],
    'montreal': [45.5019, -73.5674],
    'waterloo': [43.4643, -80.5204],
    'ottawa': [45.4235, -75.6979],
    'quebec': [46.8131, -71.2075],
    'vancouver': [49.2827, -123.1207],
    'calgary': [51.0447, -114.0719],
    'edmonton': [53.5461, -113.4937],
    'winnipeg': [49.8954, -97.1385],
}

df_loc = pd.DataFrame(hq_loc).T.reset_index()
df_loc.columns = ['headquarters', 'lat', 'lon']
df_loc.to_csv('hq_locs.csv')
# grabs a subset of deals with relevant headquarters, finds its total investment vol and top inv categories
hq_trends = df_deals.groupby(['headquarters', 'primaryTag'])['amount'].sum().reset_index()

# finds top 3 categories for each major headquarter location
top_cats = []
for hq in hq_trends['headquarters']:
        hq_data = hq_trends[hq_trends['headquarters'] == hq]
        top4 = hq_data.sort_values(by='amount', ascending=False).head(4)
        top_cats.append(top4)
top4_df = pd.concat(top_cats, ignore_index=True)

hq_trends = top4_df.merge(df_loc, on='headquarters', how='left')
num_deals = df_deals.groupby('headquarters').size().reset_index(name='num_deals') # counts number of deals for each hq
num_deals.to_csv('agg_data/num_deals.csv')
hq_trends = hq_trends.merge(num_deals, on='headquarters', how='left')
hq_trends = hq_trends.dropna(subset=['lat', 'lon']) # drops irrelevant headquarter locations
hq_trends = hq_trends.sort_values(by="amount", ascending=False)
hq_trends.to_csv("agg_data/hq_trends.csv")

