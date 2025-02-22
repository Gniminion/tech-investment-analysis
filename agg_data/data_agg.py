
import pandas as pd

# ***** HOW WE AGGREGATED THE DATA WE NEEDED FOR VISUALISATION AND MODELING ****** # 

df_comp = pd.read_csv("cleaned_data_v2/cleaned_companies.csv") # remove path names and change to og data name once were done
df_di = pd.read_csv("cleaned_data_v2/cleaned_dealInvestor.csv")
df_deals = pd.read_csv("cleaned_data_v2/cleaned_deals.csv") 
df_invs = pd.read_csv("cleaned_data_v2/cleaned_investor.csv")
df_eco = pd.read_csv("cleaned_data_v2/cleaned_ecosystem.csv")  

# ------------------------------------ DATA CLEANING ------------------------------------ 
# add any code we had to make our cleaned datas here

# identify all possible categories
print(df_deals['primaryTag'].unique())

# clean categories
fix = { # with identification help from genAI
    "blochchain": "blockchain",
    "heatlhtech": "healthtech",
    "contructiontech": "constructiontech",
    "artificial intelligences": "ai",
    "electronic health record (ehr)": "healthtech",
    "pharmaceutical manufacturing": "pharmaceuticals",
    "medical device": "medtech",
    "biotechnology": "biotech",
    "machine learning": "ai",
    "communications infrastructure": "telecommunications",
    "real estate": "propertytech",
    "renewable energy": "cleantech",
    "r&d": "research",
    "crm": "saas",
    "insurance software": "insurtech",
    "environmental services": "cleantech",
    "energy efficiency": "greentech",
    "womens health": "healthtech",
    "biotechnology research": "biotech",
    'esphera synbio': "biotech",
    'retail recyclable materials &':'cleantech',
    'data analytics':'analytics',
    'online audio and video media':'entertainmenttech',
    'technology':'information technology',
    '3d printing':'3dtech'
}

def clean_cat(cat):
    cat = cat.split(',')[0]
    cat = cat.replace('-', ' ')
    cat = cat.strip() 
    cat = fix.get(cat, cat) 
    if ('manu' in cat):
        return 'manufacturing'
    if ('software' in cat):
        return 'software development'
    if ('health' in cat):
        return 'healthtech'
    if ('transportation' in cat):
        return 'transportation'
    return cat

# make categories clean for all the relevant dfs
df_deals['primaryTag'] = df_deals['primaryTag'].apply(clean_cat)
df_comp['primaryTag'] = df_comp['primaryTag'].apply(clean_cat)

# identify all possible ecosystems
print(df_deals['ecosystemName'].unique())
# put waterloo into waterloo region
def clean_eco(eco):
  if (eco == 'waterloo'):
    return 'waterloo region'
  else:
    return eco
df_deals['ecosystemName'] = df_deals['ecosystemName'].apply(clean_eco)
df_comp['ecosystemName'] = df_comp['ecosystemName'].apply(clean_eco)
df_di['ecosystemName'] = df_di['ecosystemName'].apply(clean_eco)

# identify all possible headquarters
print(df_deals['headquarters'].unique())

# clean headquarters data
fix = { 
    "montréal": "montreal",
    "québec": "quebec",
    "vanouver": "vancouver",
    "west vancouver": "vancouver",
    "kitchener/toronto": "toronto",
    "kitchener-waterloo" : "waterloo",
    "kitchener": "waterloo" # include kitchener in the waterloo data
}
def clean_hq(hq):
    hq = hq.split('&')[0] 
    hq = hq.split(',')[0]
    hq = hq.strip()
    hq = fix.get(hq, hq) 
    return hq

# make headquarters clean for all the relevant dfs
df_deals['headquarters'] = df_deals['headquarters'].apply(clean_hq)
df_di['headquarters'] = df_di['headquarters'].apply(clean_hq)

def clean_stage(stage):
    mapping = {
        "pre seed": "Pre-Seed",
        "seed": "Seed",
        "series a": "Series A",
        "series b": "Series B",
        "series c": "Series C",
        "series d": "Series D+",
        "series e": "Series D+",
        "series f": "Series D+",
        "growth": "Growth",
        "ipo": "IPO"
    }
    return mapping.get(stage.lower(), "Other")
df_deals["roundType"] = df_deals["roundType"].apply(clean_stage)

#------------------------------------ FUNDING STAGES  ------------------------------------

# count the number of deals at each funding stage
stage_counts = df_deals['roundType'].value_counts(normalize=True) * 100  # convert to percentage
stage_counts = stage_counts.reset_index()  # reset index to turn it into a DataFramie
stage_counts.columns = ['Funding Stage', 'Proportion']
stage_counts.to_csv("agg_data/stage_counts.csv")

# deal sizes
deal_size = df_deals.groupby(["year", "roundType"])['amount'].mean().reset_index()
deal_size.to_csv("agg_data/deal_size.csv")

# trends in number and size of deals over years
deal_trends = df_deals.groupby(["year", "roundType"]).agg({"id": "count", "amount": "sum"}).reset_index()
deal_trends.to_csv("agg_data/deal_trends.csv")

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
df_loc.to_csv('agg_data/hq_locs.csv')
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
