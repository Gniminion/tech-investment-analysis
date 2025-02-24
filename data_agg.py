
import pandas as pd
import numpy as np

# ***** HOW WE AGGREGATED THE DATA WE NEEDED FOR VISUALISATION AND MODELING (USING PYTHON PANDAS AND NUMPY) ***** # 
# ***** ALL THE AGGREGATED DATA SHOULD BE IN THE dashboard/agg_data FOLDER, EXCEPT TRAINING DATA FOR PREDICTIVE MODEL ***** #

# these are the dataframe names for each of the RunQL dataset
df_comp # raw data for companies
df_di  #raw data for deal investors
df_deals  #raw data for deals
df_invs  #raw data for investors
df_eco #raw data for ecosystems

# ------------------------------------ DATA CLEANING ------------------------------------ 
# INITIAL CLEANING (STANDARDISING ALL DATASET FORMATS)
# DEALS cleaning:
# drop dupes
df_deals.drop_duplicates(subset=['id', 'companyId'], inplace=True)
# decided to drop secondary ecosystem as there are a lot of missing values so not too helpful for analysis
df_deals.drop('ecosystemSecondary', axis=1, inplace=True) 
# fill missing values with unknown
df_deals['headquarters'].fillna('unknown', inplace=True)
df_deals['leadInvestors'].fillna('unknown', inplace=True)
df_deals['investors'].fillna('unknown', inplace=True)
# filling in the small amount of missing values in categories with mode category
df_deals['primaryTag'].fillna(df_deals['primaryTag'].mode()[0], inplace=True)
# converting to date time format
df_deals['date'] = pd.to_datetime(df_deals['date'], errors='coerce')
df_deals['year'] = pd.to_numeric(df_deals['year'], errors='coerce')
# replacing unknown series with unknown
df_deals['roundType'] = df_deals['roundType'].replace({'Series ?': 'unknown'})
# converting values to lower case (except yearQuarter formats)
for column in df_deals.select_dtypes(include=['object']).columns:
    if column != 'yearQuarter':
        df_deals[column] = df_deals[column].str.lower()
        
# COMPANIES CLEANING:
df_comp.dropna(subset=['latestRoundType'], inplace=True)
# dropping irrelevant columns for analysis
df_comp.drop(columns=['ecosystemSecondary'], inplace=True)
df_comp.drop(columns=['logoUrlCdn'], inplace=True)
df_comp.drop(columns=['dateAcqusition'], inplace=True)
df_comp.drop(columns=['acquiringCompany'], inplace=True)
df_comp.drop(columns=['ipoDate'], inplace=True)
df_comp.drop(columns=['peDate'], inplace=True)
df_comp['secondaryTag'].fillna("Unknown", inplace=True)
# replacing unknown series with unknown to standardise
df_comp['latestRoundType'] = df_comp['latestRoundType'].replace({'series ?': 'unknown'})
# date time formatting
df_comp['dateFounded'] = pd.to_datetime(df_comp['dateFounded'], errors='coerce')
df_comp['latestRoundDate'] = pd.to_datetime(df_comp['latestRoundDate'], errors='coerce')
# lowercasing
text_cols = ['companyName', 'ecosystemName', 'primaryTag', 'secondaryTag', 'latestRoundType',]
for column in df_comp.select_dtypes(include=['object']).columns:
    if column in text_cols:
        df_deals[column] = df_deals[column].str.lower()

# INVESTORS CLEANING
df_invs['investorType'].fillna("Unknown", inplace=True)
df_invs['city'].fillna("Unknown", inplace=True)
df_invs['country'].fillna("Unknown", inplace=True)
df_invs['sectors'].fillna("Unknown", inplace=True)
df_invs['stages'].fillna("Unknown", inplace=True)
df_invs['logoURL'].fillna("Unknown", inplace=True)
# drop irrelevant columns
df_invs.drop(columns=['logoURL'], inplace=True)
# to lower
text_cols = ['investorName', 'investorType', 'city', 'country', 'sectors', 'stages']
for column in df_invs.select_dtypes(include=['object']).columns:
    if column in text_cols:
        df_deals[column] = df_deals[column].str.lower()
        
# DEAL INVESTORS CLEANING:
df_di.dropna(subset=['headquarters'], inplace=True)
df_di.drop(columns=['ecosystemSecondary'], inplace=True)
df_di['roundType'].fillna("Unknown", inplace=True)
df_di['investorCountry'].fillna("Unknown", inplace=True)
text_cols = ['companyName', 'headquarters', 'investorName', 'ecosystemName', 'roundType', 'investorCountry']
for column in df_di.select_dtypes(include=['object']).columns:
    if column in text_cols:
        df_di[column] = df_di[column].str.lower()
# unknown series
df_di['roundType'] = df_di['roundType'].replace({'series ?': 'unknown'})
# date time and data type formatting
df_di['date'] = pd.to_datetime(df_di['date'], errors='coerce') 
df_di['year'] = pd.to_numeric(df_di['year'], errors='coerce')  
df_di['leadInvestorFlag'] = df_di['leadInvestorFlag'].astype(int)
        
# ECOSYSTEM CLEANING (to lowercase):
for column in df_eco.select_dtypes(include=['object']).columns:
      df_eco[column] = df_eco[column].str.lower()

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
    if ('manu' in cat): # some other mappings identified after performing eda
        return 'manufacturing'
    if ('software' in cat):
        return 'software development'
    if ('health' in cat):
        return 'healthtech'
    if ('transportation' in cat):
        return 'transportation'
    if ('metaverse' in cat):
        return 'ar'
    if ('paas' in cat):
        return 'platform'
    if ('geospatial' in cat):
        return 'geotech'
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

# renaming the stages for better grouping
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

#------------------------------------ INVESTMENTS OVER TIME ------------------------------------
# sum the investment amount
temp_deals = df_deals[(df_deals['year'] >= 2019) & (df_deals['year'] <= 2024)]
inv_trends = temp_deals.groupby('year')['amount'].sum().reset_index()
inv_trends.to_csv("agg_data/inv_trends.csv")

# number of deals per year
deal_vol = temp_deals.groupby('year')['id'].count().reset_index()
deal_vol.to_csv("agg_data/deal_vol.csv")

# deal size categories
def categorize_deal(amount):
    if amount < 100000:
        return '<$100K'
    elif 1000000 <= amount < 5000000:
        return '$1M-$5M'
    elif amount >= 100000000:
        return '$100M+'
    else:
        return 'Other'

temp_deals['ds_category'] = temp_deals['amount'].apply(categorize_deal)
ds_trends = temp_deals.groupby(['year', 'ds_category'])['amount'].sum().reset_index()
ds_trends.to_csv("agg_data/ds_trends.csv")

# merge deals with ecosystems
df_deals_regs = temp_deals.merge(df_eco, on='ecosystemName', how='left')
reg_trends = df_deals_regs.groupby(['year', 'province'])['amount'].sum().reset_index()
reg_trends.to_csv("agg_data/reg_trends.csv")

# data cleaning
valid_deals = df_deals_regs [(df_deals_regs ['roundType'] != 'Unknown') & (~df_deals_regs['province'].isna())]
valid_deals['date'] = pd.to_datetime(valid_deals['date'])
valid_deals['year'] = valid_deals['date'].dt.year
valid_deals =valid_deals[['year', 'amount', 'province', 'roundType']]
valid_deals.to_csv("agg_data/valid_deals.csv")

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

#------------------------------------ INVESTOR BEHAVIOUR ------------------------------------
# categorize US & Canada, group others as Other International
country_map = {
    "usa": "USA",
    "canada": "Canada"
}
df_invs["country_grouped"] = df_invs["country"].map(country_map)
df_invs["country_grouped"] = df_invs["country_grouped"].fillna("Other International")

# count number of investment firms by country group
invs_counts = df_invs.groupby("country_grouped").size().reset_index(name="count")
invs_counts.to_csv("agg_data/invs_counts.csv")

# heatmap pivioting
invs_counts = df_invs.explode("stages").groupby(["country", "stages"]).size().reset_index(name="count") # splitting the stages for countries
invs_counts = invs_counts[invs_counts["count"] >= 5] # filter out small count investors
invs_pivot = invs_counts.pivot(index="country", columns="stages", values="count").fillna(0)
invs_pivot.to_csv("agg_data/invs_pivot.csv")

# deal size by stage
ds_by_stage = df_deals.groupby("roundType")["amount"].mean().reset_index(name="avg_deal_size")
ds_by_stage.to_csv("agg_data/ds_by_stage.csv")

# identify leading investors per stage
lead_invs_stage = df_di.groupby(["roundType", "investorName"]).agg(
    lead_count=("leadInvestorFlag", "sum"),
    total_deals=("dealId", "count")
).reset_index()

# sort and select top investors per stage
top_lead_invs = lead_invs_stage.sort_values(
    by=["roundType", "lead_count", "total_deals"], ascending=[True, False, False]
).groupby("roundType").head(3) # top 3

# influence on funding success
fund_data = df_deals.merge(df_di, on="id", how="left")
investor_funding = fund_data.groupby("investorName").agg(
    total_funding=("amount", "sum"), # success metrics
    avg_funding=("amount", "mean"),
    total_deals=("id", "count")
).reset_index()

# filter for leading investors
top_invs = investor_funding[investor_funding["investorName"].isin(top_lead_invs["investorName"])]
top_invs.to_csv('agg_data/top_invs.csv')

# ------------------------------------ REGIONAL INSIGHTS ------------------------------------
# top investment categories
cats = df_deals.groupby('primaryTag').agg({'amount': 'sum'}).reset_index()
cats = cats.sort_values(by='amount', ascending=False)
cats = cats.head(10) # top 10
cats.to_csv('agg_data/top10_cats.csv')

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

# ------------------------------------ PREDICTION MODEL TRAINING DATA ------------------------------------
# creates a new dataframe of categories (sectors) with associated total investments and number of deals per year
cat_trends = df_deals.groupby(['primaryTag', 'year']).agg(
    totalInv=('amount', 'sum'), 
    numDeals=('amount', 'count')
).reset_index()

# calculates growth rate for both number of deals and investment amount
cat_trends['invGrowth'] = cat_trends.groupby('primaryTag')['totalInv'].pct_change()
cat_trends['dealGrowth'] = cat_trends.groupby('primaryTag')['numDeals'].pct_change()

cat_trends.replace([np.inf, -np.inf], np.nan, inplace=True) # filters out invalid inf values
cat_trends = cat_trends.dropna(subset=['invGrowth', 'dealGrowth']) # drop the first years where theres no growth rate indicators

cat_trends.to_csv("training_data.csv")  
