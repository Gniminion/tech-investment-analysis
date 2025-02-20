import pandas as pd
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc, Input, Output
import plotly.express as px

df_comp = pd.read_csv("cleaned_data_v2/cleaned_companies.csv")
df_di = pd.read_csv("cleaned_data_v2/cleaned_dealInvestor.csv")
df_deals = pd.read_csv("cleaned_data_v2/cleaned_deals.csv") 
df_invs = pd.read_csv("cleaned_data_v2/cleaned_deals.csv")
df_eco = pd.read_csv("cleaned_data_v2/cleaned_ecosystem.csv")   

#------------------------------------ INVESTMENTS OVER TIME ------------------------------------
# Group by year and sum the investment amount
investment_trends = df_deals.groupby('year')['amount'].sum().reset_index()
fig_investment = px.line(investment_trends, x='year', y='amount', markers=True,
                         title='Total Investment in Canadian Startups (2019-2024)')

# Count number of deals per year
deal_volume = df_deals.groupby('year')['id'].count().reset_index()
deal_volume.rename(columns={'id': 'num_deals'}, inplace=True)
fig_deals = px.bar(deal_volume, x='year', y='num_deals',
                   title='Number of Investment Deals Per Year (2019-2024)', color='num_deals')

# Define deal size categories
def categorize_deal(amount):
    if amount < 100000:
        return '<$100K'
    elif 1000000 <= amount < 5000000:
        return '$1M-$5M'
    elif amount >= 100000000:
        return '$100M+'
    else:
        return 'Other'

df_deals['deal_size_category'] = df_deals['amount'].apply(categorize_deal)

deal_size_trends = df_deals.groupby(['year', 'deal_size_category'])['amount'].sum().reset_index()
deal_size_pivot = deal_size_trends.pivot(index='year', columns='deal_size_category', values='amount')
fig_deal_size = px.bar(deal_size_trends, x='year', y='amount', color='deal_size_category',
                        title='Investment Distribution by Deal Size (2019-2024)', barmode='stack')

# Merge deals with ecosystems
df_deals_regions = df_deals.merge(df_eco, on='ecosystemName', how='left')
region_trends = df_deals_regions.groupby(['year', 'province'])['amount'].sum().reset_index()
fig_region = px.line(region_trends, x='year', y='amount', color='province', markers=True,
                      title='Investment Trends by Region (2019-2024)')

# Data cleaning
valid_deals = df_deals_regions [(df_deals_regions ['roundType'] != 'Unknown') & (~df_deals_regions['province'].isna())]
valid_deals['date'] = pd.to_datetime(valid_deals['date'])
valid_deals['year'] = valid_deals['date'].dt.year

# Get dropdown options
year_options = sorted(valid_deals['year'].unique())
province_options = sorted(valid_deals['province'].dropna().unique())
stage_options = sorted(valid_deals['roundType'].unique())
stage_options = [stage for stage in stage_options if stage.lower() != 'unknown']

# ------------------------------------ REGIONAL INSIGHTS ------------------------------------
# top investment categories
cats = df_deals.groupby('primaryTag').agg({'amount': 'sum'}).reset_index()
cats = cats.sort_values(by='amount', ascending=False)
cats = cats.head(10) # top 10
top10_cats = px.bar(cats,
             x='amount', y='primaryTag', 
             color = 'primaryTag',
             title='Top 10 Investment Categories by Amount',
             labels={'primaryTag': 'Investment Category', 'amount': 'Total Investment ($)'},
             color_discrete_sequence=px.colors.qualitative.Prism,)
top10_cats.update_layout(showlegend=False)  

# average deal size
avg_deal = df_deals.groupby('ecosystemName')['amount'].mean().reset_index()
avg_reg = px.bar(avg_deal, x='ecosystemName', y='amount', 
             labels={'ecosystemName':'Region', 'amount':'Average Investment Volume'},
             title='Average Deal Size by Region',)
avg_reg.update_xaxes(categoryorder='total descending')

# total investment vol and category
cat_pref = df_deals.groupby(['ecosystemName', 'primaryTag'])['amount'].sum().reset_index()
cat_pref = cat_pref.sort_values(by='amount', ascending=False)
top20 = df_deals.groupby('primaryTag')['amount'].sum().sort_values(ascending=False)
top20 = top20.head(25).index
cat_pref = cat_pref[cat_pref['primaryTag'].isin(top20)]

inv_reg = px.bar(cat_pref, x='ecosystemName', y='amount', color='primaryTag',
             labels={'ecosystemName':'Region', 'amount':'Investment Volume', 'primaryTag':'Categories (Top 25)'},
             title='Category Preferences and Investment Volume by Region', barmode='stack',
             color_discrete_sequence=px.colors.qualitative.Dark24,)
inv_reg.update_xaxes(categoryorder='total descending')

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
hq_trends = hq_trends.merge(num_deals, on='headquarters', how='left')
hq_trends = hq_trends.dropna(subset=['lat', 'lon']) # drops irrelevant headquarter locations
hq_trends = hq_trends.sort_values(by="amount", ascending=False)

map_hq = px.scatter_map(
    hq_trends,
    lat='lat',
    lon='lon',
    size='amount',  # bubble size represents investment volume
    color="primaryTag", # color represents top category
    hover_name='headquarters',
    hover_data={'primaryTag': True, "amount": True, 'num_deals': True, "lat": False, "lon":False},
    labels={'amount':'Investment Volume by Category', 'primaryTag':'Top Investment Categories',
            'num_deals': 'Total Number of Deals for Headquarter'},
    color_discrete_sequence=px.colors.qualitative.Prism,
    size_max=60, 
    zoom=3
)

map_hq.update_layout(
    mapbox_style='carto-positron',
    mapbox_center={"lat": 56, "lon": -106}, 
)

# Putting everything into a dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Tech Investments Dashboard"),
        dcc.Tabs([
            dcc.Tab(label='Investment Trends Over Time', children=[
                html.Div([
                    html.H3("Investment Trends Over Time", style={'margin-top': '50px', 'margin-left': '50px'}),
                    html.P([
                        "Analyze total tech investment per year (2019-2024), identifying major shifts.",
                        html.Br(),
                        "Examine deal volume and funding size trends over time.",
                        html.Br(),
                        "Investigate how investment has changed across different deal sizes."
                    ], style={'margin-left': '50px'}),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Year"),
                            dcc.Dropdown(
                                id='year-dropdown', 
                                options=[{'label': str(y), 'value': y} for y in year_options],
                                multi=True, value=[year_options[-1]]
                            ),
                            
                            html.Label("Select Province", style={'margin-top': '20px'}),
                            dcc.Dropdown(
                                id='province-dropdown', 
                                options=[{'label': p, 'value': p} for p in province_options],
                                multi=True, value=province_options
                            ),
                            
                            html.Label("Select Funding Stage", style={'margin-top': '20px'}),
                            dcc.Dropdown(
                                id='stage-dropdown', 
                                options=[{'label': s, 'value': s} for s in stage_options],
                                multi=True, value=stage_options
                            )
                        ], width=3, style={'padding': '20px'}),
                        dbc.Col([
                            dcc.Graph(
                                id='investment-bar-chart', 
                                style={'width': '100%', 'height': '600px'}
                            )
                        ], width=9)  # Graph takes more space
                    ], justify='start', style={'margin-left': '80px', 'margin-right': '80px'}),
                ]),
                dcc.Tab(label='Total Investment', children=[dcc.Graph(figure=fig_investment,style={'width': '80%', 'height': '500px', 'margin': 'auto'})]),
                dcc.Tab(label='Deal Volume', children=[dcc.Graph(figure=fig_deals,style={'width': '80%', 'height': '500px', 'margin': 'auto'})]),
                dcc.Tab(label='Deal Size Distribution', children=[dcc.Graph(figure=fig_deal_size,style={'width': '80%', 'height': '500px', 'margin': 'auto'})]),
                dcc.Tab(label='Regional Trends', children=[dcc.Graph(figure=fig_region,style={'width': '80%', 'height': '500px', 'margin': 'auto'})])
            ]),
            dcc.Tab(label=' Funding Stages Analysis', children=[
            
            ]),
            dcc.Tab(label='Investor Demographics & Behavior', children=[
            
            ]),
            dcc.Tab(label='Regional Insights', children=[
                html.Div([
                    html.H3("Sectoral & Regional Insights",style={'margin-top': '50px', 'margin-left': '50px'}),
                    html.P(["Identify the top investment categories nationally (e.g., SaaS, FinTech, HealthTech, AI, Blockchain).",
                           html.Br(),
                           "Compare investment trends across key Canadian regions (Toronto, Vancouver, Montreal, Calgary, Waterloo, etc.).",
                           html.Br(),
                           "Examine regional differences in investment volume, deal sizes, and category preferences."],
                    style={'margin-left': '50px'}),
                    dcc.Graph(figure=top10_cats,
                            style={'width': '90%', 'height': '700px', 'margin': 'auto'})
                ]),
                html.Div([
                    html.H3("Investment Trends by Major Headquarter Locations",style={'margin-top': '50px', 'margin-left': '50px'}),
                    html.P("Hover over the headquarter to see information such as total investment funding for each category, and number of deals!",
                        style={'margin-left': '50px'}),
                    dcc.Graph(
                        figure=map_hq,
                        style={'width': '90%', 'height': '700px'}
                    )
                ]),
                html.Div([
                    dcc.Graph(figure=inv_reg,
                            style={'width': '80%', 'height': '500px', 'margin': 'auto'},
                    ),
                    dcc.Graph(figure=avg_reg,
                            style={'width': '80%', 'height': '500px', 'margin': 'auto'},
                    )
                ])
            ]),
        ])
])

# Callbacks
@app.callback(
    Output('investment-bar-chart', 'figure'),
    Input('year-dropdown', 'value'),
    Input('province-dropdown', 'value'),
    Input('stage-dropdown', 'value')
)
def update_graph(selected_years, selected_provinces, selected_stages):
    filtered_df = valid_deals[
        (valid_deals['year'].isin(selected_years)) &
        (valid_deals['province'].isin(selected_provinces)) &
        (valid_deals['roundType'].isin(selected_stages))
    ]
    
    summary = filtered_df.groupby(['province'])['amount'].sum().reset_index()
    fig = px.bar(summary, x='province', y='amount', title='Investment Amount by Province',
                 labels={'province': 'Province', 'amount': 'Total Investment ($)'},
                 color='province', color_discrete_sequence=px.colors.qualitative.Prism,)
    fig.update_xaxes(categoryorder='total descending')
    return fig

if __name__ == '__main__':
    print("App is starting")
    app.run_server(debug=True)
