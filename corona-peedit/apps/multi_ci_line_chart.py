import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output

from app import app

external_stylesheets = [
    'assets/bWLwgP.css',
    'assets/index.css',
]

country_series = pd.read_csv('res/IDS_CSV/IDScountry-series.csv')
country_series.drop(['Unnamed: 3'], axis=1, inplace=True)

country = pd.read_csv('res/IDS_CSV/IDSCountry.csv')
country.drop(['Unnamed: 31'], axis=1, inplace=True)

idsdata = pd.read_csv('res/IDS_CSV/IDSData.csv')
idsdata.drop(['Unnamed: 61'], axis=1, inplace=True)

idsdata_ = idsdata.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name',
                                 'Indicator Code'], var_name='Year', value_name='Value').fillna(0)
idsdata_['Year'] = idsdata_['Year'].apply(lambda x: int(x))
idsdata_['CI Name'] = idsdata_['Country Name'] + \
    idsdata_['Country Name'].apply(lambda x: ', ')+idsdata_['Indicator Name']

footnote = pd.read_csv('res/IDS_CSV/IDSfootnote.csv')
footnote.drop(['Unnamed: 4'], axis=1, inplace=True)

series = pd.read_csv('res/IDS_CSV/IDSSeries.csv')
series.drop(['Unnamed: 20'], axis=1, inplace=True)

series_time = pd.read_csv('res/IDS_CSV/IDSseries-time.csv')
series_time.drop(['Unnamed: 3'], axis=1, inplace=True)

ind_str = """
Net financial flows, others
Net financial flows, bilateral (NFL, current US$)"""
indicators = [x.strip() for x in ind_str.split('\n') if x.strip() != '']
indicators
codes = list()
for x in indicators:
    codes += series[series['Indicator Name']
                    .str.find(x) > -1]['Series Code'].to_list()
codes

countries = idsdata[['Country Code', 'Country Name']]
countries = countries.drop_duplicates()
countries_codes_names = [(x, y) for x, y in zip(
    countries['Country Code'].to_list(), countries['Country Name'].to_list())]
indicator_codes_names = [(x, y) for x, y in zip(
    series['Series Code'].to_list(), series['Indicator Name'].to_list())]

all_indicators = pd.unique(idsdata['Indicator Name'])

country_codes_str = """
AFG
IND
PAK
CHN"""
country_codes = [x.strip()
                 for x in country_codes_str.split('\n') if x.strip() != '']

indicator_codes = codes

idsdata[(idsdata['Country Code'].isin(country_codes)) &
        (idsdata['Indicator Code'].isin(indicator_codes))]


dropdown_options = [{'label': y, 'value': x} for x, y in countries_codes_names]

piechart_options = [
    {'label': 'Modes Bilateral/Multilateral/Others', 'value': 'mode'},
    # ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD']
    {'label': 'Sources IMF/IDA/IBRD/RDB', 'value': 'source'},
    {'label': 'Type concessional/nonconcessional', 'value': 'type'}]
# ['DT.NFL.IMFC.CD','DT.NFL.IMFN.CD','DT.NFL.MIBR.CD','DT.NFL.MIDA.CD','DT.NFL.RDBC.CD','DT.NFL.RDBN.CD']


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


explaination = """
# MCI
## grants
India and Pakistan
Pakistan observed a sudden decrease in grants received by it after 2011, when Osama was encountered.
In 2016, due to America's pro-America policy in Trump rule, grants saw a decrease in all under developed countries
This is especially profound in Arab Islamic Countries.
"""
side_elements = html.Div(className='column', children=[
    html.Div(id='mci-table', className='table-pane', children=[
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) & (idsdata_[
                       'Indicator Code'].isin(['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'])) & (idsdata_['Year'] <= 2018)], max_rows=np.inf)

    ]),
    html.Div(className='text-pane', children=[
        dcc.Markdown(explaination)
        # generate_table(df, np.inf),
    ]),

])

world_map = dcc.Graph(id='mci-world-map', figure=px.choropleth(
    locations=['IND', 'CHN']))


indicator_line_chart = dcc.Graph(id='mci-ind-line-chart', figure=px.line(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) & (idsdata_['Indicator Code'].isin(
    ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'])) & (idsdata_['Year'] <= 2018) & (idsdata_['Year'] >= 2008) & (idsdata_['Year'] <= 2018)], x="Year", y="Value", color="Indicator Name"))


main_space = html.Div(className="mid-pane", children=[
    world_map,

    html.Label('Countries'),
    dcc.Dropdown(
        id='mci-countries',
        options=dropdown_options,
        value=['IND', 'CHN', 'BRA', 'RUS', 'ZAF', 'PAK'],
        multi=True
    ),

    html.Label('Indicator'),
    dcc.Dropdown(
        id='mci-indicators',
        options=[{'label': y, 'value': x} for x, y in indicator_codes_names],
        value=['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'],
        multi=True
    ),

    html.Label('Time window'),
    dcc.RangeSlider(
        id='mci-time-window-slider',
        min=1970,
        max=2018,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i)
               for i in range(1970, 2018, 10)},

        step=1,
        value=[2008, 2018]
    ),
    html.Div(id='mci-Range', children=[
        html.Label('2000 - 2000'),
    ]),


    indicator_line_chart,

])


@app.callback(Output('mci-world-map', 'figure'), [Input('mci-countries', 'value')])
def update_world_map(selected_value):
    return px.choropleth(locations=selected_value)


@app.callback(Output('mci-ind-line-chart', 'figure'), [Input('mci-countries', 'value'), Input('mci-indicators', 'value'), Input('mci-time-window-slider', 'value')])
def update_ind_line_chart(country_vals, ind_val, time_val):
    return px.line(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'].isin(ind_val)) & (idsdata_['Year'] <= 2018) & (idsdata_['Year'] >= time_val[0]) & (idsdata_['Year'] <= time_val[1])], x="Year", y="Value", color="CI Name")


@app.callback(Output('mci-table', 'children'), [Input('mci-countries', 'value'), Input('mci-indicators', 'value'), Input('mci-time-window-slider', 'value')])
def update_table(country_vals, ind_val, time_val):
    return [
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'].isin(ind_val)) & (
            idsdata_['Year'] <= 2018) & (idsdata_['Year'] >= time_val[0]) & (idsdata_['Year'] <= time_val[1])], max_rows=np.inf)
    ]


@app.callback(Output('mci-Range', 'children'), [Input('mci-time-window-slider', 'value')])
def update_range_display(time_val):
    return [html.Label('{} - {}'.format(time_val[0], time_val[1]))]


layout = [html.Div(children=[
    html.Div(className='row', children=[
        dcc.Link('Explore Indicators and Countries', href='/',
                 style={"margin-left": "15px"}),
        dcc.Link('Explore Countries', href='/mlc',
                 style={"margin-left": "15px"}),
        dcc.Link('Explore Indicators', href='/mli',
                 style={"margin-left": "15px"}),
        dcc.Link('Country Wise Breakup of Debt',
                 href='/pie-chart', style={"margin-left": "15px"}),
        dcc.Link('Debt Breakup Comparison', href='/stacked-bar',
                 style={"margin-left": "15px"}),
    ]),
    html.Div(className="row", children=[
        html.Div(className="left-panel", children=[
            side_elements
        ]),
        main_space,
    ])
])]
