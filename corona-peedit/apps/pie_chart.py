import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    {'label': 'Sources IMF/IDA/IBRD/RDB', 'value': 'source'}]
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
# Pie chart
India mode of Debt
India has changed its policy more towards bilateral commitments than multilateral in the Modi Government.
"""

side_elements = html.Div(className='column', children=[
    html.Div(id='pie-table', className='table-pane', children=[
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) & (idsdata_[
                       'Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] <= 2018)], max_rows=np.inf)

    ]),
    html.Div(className='text-pane', children=[
        dcc.Markdown(explaination)
        # generate_table(df, np.inf),
    ]),

])

world_map = dcc.Graph(id='pie-world-map', figure=px.choropleth(
    locations=['IND']))


bar_graph = dcc.Graph(id='pie-bar-graph', figure=go.Figure(data=[
    go.Bar(name='Net financial flows, bilateral (NFL, current US$)', x=['IND'], y=idsdata_[(idsdata_[
           'Country Code'].isin(['IND'])) & (idsdata_['Indicator Code'] == 'DT.NFL.BLAT.CD') & (idsdata_['Year'] == 2008)]['Value'].to_list()),
    go.Bar(name='Net financial flows, multilateral (NFL, current US$)', x=['IND'], y=idsdata_[(idsdata_[
           'Country Code'].isin(['IND'])) & (idsdata_['Indicator Code'] == 'DT.NFL.MLAT.CD') & (idsdata_['Year'] == 2008)]['Value'].to_list()),
    go.Bar(name='Net financial flows, others (NFL, current US$)', x=['IND'], y=idsdata_[(idsdata_[
           'Country Code'].isin(['IND'])) & (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] == 2008)]['Value'].to_list()),
]))

piechart = dcc.Graph(id='pie-pie-chart', figure=px.pie(idsdata_[(idsdata_['Country Code'] == "IND") & (idsdata_['Year'] == 2008) & (idsdata_['Indicator Code'].isin(
    ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD']))], values='Value', names='Indicator Name', title='Net financial flow from various agencies'))

main_space = html.Div(className="mid-pane", children=[
    world_map,

    html.Label('Countries'),
    dcc.Dropdown(
        id='pie-countries',
        options=dropdown_options,
        value='IND',
    ),

    html.Label('Feature'),
    dcc.Dropdown(
        id='pie-feature',
        options=piechart_options,
        value='mode',
    ),

    html.Label('Year'),
    dcc.Slider(
        id='pie-time',
        min=1970,
        max=2018,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i)
               for i in range(1970, 2018, 10)},
        value=2008,
    ),
    html.Div(id='pie-year_holder', children=[
        html.Label('2008'),
    ]),
    piechart,
    bar_graph,

])


@app.callback(Output('pie-world-map', 'figure'), [Input('pie-countries', 'value')])
def update_world_map(selected_value):
    return px.choropleth(locations=[selected_value])


@app.callback(Output('pie-bar-graph', 'figure'), [Input('pie-countries', 'value'), Input('pie-feature', 'value'), Input('pie-time', 'value')])
def update_bar_graph(country_val, ind_val, time_val):
    ind_list = ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'] if ind_val == 'mode' else [
        'DT.NFL.IMFC.CD', 'DT.NFL.IMFN.CD', 'DT.NFL.MIBR.CD', 'DT.NFL.MIDA.CD', 'DT.NFL.RDBC.CD', 'DT.NFL.RDBN.CD']

    return go.Figure(data=[
        go.Bar(name=series[series['Series Code'] == x]['Indicator Name'].to_list()[0], x=[country_val], y=idsdata_[(idsdata_[
            'Country Code'].isin([country_val])) & (idsdata_['Indicator Code'] == x) & (idsdata_['Year'] == time_val)]['Value'].to_list())
        for x in ind_list
    ])


@app.callback(Output('pie-pie-chart', 'figure'), [Input('pie-countries', 'value'), Input('pie-feature', 'value'), Input('pie-time', 'value')])
def update_pie_chart(country_val, ind_val, time_val):
    ind_list = ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'] if ind_val == 'mode' else [
        'DT.NFL.IMFC.CD', 'DT.NFL.IMFN.CD', 'DT.NFL.MIBR.CD', 'DT.NFL.MIDA.CD', 'DT.NFL.RDBC.CD', 'DT.NFL.RDBN.CD']
    return px.pie(idsdata_[(idsdata_['Country Code'] == country_val) & (idsdata_['Year'] == time_val) & (idsdata_['Indicator Code'].isin(
        ind_list))], values='Value', names='Indicator Name', title='Net financial flow from various agencies')


@app.callback(Output('pie-table', 'children'), [Input('pie-countries', 'value'), Input('pie-feature', 'value'), Input('pie-time', 'value')])
def update_table(country_val, ind_val, time_val):
    ind_list = ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'] if ind_val == 'mode' else [
        'DT.NFL.IMFC.CD', 'DT.NFL.IMFN.CD', 'DT.NFL.MIBR.CD', 'DT.NFL.MIDA.CD', 'DT.NFL.RDBC.CD', 'DT.NFL.RDBN.CD']
    return [
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'] == country_val) & (idsdata_['Indicator Code'].isin(ind_list)) & (
            idsdata_['Year'] == time_val)], max_rows=np.inf)
    ]


@app.callback(Output('pie-year_holder', 'children'), [Input('pie-time', 'value')])
def update_range_display(time_val):
    return [html.Label(time_val)]


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
