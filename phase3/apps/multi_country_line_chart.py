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
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    './assets/index.css',
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


side_elements = html.Div(className='column', children=[
    html.Div(id='mcl-table', className='table-pane', children=[
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) & (idsdata_[
                       'Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] <= 2018)], max_rows=np.inf)

    ]),
    html.Div(className='text-pane', children=[
        html.Label('text'),
        # generate_table(df, np.inf),
    ]),

])

world_map = dcc.Graph(id='mcl-world-map', figure=px.choropleth(
    locations=['IND', 'CHN']))


indicator_line_chart = dcc.Graph(id='mcl-ind-line-chart', figure=px.line(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) &
                                                                                  (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] <= 2018)], x="Year", y="Value", color='Country Name'))


main_space = html.Div(className="mid-pane", children=[
    world_map,

    html.Label('Countries'),
    dcc.Dropdown(
        id='mcl-countries',
        options=dropdown_options,
        value=['IND', 'CHN'],
        multi=True
    ),

    html.Label('Indicator'),
    dcc.Dropdown(
        id='mcl-indicators',
        options=[{'label': y, 'value': x} for x, y in indicator_codes_names],
        value='DT.NFL.MOTH.CD',
    ),

    html.Label('Time window'),
    dcc.RangeSlider(
        id='mcl-time-window-slider',
        min=1970,
        max=2018,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i)
               for i in range(1970, 2018, 10)},

        step=1,
        value=[2008, 2018]
    ),
    html.Div(id='mcl-Range', children=[
        html.Label('2000 - 2000'),
    ]),


    indicator_line_chart,

])


@app.callback(Output('mlc-world-map', 'figure'), [Input('mlc-countries', 'value')])
def update_world_map(selected_value):
    return px.choropleth(locations=selected_value)


@app.callback(Output('mlc-ind-line-chart', 'figure'), [Input('mlc-countries', 'value'), Input('mlc-indicators', 'value'), Input('mlc-time-window-slider', 'value')])
def update_ind_line_chart(country_vals, ind_val, time_val):
    return px.line(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == ind_val) & (idsdata_['Year'] <= 2018) & (idsdata_['Year'] >= time_val[0]) & (idsdata_['Year'] <= time_val[1])], x="Year", y="Value", color="Country Name")


@app.callback(Output('mlc-table', 'children'), [Input('mlc-countries', 'value'), Input('mlc-indicators', 'value'), Input('mlc-time-window-slider', 'value')])
def update_table(country_vals, ind_val, time_val):
    return [
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == ind_val) & (
            idsdata_['Year'] <= 2018) & (idsdata_['Year'] >= time_val[0]) & (idsdata_['Year'] <= time_val[1])], max_rows=np.inf)
    ]


@app.callback(Output('mlc-Range', 'children'), [Input('mlc-time-window-slider', 'value')])
def update_range_display(time_val):
    return [html.Label('{} - {}'.format(time_val[0], time_val[1]))]


layout = [html.Div(children=[
    html.Div(className='row', children=[
        dcc.Link('Tour', href='/'),
        html.Br(),
        dcc.Link('Country Wise Breakup of Debt',
                 href='/pie-chart', style={"margin-left": "15px"}),
        html.Label(' '),
        dcc.Link('Debt Breakup Comparison', href='/stacked-bar',
                 style={"margin-left": "15px"}),
        html.Label(' '),
        dcc.Link('Explore Countries', href='/mlc',
                 style={"margin-left": "15px"}),
        html.Label(' '),

    ]),
    html.Div(className="row", children=[
        html.Div(className="left-panel", children=[
            side_elements
        ]),
        main_space,
    ])
])]
