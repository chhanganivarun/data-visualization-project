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


side_elements = html.Div(className='column', children=[
    html.Div(id='tour-table', className='table-pane', children=[
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) & (idsdata_[
                       'Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] <= 2018)], max_rows=np.inf)

    ]),
    html.Div(className='text-pane', children=[
        html.Label('text'),
        # generate_table(df, np.inf),
    ]),

])

slider_elements = html.Div(className='slider-pane', children=[
    html.Label('Slide'),
    dcc.Slider(
        id='tour-slider',
        min=1,
        max=10,
        value=1,
    ),
    html.Div(id='tour-progress', children=[
        html.Label('1'),

    ]),
    html.Button('Back', id='tour-slide-back', n_clicks=0),
    html.Button('Next', id='tour-slide-next', n_clicks=0),

])

world_map = dcc.Graph(id='tour-world-map', figure=px.choropleth(
    locations=['IND', 'CHN']))


chart = dcc.Graph(id='tour-chart', figure=px.line(idsdata_[(idsdata_['Country Code'].isin(['IND'])) &
                                                           (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] <= 2018)], x="Year", y="Value", color='CI Name'))


main_space = html.Div(className="mid-pane", children=[
    world_map,
    chart,

])


@app.callback(Output('tour-slider', 'value'), [Input('tour-slide-back', 'n_clicks'), Input('tour-slide-next', 'n_clicks')])
def on_click(back_click, next_click):
    return min(max(1-back_click+next_click, 1), 10)


@app.callback(Output('tour-progress', 'children'), [Input('tour-slider', 'value')])
def on_change(value):
    return [
        html.Label(value)
    ]


@app.callback(Output('tour-chart', 'figure'), [(Input('tour-slider', 'value'))])
def update_chart(value):
    if value == 1:
        countries = ['IND', 'CHN', 'PAK', 'RUS', 'BRA']
        temp1 = idsdata_[idsdata_['Indicator Code'] ==
                         'BX.GRT.EXTA.CD.DT'].sort_values('Country Code')
        temp2 = idsdata_[idsdata_['Indicator Code'] ==
                         'BX.GRT.TECH.CD.DT'].sort_values('Country Code')
        # temp1.join(temp2,on = ['Country Code'],how = 'out')
        temp3 = pd.merge(temp1, temp2, on=[
                         'Year', 'Country Code', 'Country Name'], how='outer').fillna(0)
        temp3['Grants'] = temp3['Value_x'] + temp3['Value_y']
        temp3[['Country Code', 'Country Name', 'Year', 'Grants']]
        years = [2008, 2016]
        df = temp3[(temp3['Country Code'].isin(countries)) & (temp3[
            'Year'] <= 2018) & (temp3['Year'] >= years[0]) & (temp3['Year'] <= years[1])].sort_values(['Country Code', 'Year'])
        return px.line(df, x="Year", y="Grants", color='Country Name')


@app.callback(Output('tour-world-map', 'figure'), [Input('tour-slider', 'value')])
def update_world_map(value):
    if value == 1:
        locs = ['IND', 'CHN', 'PAK', 'RUS', 'BRA']
        return px.choropleth(locations=locs)
    else:
        return px.choropleth(locations=['IND'])


@app.callback(Output('tour-table', 'children'), [Input('tour-slider', 'value')])
def update_table(value):
    if value == 1:
        countries = ['IND', 'CHN', 'PAK', 'RUS', 'BRA']
        temp1 = idsdata_[idsdata_['Indicator Code'] ==
                         'BX.GRT.EXTA.CD.DT'].sort_values('Country Code')
        temp2 = idsdata_[idsdata_['Indicator Code'] ==
                         'BX.GRT.TECH.CD.DT'].sort_values('Country Code')
        # temp1.join(temp2,on = ['Country Code'],how = 'out')
        temp3 = pd.merge(temp1, temp2, on=[
                         'Year', 'Country Code', 'Country Name'], how='outer').fillna(0)
        temp3['Grants'] = temp3['Value_x'] + temp3['Value_y']
        temp3[['Country Code', 'Country Name', 'Year', 'Grants']]
        years = [2008, 2016]
        df = temp3[(temp3['Country Code'].isin(countries)) & (temp3[
            'Year'] <= 2018) & (temp3['Year'] >= years[0]) & (temp3['Year'] <= years[1])]
        return [
            html.Label('Data'),
            generate_table(df),
        ]
    return [
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == ind_val) & (
            idsdata_['Year'] <= 2018) & (idsdata_['Year'] >= time_val[0]) & (idsdata_['Year'] <= time_val[1])], max_rows=np.inf)
    ]


@app.callback(Output('tour-Range', 'children'), [Input('tour-time-window-slider', 'value')])
def update_range_display(time_val):
    return [html.Label('{} - {}'.format(time_val[0], time_val[1]))]


layout = [html.Div(children=[
    html.Div(className='row', children=[
        dcc.Link('Tour', href='/'),
        dcc.Link('Country Wise Breakup of Debt',
                 href='/pie-chart', style={"margin-left": "15px"}),
        dcc.Link('Debt Breakup Comparison', href='/stacked-bar',
                 style={"margin-left": "15px"}),
        dcc.Link('Explore Countries', href='/mlc',
                 style={"margin-left": "15px"}),
        dcc.Link('Explore Indicators', href='/mli',
                 style={"margin-left": "15px"}),
        dcc.Link('Explore Indicators and Countries', href='/mci',
                 style={"margin-left": "15px"}),
    ]),
    html.Div(className="row", children=[
        slider_elements
    ]),
    html.Div(className="row", children=[
        html.Div(className="left-panel", children=[
            side_elements
        ]),
        main_space,
    ])
])]
