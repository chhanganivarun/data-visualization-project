import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    './assets/index.css',
]
# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')

country_series = pd.read_csv('../IDS_CSV/IDScountry-series.csv')
country_series.drop(['Unnamed: 3'], axis=1, inplace=True)

country = pd.read_csv('../IDS_CSV/IDSCountry.csv')
country.drop(['Unnamed: 31'], axis=1, inplace=True)

idsdata = pd.read_csv('../IDS_CSV/IDSData.csv')
idsdata.drop(['Unnamed: 61'], axis=1, inplace=True)

idsdata_ = idsdata.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name',
                                 'Indicator Code'], var_name='Year', value_name='Value').fillna(0)
idsdata_['Year'] = idsdata_['Year'].apply(lambda x: int(x))

footnote = pd.read_csv('../IDS_CSV/IDSfootnote.csv')
footnote.drop(['Unnamed: 4'], axis=1, inplace=True)

series = pd.read_csv('../IDS_CSV/IDSSeries.csv')
series.drop(['Unnamed: 20'], axis=1, inplace=True)

series_time = pd.read_csv('../IDS_CSV/IDSseries-time.csv')
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

bar_options = [
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


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


side_elements = html.Div(className='column', children=[
    html.Div(id='table', className='table-pane', children=[
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(['IND', 'CHN'])) & (idsdata_[
                       'Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] <= 2018)], max_rows=np.inf)

    ]),
    html.Div(className='text-pane', children=[
        html.Label('text'),
        # generate_table(df, np.inf),
    ]),

])

world_map = dcc.Graph(id='world-map', figure=px.choropleth(
    locations=['IND', 'CHN']))
# ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD']
stacked_bar_graph = dcc.Graph(id='stacked-bar-graph', figure=go.Figure(data=[
    go.Bar(name='Net financial flows, bilateral (NFL, current US$)', x=['IND', 'CHN'], y=idsdata_[(idsdata_[
           'Country Code'].isin(sorted(['IND', 'CHN']))) & (idsdata_['Indicator Code'] == 'DT.NFL.BLAT.CD') & (idsdata_['Year'] == 2008)].sort_values('Country Code')['Value'].to_list()),
    go.Bar(name='Net financial flows, multilateral (NFL, current US$)', x=['IND', 'CHN'], y=idsdata_[(idsdata_[
           'Country Code'].isin(sorted(['IND', 'CHN']))) & (idsdata_['Indicator Code'] == 'DT.NFL.MLAT.CD') & (idsdata_['Year'] == 2008)].sort_values('Country Code')['Value'].to_list()),
    go.Bar(name='Net financial flows, others (NFL, current US$)', x=['IND', 'CHN'], y=idsdata_[(idsdata_[
           'Country Code'].isin(sorted(['IND', 'CHN']))) & (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] == 2008)].sort_values('Country Code')['Value'].to_list()),
], layout={'barmode': 'stack'}))
bar_graph = dcc.Graph(id='bar-graph', figure=go.Figure(data=[
    go.Bar(name='Net financial flows, bilateral (NFL, current US$)', x=['IND', 'CHN'], y=idsdata_[(idsdata_[
           'Country Code'].isin(sorted(['IND', 'CHN']))) & (idsdata_['Indicator Code'] == 'DT.NFL.BLAT.CD') & (idsdata_['Year'] == 2008)].sort_values('Country Code')['Value'].to_list()),
    go.Bar(name='Net financial flows, multilateral (NFL, current US$)', x=['IND', 'CHN'], y=idsdata_[(idsdata_[
           'Country Code'].isin(sorted(['IND', 'CHN']))) & (idsdata_['Indicator Code'] == 'DT.NFL.MLAT.CD') & (idsdata_['Year'] == 2008)].sort_values('Country Code')['Value'].to_list()),
    go.Bar(name='Net financial flows, others (NFL, current US$)', x=['IND', 'CHN'], y=idsdata_[(idsdata_[
           'Country Code'].isin(sorted(['IND', 'CHN']))) & (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] == 2008)].sort_values('Country Code')['Value'].to_list()),
]))


main_space = html.Div(className="mid-pane", children=[
    world_map,

    html.Label('Countries'),
    dcc.Dropdown(
        id='countries',
        options=dropdown_options,
        value=['IND', 'CHN'],
        multi=True
    ),

    html.Label('Feature'),
    dcc.Dropdown(
        id='feature',
        options=bar_options,
        value='mode',
    ),

    html.Label('Year'),
    dcc.Slider(
        id='time',
        min=1970,
        max=2018,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i)
               for i in range(1970, 2018, 10)},
        value=2008,
    ),
    html.Div(id='year_holder', children=[
        html.Label('2008'),
    ]),


    stacked_bar_graph,
    bar_graph,

])


@app.callback(Output('world-map', 'figure'), [Input('countries', 'value')])
def update_world_map(selected_value):
    return px.choropleth(locations=selected_value)


@app.callback(Output('stacked-bar-graph', 'figure'), [Input('countries', 'value'), Input('feature', 'value'), Input('time', 'value')])
def update_stacked_bar_graph(country_values, ind_val, time_val):
    country_vals = sorted(country_values)
    if ind_val == 'mode':
        return go.Figure(data=[
            go.Bar(name='Net financial flows, bilateral (NFL, current US$)', x=country_vals, y=idsdata_[(idsdata_[
                'Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.BLAT.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()),
            go.Bar(name='Net financial flows, multilateral (NFL, current US$)', x=country_vals, y=idsdata_[(idsdata_[
                'Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MLAT.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()),
            go.Bar(name='Net financial flows, others (NFL, current US$)', x=country_vals, y=idsdata_[(idsdata_[
                'Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()),
        ], layout={'barmode': 'stack'})
    elif ind_val == 'source':
        imf = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) +\
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFN.CD') & (
                idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        rdb = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) +\
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBN.CD') & (
                idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        ibr = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIBR.CD') & (
            idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        ida = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIDA.CD') & (
            idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        return go.Figure(data=[
            go.Bar(name='IMF (NFL, current US$)',
                   x=country_vals, y=imf),
            go.Bar(name='RDB (NFL, current US$)',
                   x=country_vals, y=rdb),
            go.Bar(name='IBR (NFL, current US$)',
                   x=country_vals, y=ibr),
            go.Bar(name='IDA (NFL, current US$)',
                   x=country_vals, y=ida),
        ], layout={'barmode': 'stack'})

    else:  # ind_val == 'type'
        concessional = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')[
            'Value'].to_list()) + np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())

        nonconcessional = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIBR.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) + \
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIDA.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) + \
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFN.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) + \
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBN.CD') & (
                idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())

        return go.Figure(data=[
            go.Bar(name='Concessional (NFL, current US$)',
                   x=country_vals, y=concessional),
            go.Bar(name='Non-Concessional (NFL, current US$)',
                   x=country_vals, y=nonconcessional),
        ], layout={'barmode': 'stack'})


@app.callback(Output('bar-graph', 'figure'), [Input('countries', 'value'), Input('feature', 'value'), Input('time', 'value')])
def update_bar_graph(country_values, ind_val, time_val):
    country_vals = sorted(country_values)
    if ind_val == 'mode':
        return go.Figure(data=[
            go.Bar(name='Net financial flows, bilateral (NFL, current US$)', x=country_vals, y=idsdata_[(idsdata_[
                'Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.BLAT.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()),
            go.Bar(name='Net financial flows, multilateral (NFL, current US$)', x=country_vals, y=idsdata_[(idsdata_[
                'Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MLAT.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()),
            go.Bar(name='Net financial flows, others (NFL, current US$)', x=country_vals, y=idsdata_[(idsdata_[
                'Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MOTH.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()),
        ])
    elif ind_val == 'source':
        imf = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) +\
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFN.CD') & (
                idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        rdb = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) +\
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBN.CD') & (
                idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        ibr = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIBR.CD') & (
            idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        ida = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIDA.CD') & (
            idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        return go.Figure(data=[
            go.Bar(name='IMF (NFL, current US$)',
                   x=country_vals, y=imf),
            go.Bar(name='RDB (NFL, current US$)',
                   x=country_vals, y=rdb),
            go.Bar(name='IBR (NFL, current US$)',
                   x=country_vals, y=ibr),
            go.Bar(name='IDA (NFL, current US$)',
                   x=country_vals, y=ida),
        ])
    else:  # ind_val == 'type'
        concessional = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')[
            'Value'].to_list()) + np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBC.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())

        nonconcessional = np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIBR.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) + \
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.MIDA.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) + \
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.IMFN.CD') & (idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list()) + \
            np.array(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'] == 'DT.NFL.RDBN.CD') & (
                idsdata_['Year'] == time_val)].sort_values('Country Code')['Value'].to_list())
        return go.Figure(data=[
            go.Bar(name='Concessional (NFL, current US$)',
                   x=country_vals, y=concessional),
            go.Bar(name='Non-Concessional (NFL, current US$)',
                   x=country_vals, y=nonconcessional),
        ])


@app.callback(Output('table', 'children'), [Input('countries', 'value'), Input('feature', 'value'), Input('time', 'value')])
def update_table(country_vals, ind_val, time_val):
    print(time_val)
    ind_list = ['DT.NFL.BLAT.CD', 'DT.NFL.MLAT.CD', 'DT.NFL.MOTH.CD'] if ind_val == 'mode' else [
        'DT.NFL.IMFC.CD', 'DT.NFL.IMFN.CD', 'DT.NFL.MIBR.CD', 'DT.NFL.MIDA.CD', 'DT.NFL.RDBC.CD', 'DT.NFL.RDBN.CD']
    return [
        html.Label('Data'),
        generate_table(idsdata_[(idsdata_['Country Code'].isin(country_vals)) & (idsdata_['Indicator Code'].isin(ind_list)) & (
            idsdata_['Year'] == time_val)], max_rows=np.inf)
    ]


@app.callback(Output('year_holder', 'children'), [Input('time', 'value')])
def update_range_display(time_val):
    return [html.Label(time_val)]


app.layout = html.Div(className="row", children=[
    html.Div(className="left-panel", children=[
        side_elements
    ]),
    main_space,
])
# webbrowser.open('http://localhost:8050', new=2)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
