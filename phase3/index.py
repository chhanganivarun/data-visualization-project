from apps import tour, stacked_bar, pie_chart, multi_country_line_chart, multi_indicator_line_chart
from app import app
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import webbrowser
import os

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return tour.layout
    if pathname == '/pie-chart':
        return pie_chart.layout
    elif pathname == '/stacked-bar':
        return stacked_bar.layout
    elif pathname == '/mlc':
        return multi_country_line_chart.layout
    elif pathname == '/mli':
        return multi_indicator_line_chart.layout
    else:
        return '404'


# webbrowser.open('http://localhost:8050', new=2)

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=True)
