import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import webbrowser

from app import app
from apps import tour, pie_chart, multi_country_line_chart, stacked_bar


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return tour.layout
    if pathname == '/apps/pie_chart':
        return pie_chart.layout
    elif pathname == '/apps/stacked_bar':
        return stacked_bar.layout
    else:
        return '404'


webbrowser.open('http://localhost:8050', new=2)

if __name__ == '__main__':
    app.run_server(debug=True)
