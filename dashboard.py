# dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Fetch fraud stats from the Flask backend
def fetch_fraud_stats():
    response = requests.get("http://localhost:5000/fraud-stats")
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Layout for the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'text-align': 'center'}),
    
    # Summary Boxes
    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.P(id="total-transactions", children="Loading...")
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '20px'}),
        html.Div([
            html.H3("Fraud Cases"),
            html.P(id="fraud-cases", children="Loading...")
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '20px'}),
        html.Div([
            html.H3("Fraud Percentage"),
            html.P(id="fraud-percentage", children="Loading...")
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '20px'})
    ]),
    
    # Line Chart for Fraud Trends
    dcc.Graph(id="fraud-trends"),
    
    # Bar Chart for Device and Browser Analysis
    dcc.Graph(id="device-browser-analysis"),
    
    # Geographical Distribution of Fraud Cases
    dcc.Graph(id="fraud-geolocation"),
    
    # Add an interval component to trigger periodic updates
    dcc.Interval(
        id="interval-component",
        interval=60 * 1000,  # Update every 60 seconds
        n_intervals=0
    )
])

# Callbacks for updating dashboard components
@app.callback(
    [Output("total-transactions", "children"),
     Output("fraud-cases", "children"),
     Output("fraud-percentage", "children"),
     Output("fraud-trends", "figure"),
     Output("device-browser-analysis", "figure")],
    [Input("interval-component", "n_intervals")]  # Triggered by the interval component
)
def update_dashboard(n):
    stats = fetch_fraud_stats()
    if stats:
        # Extract summary stats
        total_transactions = stats["summary"]["total_transactions"]
        fraud_cases = stats["summary"]["fraud_cases"]
        fraud_percentage = f"{stats['summary']['fraud_percentage']:.2f}%"
        
        # Create fraud trends line chart
        fraud_trends = pd.DataFrame(stats["fraud_trends"])
        fig_fraud_trends = px.line(fraud_trends, x="hour_of_day", y="count", title="Fraud Cases Over Time")
        
        # Create device/browser bar chart
        device_browser_fraud = pd.DataFrame(stats["device_browser_fraud"])
        fig_device_browser = px.bar(device_browser_fraud, x="device_id", y="fraud_count", color="browser", title="Fraud Cases by Device and Browser")
        
        return (
            total_transactions,
            fraud_cases,
            fraud_percentage,
            fig_fraud_trends,
            fig_device_browser
        )
    else:
        return ("Error", "Error", "Error", {}, {})

# Add a callback for geolocation
@app.callback(
    Output("fraud-geolocation", "figure"),
    [Input("interval-component", "n_intervals")]  # Triggered by the interval component
)
def update_geolocation(n):
    response = requests.get("http://localhost:5000/fraud-geolocation")
    if response.status_code == 200:
        geolocation_data = pd.DataFrame(response.json()["geolocation"])
        fig_geolocation = px.choropleth(
            geolocation_data,
            locations="country",
            locationmode="country names",
            color="fraud_count",
            title="Geographical Distribution of Fraud Cases"
        )
        return fig_geolocation
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)