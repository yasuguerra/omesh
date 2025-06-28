import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import plotly.graph_objects as go
import numpy as np
import io
import base64
from wordcloud import WordCloud
import logging

# --- Funciones de UI reutilizables ---

def create_ai_chat_interface(tab_id_prefix):
    """Create chat interface for a specific tab."""
    return dbc.Card([
        dbc.CardHeader(f"Chat with Omesh AI 🤖 ({tab_id_prefix})", className="text-white bg-primary"),
        dbc.CardBody([
            html.Div(id=f'{tab_id_prefix}-chat-history', style={'height': '150px', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'padding': '10px', 'marginBottom': '10px', 'background': '#f8f9fa'}),
            dbc.InputGroup([
                dbc.Input(id=f'{tab_id_prefix}-chat-input', placeholder="Ask the AI something..."),
                dbc.Button("Send", id=f'{tab_id_prefix}-chat-submit', color="primary", n_clicks=0),
            ])
        ])
    ], className="mt-4")

def create_ai_insight_card(card_id_visible, title="🤖 AI Analysis"):
    """Create a card to display AI analysis."""
    return dbc.Card(dbc.CardBody([
        html.H4(title, className="card-title text-primary"),
        html.P(id=card_id_visible)
    ]), className="mt-3 shadow-sm", color="light")

def add_trendline(fig, df, x_col, y_col):
    """Add a trendline to a Plotly figure."""
    if not df.empty and y_col in df.columns and x_col in df.columns:
        df_sorted = df.sort_values(x_col).dropna(subset=[y_col]).copy()
        if len(df_sorted) >= 2:
            x_numeric = np.arange(len(df_sorted))
            y_numeric = df_sorted[y_col].values
            try:
                coeffs = np.polyfit(x_numeric, y_numeric, 1)
                trendline = np.poly1d(coeffs)(x_numeric)
                fig.add_trace(go.Scatter(
                    x=df_sorted[x_col],
                    y=trendline,
                    mode='lines',
                    name=f'Tendencia {y_col}',
                    line=dict(color='darkred', dash='dash', width=2.5)
                ))
            except Exception as e:
                logging.warning(f"Trendline computation failed para {y_col}: {e}")
    return fig

def generate_wordcloud(text):
    """Genera una imagen de nube de palabras en base64."""
    if not text or not isinstance(text, str) or not text.strip():
        return ""
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        return f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
    except Exception as e:
        logging.error(f"Error generando wordcloud: {e}")
        return ""

# --- Layouts Principales ---

def create_ops_sales_layout():
    """Crea el layout para la pestaña de Operaciones y Ventas."""
    ai_insight_card_style = {"marginTop": "20px", "marginBottom": "20px"}
    return html.Div([
        html.H1("🛩️ Sky Ride Comparison", style={'textAlign': 'center', 'margin-bottom': 20}),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag or ', html.A('Select one or more CSV files')]),
            style={
                'width': '98%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                'textAlign': 'center', 'margin': 'auto', 'margin-bottom': '30px'
            },
            multiple=True
        ),
        dbc.Container(id='output-kpis', fluid=True, className="mb-4"),
        html.Div([
            html.Div([
                html.Label("Filter by destination:"),
                dcc.Dropdown(id='destino-filter', multi=True, placeholder="Select one or more destinations")
            ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
            html.Div([
                html.Label("Filter by operator:"),
                dcc.Dropdown(id='operador-filter', multi=True, placeholder="Select one or more operators")
            ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
            html.Div([
                html.Label("Filter by month:"),
                dcc.Dropdown(id='mes-filter', multi=True, placeholder="Select one or more months")
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'margin-bottom': '30px'}),
        dcc.Tabs([
            dcc.Tab(label='General Comparison', children=[
                html.H3("Monthly Comparison"),
                dcc.Graph(id='vuelos-mes'),
                dcc.Graph(id='ingresos-mes'),
                dcc.Graph(id='ganancia-mes'),
                html.Hr(),
                html.H3("Total Monthly Profit (timeline + trend)"),
                dcc.Graph(id='ganancia-total-mes'),
                html.H3("Total Monthly Operations (trend)"),
                dcc.Graph(id='ops-total-mes'),
                html.Hr(),
                html.H3("Series de Tiempo Comparativas (Semanal)"),
                dcc.Graph(id='vuelos-tiempo'),
                dcc.Graph(id='ingresos-tiempo'),
                dcc.Graph(id='ganancia-tiempo'),
                dbc.Card(dbc.CardBody([
                    html.H4("🤖 AI Analysis", className="card-title text-primary"),
                    html.P(id='ai-insight-comparativo-general')
                ]), style=ai_insight_card_style, color="light", className="shadow-sm")
            ]),
            dcc.Tab(label='Flights and Destinations', children=[
                html.H3("Top Destinations by Number of Flights (descending)"),
                dcc.Graph(id='top-destinos-vuelos'),
                html.H3("Top Destinations by Profit (descending)"),
                dcc.Graph(id='top-destinos-ganancia'),
                html.H3("Top Destinations by Passengers (descending)"),
                dcc.Graph(id='pasajeros-destino'),
                dbc.Card(dbc.CardBody([
                    html.H4("🤖 AI Analysis", className="card-title text-primary"),
                    html.P(id='ai-insight-vuelos-destinos')
                ]), style=ai_insight_card_style, color="light", className="shadow-sm")
            ]),
            dcc.Tab(label='Operators and Aircraft', children=[
                html.H3("Flights per Operator (descending)"),
                dcc.Graph(id='vuelos-operador'),
                html.H3("Profit by Aircraft (descending)"),
                dcc.Graph(id='ganancia-aeronave'),
                html.H3("Operators with Highest Total Profit"),
                dcc.Graph(id='top-ganancia-operador'),
                html.H3("Aircraft with Highest Total Profit"),
                dcc.Graph(id='top-ganancia-aeronave'),
                dbc.Card(dbc.CardBody([
                    html.H4("🤖 AI Analysis", className="card-title text-primary"),
                    html.P(id='ai-insight-operadores-aeronaves')
                ]), style=ai_insight_card_style, color="light", className="shadow-sm")
            ]),
            dcc.Tab(label='Advanced Analysis', children=[
                html.Div([
                    html.Label("Select Destination for Heatmap:"),
                    dcc.Dropdown(id='destino-heatmap', placeholder="Choose destination", style={'width': '50%'}),
                ], style={'margin-bottom': '20px'}),
                html.H3("Heatmap: Day and Hour per Destination (Profit)"),
                dcc.Graph(id='heatmap-gain-destino-dia'),
                html.H3("Heatmap: Day and Hour per Destination (Operations)"),
                dcc.Graph(id='heatmap-count-destino-dia'),
                html.H3("Flights by Day and Hour (6am-18:00, last year, rdylbu scale)"),
                dcc.Graph(id='heatmap-dia-hora'),
                html.H3("Average Ticket by Destination and Year"),
                dcc.Graph(id='ticket-promedio'),
                dbc.Card(dbc.CardBody([
                    html.H4("🤖 AI Analysis", className="card-title text-primary"),
                    html.P(id='ai-insight-analisis-avanzado')
                ]), style=ai_insight_card_style, color="light", className="shadow-sm")
            ]),
            dcc.Tab(label='Tabla Detallada', children=[
                dash_table.DataTable(id='tabla-detallada', page_size=15, style_table={'overflowX': 'auto'})
            ])
        ]),
        html.Div(id='error-message', style={'color': 'red', 'fontWeight': 'bold', 'marginTop': 20, 'textAlign': 'center'})
    ])

def create_web_social_layout(min_date_allowed, max_date_allowed, start_date_val, end_date_val):
    """Create the layout for the Web and Social Analytics tab."""
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.DatePickerRange(id='date-picker', min_date_allowed=min_date_allowed, max_date_allowed=max_date_allowed, start_date=start_date_val, end_date=end_date_val, display_format='YYYY-MM-DD', className='mb-2'), width=12, md=6),
        ], className="mb-4"),
        html.Hr(),
        dcc.Tabs(id='main-tabs-selector-ws', value='overview_ws', children=[
            dcc.Tab(label='Business Overview 🌐', value='overview_ws'),
            dcc.Tab(label='Google Analytics 📈', value='google_ws'),
            dcc.Tab(label='Google Ads 💰', value='google_ads_ws'),
            dcc.Tab(label='Social Media 📱', value='social_media_ws'),
        ], className='mb-4'),
        dcc.Loading(id="loading-tabs-ws", type="circle", children=html.Div(id='main-tabs-content-ws')),
    ])