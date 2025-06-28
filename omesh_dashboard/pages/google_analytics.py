from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, dash_table, callback # Added callback
import dash_bootstrap_components as dbc
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore
from sklearn.preprocessing import MinMaxScaler
import logging
import typing

# Project dependencies
from ..data import ga_data_access as ga_data # Use new data access layer
from ..ai import get_openai_response
from ..ui.components import create_ai_insight_card, create_ai_chat_interface, add_trendline

logger = logging.getLogger(__name__)

# Standard margin for all figures
FIGURE_MARGIN = dict(t=40, l=20, r=20, b=40)
DEFAULT_NO_DATA_FIGURE_TITLE = "No data to display"

# Helper function to create an empty figure
def create_empty_figure(title: str = DEFAULT_NO_DATA_FIGURE_TITLE) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': title, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}],
        margin=FIGURE_MARGIN
    )
    return fig

# --- Figure Generation Functions ---
# Overview Tab
def get_fig_sessions_overview(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'date' not in df.columns or 'sessions' not in df.columns:
        return create_empty_figure("Sessions Over Time")
    fig = px.line(df, x='date', y='sessions', title='Sessions Over Time', markers=True)
    add_trendline(fig, df, 'date', 'sessions', trendline_name_prefix="Trend")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Sessions")
    return fig

def get_fig_users_overview(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'date' not in df.columns or 'activeUsers' not in df.columns:
        return create_empty_figure("Users Over Time")
    fig = px.line(df, x='date', y='activeUsers', title='Active Users Over Time', markers=True)
    add_trendline(fig, df, 'date', 'activeUsers', trendline_name_prefix="Trend")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Active Users")
    return fig

def get_fig_conversions_overview(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'date' not in df.columns or 'conversions' not in df.columns:
        return create_empty_figure("Conversions Over Time")
    fig = px.line(df, x='date', y='conversions', title='Conversions Over Time', markers=True)
    add_trendline(fig, df, 'date', 'conversions', trendline_name_prefix="Trend")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Conversions")
    return fig

def get_fig_conversion_rate_overview(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'date' not in df.columns or 'conversionRate' not in df.columns:
        return create_empty_figure("Conversion Rate Over Time (%)")
    fig = px.line(df, x='date', y='conversionRate', title='Conversion Rate Over Time (%)', markers=True)
    add_trendline(fig, df, 'date', 'conversionRate', trendline_name_prefix="Trend")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Conversion Rate (%)")
    return fig

def get_fig_normalized_trends_overview(df_acq: pd.DataFrame) -> go.Figure:
    if df_acq.empty: return create_empty_figure("Normalized Trends")
    cols_to_normalize = ['sessions', 'activeUsers', 'conversions']
    existing_cols = [col for col in cols_to_normalize if col in df_acq.columns]
    if not existing_cols: return create_empty_figure("Normalized Trends (No data)")

    df_norm_src = df_acq[existing_cols].copy().fillna(0)
    if df_norm_src.empty or df_norm_src.isnull().all().all() or (df_norm_src.max() - df_norm_src.min()).sum() == 0: # Avoid division by zero if all values are same
        return create_empty_figure("Normalized Trends (Insufficient variance or data)")

    scaler = MinMaxScaler()
    df_norm_values = scaler.fit_transform(df_norm_src)
    df_norm = pd.DataFrame(df_norm_values, columns=df_norm_src.columns, index=df_acq['date'] if 'date' in df_acq else None)
    fig = px.line(df_norm, title='Normalized Trends')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Normalized Value (0 to 1)")
    return fig

# Demography & Geo Tab
def get_fig_users_by_gender(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'userGender' not in df.columns or 'activeUsers' not in df.columns:
        return create_empty_figure("Users by Gender")
    df_filtered = df[~df['userGender'].isin(['unknown', 'Others', None, '', '(not set)'])].copy()
    if df_filtered.empty: return create_empty_figure("Users by Gender (No data after filtering)")
    fig = px.pie(df_filtered, names='userGender', values='activeUsers', title='Users by Gender')
    fig.update_layout(margin=FIGURE_MARGIN)
    return fig

def get_fig_users_by_age(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'userAgeBracket' not in df.columns or 'activeUsers' not in df.columns:
        return create_empty_figure("Users by Age Bracket")
    df_filtered = df[~df['userAgeBracket'].isin(['unknown', 'Others', None, '', '(not set)'])].copy()
    if df_filtered.empty: return create_empty_figure("Users by Age Bracket (No data after filtering)")
    fig = px.bar(df_filtered.sort_values('userAgeBracket'), x='userAgeBracket', y='activeUsers', title='Users by Age Bracket')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Age Bracket", yaxis_title="Active Users")
    return fig

def get_fig_top_countries_by_users(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'country' not in df.columns or 'activeUsers' not in df.columns:
        return create_empty_figure("Top 10 Countries by Users")
    df_filtered = df[~df['country'].isin(['unknown', 'Others', None, '', '(not set)'])].copy()
    if df_filtered.empty: return create_empty_figure("Top Countries (No data after filtering)")
    top_data = df_filtered.groupby('country', as_index=False)['activeUsers'].sum().sort_values('activeUsers', ascending=False).head(10)
    if top_data.empty: return create_empty_figure("Top 10 Countries by Users (No data for top countries)")
    fig = px.bar(top_data, x='country', y='activeUsers', title='Top 10 Countries by Users')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Country", yaxis_title="Active Users")
    return fig

def get_fig_top_cities_by_users(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'city' not in df.columns or 'activeUsers' not in df.columns:
        return create_empty_figure("Top 10 Cities by Users")
    df_filtered = df[~df['city'].isin(['unknown', 'Others', None, '', '(not set)'])].copy()
    if df_filtered.empty: return create_empty_figure("Top Cities (No data after filtering)")
    top_data = df_filtered.groupby('city', as_index=False)['activeUsers'].sum().sort_values('activeUsers', ascending=False).head(10)
    if top_data.empty: return create_empty_figure("Top 10 Cities by Users (No data for top cities)")
    fig = px.bar(top_data, x='city', y='activeUsers', title='Top 10 Cities by Users')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="City", yaxis_title="Active Users")
    return fig

def get_fig_geo_opportunity_map(df_opportunities: pd.DataFrame) -> go.Figure:
    if df_opportunities.empty or 'country' not in df_opportunities.columns or 'sessions' not in df_opportunities.columns:
        return create_empty_figure("Geo Opportunities: Countries with Sessions & Zero Conversions")
    country_data = df_opportunities.groupby('country', as_index=False)['sessions'].sum().sort_values(by='sessions', ascending=False)
    if country_data.empty : return create_empty_figure("Geo Opportunities Map (No country data)")
    fig = px.choropleth(country_data, locations="country", locationmode="country names", color="sessions",
                        hover_name="country", color_continuous_scale=px.colors.sequential.OrRd,
                        title="Geo Opportunities: Countries with Sessions & Zero Conversions")
    fig.update_layout(margin=FIGURE_MARGIN)
    return fig

# Funnels & Paths Tab
def get_fig_event_evolution(df_events_pivot: pd.DataFrame, kpi_event_list: list[str]) -> go.Figure:
    if df_events_pivot.empty:
        return create_empty_figure("Event Conversion Evolution")

    cols_to_plot = [col for col in kpi_event_list if col in df_events_pivot.columns]
    if not cols_to_plot or 'date' not in df_events_pivot.columns:
        return create_empty_figure("Event Conversion Evolution (Missing data columns)")

    fig = px.line(df_events_pivot.sort_values('date'), x='date', y=cols_to_plot, title="Event Conversion Evolution by Channel")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Event Count")
    return fig

def get_fig_acquisition_channels(df_acq_source: pd.DataFrame) -> go.Figure:
    if df_acq_source.empty or not {'sessionSourceMedium', 'sessions', 'conversions'}.issubset(df_acq_source.columns):
        return create_empty_figure("Acquisition and Conversion by Channel")
    df_plot = df_acq_source.sort_values('sessions', ascending=False).head(10)
    fig = px.bar(df_plot, x='sessionSourceMedium', y=['sessions', 'conversions'], title="Top 10 Acquisition & Conversion by Channel", barmode='group', text_auto=True)
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Source / Medium", yaxis_title="Count")
    return fig

def get_fig_page_visits(df_page_data: pd.DataFrame) -> go.Figure:
    if df_page_data.empty or not {'pagePath', 'sessions'}.issubset(df_page_data.columns):
        return create_empty_figure("Top 10 Visited Pages")
    df_plot = df_page_data.sort_values('sessions', ascending=False).head(10)
    fig = px.bar(df_plot, x='pagePath', y='sessions', title='Top 10 Visited Pages', text_auto=True, height=700)
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Page Path", yaxis_title="Sessions", xaxis_tickangle=45)
    return fig

def get_fig_page_bounce_rate(df_page_data: pd.DataFrame) -> go.Figure:
    if df_page_data.empty or not {'pagePath', 'bounceRate'}.issubset(df_page_data.columns):
        return create_empty_figure("Top 10 Pages by Bounce Rate")
    df_plot = df_page_data.sort_values('bounceRate', ascending=False).head(10)
    # Bounce rate from GA4 is a ratio (0.0 to 1.0). Multiply by 100 for percentage.
    df_plot['bounceRatePct'] = df_plot['bounceRate'] * 100
    fig = px.bar(df_plot, x='pagePath', y='bounceRatePct', title='Top 10 Pages by Bounce Rate (%)', text_auto='.1f', height=700)
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Page Path", yaxis_title="Bounce Rate (%)", xaxis_tickangle=45)
    return fig

def get_fig_page_duration(df_page_data: pd.DataFrame) -> go.Figure:
    if df_page_data.empty or not {'pagePath', 'averageSessionDuration'}.issubset(df_page_data.columns):
        return create_empty_figure("Top 10 Pages by Average Session Duration")
    df_plot = df_page_data.sort_values('averageSessionDuration', ascending=False).head(10)
    fig = px.bar(df_plot, x='pagePath', y='averageSessionDuration', title='Top 10 Pages by Avg. Session Duration (sec)', text_auto='.2f', height=700)
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Page Path", yaxis_title="Avg. Session Duration (seconds)", xaxis_tickangle=45)
    return fig

def get_fig_funnel_chart(labels: list[str], counts: list[int], title: str) -> go.Figure:
    if not labels or not counts or len(labels) != len(counts) or sum(counts) == 0: # Check if counts sum to 0
        return create_empty_figure(f"{title} (No data)")
    fig = go.Figure(go.Funnel(y=labels, x=counts, textinfo="value+percent previous"))
    fig.update_layout(title=title, margin=FIGURE_MARGIN)
    return fig

def get_fig_sankey_user_flow(df_sankey_src: pd.DataFrame, key_events: list[str]) -> go.Figure:
    if df_sankey_src.empty or not {'sessionSourceMedium', 'eventName', 'sessions'}.issubset(df_sankey_src.columns):
        return create_empty_figure("User Flow (Source -> Event) - No Data")

    df_sankey_data = df_sankey_src[df_sankey_src['eventName'].isin(key_events)].copy()
    if df_sankey_data.empty:
        return create_empty_figure("User Flow (Source -> Event) - No matching event data")

    all_nodes = list(pd.concat([df_sankey_data['sessionSourceMedium'], df_sankey_data['eventName']]).unique())
    label_map = {label: i for i, label in enumerate(all_nodes)}

    source_indices = df_sankey_data['sessionSourceMedium'].map(label_map).tolist()
    target_indices = df_sankey_data['eventName'].map(label_map).tolist()
    values = df_sankey_data['sessions'].apply(lambda x: max(x, 0.1)).tolist() # Ensure positive values for Sankey

    if not source_indices:
         return create_empty_figure("User Flow (Sankey) - Could not map nodes")

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color="blue"),
        link=dict(source=source_indices, target=target_indices, value=values)
    )])
    fig.update_layout(title_text="User Flow (Source/Medium -> Key Event)", font_size=10, height=700, margin=FIGURE_MARGIN) # Adjusted font size
    return fig

# Temporal Analysis Tab
def get_fig_temporal_decomposition(df_series: pd.Series | None, period: int = 7) -> go.Figure:
    if df_series is None or df_series.empty or len(df_series) < period * 2: # Need at least 2 periods for decomposition
        return create_empty_figure("Temporal Decomposition & Anomalies (Insufficient data)")
    try:
        decomposition = seasonal_decompose(df_series, model='additive', period=period)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_series.index, y=df_series, mode='lines', name='Original'))
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonality'))

        # Anomaly Detection (basic example: residual > 2*std_dev)
        std_dev_resid = decomposition.resid.std()
        if pd.notna(std_dev_resid) and std_dev_resid > 0:
            anomalies_df = pd.DataFrame({'date': df_series.index, 'value': df_series.values, 'resid': decomposition.resid})
            anomalies = anomalies_df[(anomalies_df['resid'].notna()) & (abs(anomalies_df['resid']) > 2 * std_dev_resid)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['value'], mode='markers', name='Anomalies', marker=dict(color='red', size=10, symbol='x')))

        fig.update_layout(title='Temporal Decomposition & Anomalies (Daily Sessions)', hovermode='x unified', margin=FIGURE_MARGIN)
        return fig
    except Exception as e:
        logger.error(f"Error in temporal decomposition: {e}", exc_info=True)
        return create_empty_figure(f"Temporal Decomposition Error: {e}")

# Correlations Tab
def get_fig_correlation_scatter_matrix(df_corr_data: pd.DataFrame, metrics_list: list[str], color_dim: str) -> go.Figure:
    if df_corr_data.empty:
        return create_empty_figure("Correlation Matrix (No data)")

    plot_dims = [m for m in metrics_list if m in df_corr_data.columns and df_corr_data[m].notna().any()]
    if len(plot_dims) < 2 or color_dim not in df_corr_data.columns:
        return create_empty_figure("Correlation Matrix (Insufficient dimensions or color variable)")

    df_plot = df_corr_data[df_corr_data[color_dim] != '(not set)'].copy() # Filter out '(not set)' for color dimension
    if df_plot.empty:
        return create_empty_figure("Correlation Matrix (No data after filtering color dimension)")

    try:
        fig = px.scatter_matrix(df_plot, dimensions=plot_dims, color=color_dim, title=f"Correlation Matrix by {color_dim.replace('deviceCategory', 'Device Category')}")
        fig.update_layout(height=800, margin=FIGURE_MARGIN)
        return fig
    except Exception as e:
        logger.error(f"Error generating scatter matrix: {e}", exc_info=True)
        return create_empty_figure(f"Correlation Matrix Error: {e}")


def get_fig_conversions_boxplot(df_data: pd.DataFrame, group_by_col: str, title_suffix: str) -> go.Figure:
    if df_data.empty or group_by_col not in df_data.columns or 'conversions' not in df_data.columns:
        return create_empty_figure(f"Conversions by {title_suffix} (No data)")

    df_plot = df_data.copy()
    if group_by_col == 'userAgeBracket' or group_by_col == 'userGender': # Filter out common noise for these dimensions
         df_plot = df_plot[~df_plot[group_by_col].isin(['unknown', 'Others', None, '', '(not set)'])].copy()

    if df_plot.empty:
        return create_empty_figure(f"Conversions by {title_suffix} (No data after filtering)")

    fig = px.box(df_plot, x=group_by_col, y="conversions", title=f"Conversions by {title_suffix}", points="all")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title=title_suffix, yaxis_title="Conversions")
    return fig

# Cohort Analysis Tab
def get_fig_cohort_retention_heatmap(retention_matrix_display: pd.DataFrame) -> go.Figure:
    if retention_matrix_display.empty:
        return create_empty_figure("Cohort Analysis – User Retention (%) (No data)")

    fig = px.imshow(retention_matrix_display,
                    labels=dict(x="Days Since First Session", y="Cohort (First Session Date)", color="Retention (%)"),
                    color_continuous_scale='Blues', aspect='auto', text_auto=".1f")
    fig.update_layout(title="Cohort Analysis – User Retention (%)",
                      xaxis_title="Days Since First Session",
                      yaxis_title="First Session Date (Cohort)",
                      margin=FIGURE_MARGIN)
    fig.update_xaxes(type='category') # Ensure days are treated as categories
    fig.update_yaxes(type='category', tickformat='%Y-%m-%d') # Format cohort dates
    return fig

# --- Layout Definition ---
def layout() -> html.Div:
    # Note: DatePickerRange ID is 'ga-date-picker', used by callbacks.
    # Sub-tab IDs are also referenced in callbacks.
    return html.Div([
        dcc.DatePickerRange(
            id="ga-date-picker", # Specific ID for this page
            display_format="YYYY-MM-DD",
            # min_date_allowed, max_date_allowed, start_date, end_date can be set in app.py or via callback if dynamic
        ),
        dcc.Tabs(id="ga-subtabs", value="overview", children=[ # Renamed IDs and default
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Demographics & Geo", value="demography"),
            dcc.Tab(label="Funnels & Paths", value="funnels"),
            dcc.Tab(label="Temporal Analysis", value="temporal"),
            dcc.Tab(label="Correlations", value="correlations"),
            dcc.Tab(label="Cohort Analysis", value="cohort"),
            dcc.Tab(label="What If Simulator", value="what_if"),
        ]),
        dcc.Loading(id="ga-loading-indicator", type="default", children=html.Div(id="ga-subtabs-content")),
    ])

# --- Funnel Definitions (English Labels) ---
# These are used by the Funnels & Paths tab
FUNNEL_CONFIG_WHATSAPP = [
    {"label": "Site Visit (Page View)", "type": "event", "dimension": "eventName", "value": "page_view"},
    {"label": "Clicked WhatsApp", "type": "event", "dimension": "eventName", "value": "Clic_Whatsapp"} # Assuming 'Clic_Whatsapp' is the actual eventName from GA
]
FUNNEL_CONFIG_FORM = [
    {"label": "Site Visit (Page View)", "type": "event", "dimension": "eventName", "value": "page_view"},
    {"label": "Form Started", "type": "event", "dimension": "eventName", "value": "form_start"},
    {"label": "Form Submitted", "type": "event", "dimension": "eventName", "value": "Lleno Formulario"} # Assuming 'Lleno Formulario' is actual eventName
]
FUNNEL_CONFIG_CALLS = [
    {"label": "Site Visit (Page View)", "type": "event", "dimension": "eventName", "value": "page_view"},
    {"label": "Clicked Call Button", "type": "event", "dimension": "eventName", "value": "Clic_Boton_Llamanos"} # Assuming 'Clic_Boton_Llamanos' is actual eventName
]
KPI_EVENTS_FOR_EVOLUTION_CHART = ['Clic_Whatsapp', 'Lleno Formulario', 'Clic_Boton_Llamanos'] # Actual event names from GA

# --- Callbacks ---
def register_callbacks(app: dash.Dash) -> None:
    """Registers all callbacks for the Google Analytics page."""

    # Main callback to render content for each sub-tab
    @app.callback(
        Output('ga-subtabs-content', 'children'),
        Input('ga-subtabs', 'value'),
        State('ga-date-picker', 'start_date'),
        State('ga-date-picker', 'end_date')
    )
    def render_ga_subtab_content(subtab_value: str, start_date_str: str | None, end_date_str: str | None) -> html.Div:
        if not start_date_str or not end_date_str:
            return html.Div(html.P("Please select a date range to view Google Analytics data.", className="text-center mt-5 text-warning"))

        default_no_data_ai_text = "Insufficient data for AI analysis."

        # Overview Tab
        if subtab_value == 'overview':
            df_overview = ga_data.get_traffic_overview_df(start_date_str, end_date_str)
            if df_overview.empty:
                return html.Div([
                    html.P("No data available for GA Overview in the selected date range.", className="text-center mt-3"),
                    create_ai_insight_card('ga-overview-ai-card', title="🤖 AI Analysis: Overview"),
                    html.Div(default_no_data_ai_text, id='ga-overview-ai-data', style={'display': 'none'}),
                    create_ai_chat_interface('ga_overview_chat')
                ])

            if 'sessions' in df_overview.columns and 'conversions' in df_overview.columns:
                 df_overview['conversionRate'] = (df_overview['conversions'].fillna(0) / df_overview['sessions'].replace(0, np.nan).fillna(1) * 100).fillna(0)
            else:
                 df_overview['conversionRate'] = 0.0
            df_overview = df_overview.sort_values('date') if 'date' in df_overview else df_overview

            fig_sessions = get_fig_sessions_overview(df_overview)
            fig_users = get_fig_users_overview(df_overview)
            fig_conversions = get_fig_conversions_overview(df_overview)
            fig_conv_rate = get_fig_conversion_rate_overview(df_overview)
            fig_normalized = get_fig_normalized_trends_overview(df_overview)

            total_sessions = df_overview['sessions'].sum() if 'sessions' in df_overview else 0
            total_users = df_overview['activeUsers'].sum() if 'activeUsers' in df_overview else 0
            total_conversions = df_overview['conversions'].sum() if 'conversions' in df_overview else 0
            avg_conv_rate = df_overview['conversionRate'].mean() if 'conversionRate' in df_overview and not df_overview['conversionRate'].empty else 0.0

            context_overview = (
                f"GA Overview Summary: Total Sessions: {total_sessions:,.0f}. "
                f"Total Active Users: {total_users:,.0f}. "
                f"Total Conversions: {total_conversions:,.0f}. "
                f"Average Conversion Rate: {avg_conv_rate:.2f}%."
            )
            prompt_overview = "Analyze trends in sessions, users, conversions, and conversion rate. Provide a diagnosis and a powerful, actionable recommendation."
            ai_text_overview = get_openai_response(prompt_overview, context_overview) if not df_overview.empty else default_no_data_ai_text

            return html.Div([
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_sessions), md=6), dbc.Col(dcc.Graph(figure=fig_users), md=6)]),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_conversions), md=6), dbc.Col(dcc.Graph(figure=fig_conv_rate), md=6)], className="mt-3"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_normalized), md=12)], className="mt-3"),
                create_ai_insight_card('ga-overview-ai-card', title="💡 AI Diagnosis & Action (GA Overview)"),
                html.Div(ai_text_overview, id='ga-overview-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_overview_chat')
            ])

        # Demographics & Geo Tab
        elif subtab_value == 'demography':
            df_gender = ga_data.get_demographics_df(start_date_str, end_date_str, dimension='userGender')
            df_age = ga_data.get_demographics_df(start_date_str, end_date_str, dimension='userAgeBracket')
            df_country_users = ga_data.get_demographics_df(start_date_str, end_date_str, dimension='country') # Users by country
            df_city_users = ga_data.get_demographics_df(start_date_str, end_date_str, dimension='city') # Users by city

            df_geo_sessions_conv_city = ga_data.query_ga(metrics=['sessions', 'conversions'], dimensions=['city'], start_date=start_date_str, end_date=end_date_str)
            df_geo_sessions_conv_country = ga_data.query_ga(metrics=['sessions', 'conversions'], dimensions=['country'], start_date=start_date_str, end_date=end_date_str)

            graphs_content = []
            context_parts = []

            if not df_gender.empty:
                graphs_content.append(dbc.Col(dcc.Graph(figure=get_fig_users_by_gender(df_gender)), md=6))
                context_parts.append(f"Users by Gender: {df_gender.to_string(index=False)}")
            if not df_age.empty:
                graphs_content.append(dbc.Col(dcc.Graph(figure=get_fig_users_by_age(df_age)), md=6))
                context_parts.append(f"Users by Age: {df_age.to_string(index=False)}")
            if not df_country_users.empty:
                graphs_content.append(dbc.Col(dcc.Graph(figure=get_fig_top_countries_by_users(df_country_users)), md=6))
                context_parts.append(f"Top Countries by Users: {df_country_users.head().to_string(index=False)}")
            if not df_city_users.empty:
                graphs_content.append(dbc.Col(dcc.Graph(figure=get_fig_top_cities_by_users(df_city_users)), md=6))
                context_parts.append(f"Top Cities by Users: {df_city_users.head().to_string(index=False)}")

            geo_opp_layout_elements = [html.H4("Geo-Opportunities Analysis", className="mt-5 text-center")]

            if df_geo_sessions_conv_city.empty and df_geo_sessions_conv_country.empty:
                geo_opp_layout_elements.append(html.P("No geographic data for opportunity analysis.", className="text-center"))
            else:
                session_threshold = 10
                if 'sessions' in df_geo_sessions_conv_city and df_geo_sessions_conv_city['sessions'].notna().any():
                    session_threshold = max(10, df_geo_sessions_conv_city['sessions'].quantile(0.70))

                df_opportunities_city = pd.DataFrame()
                if 'sessions' in df_geo_sessions_conv_city and 'conversions' in df_geo_sessions_conv_city:
                    df_opportunities_city = df_geo_sessions_conv_city[(df_geo_sessions_conv_city['sessions'].fillna(0) >= session_threshold) & (df_geo_sessions_conv_city['conversions'].fillna(0) == 0)].copy()

                fig_geo_map = create_empty_figure("Geo Opportunities Map (No country data with 0 conversions)")
                if not df_geo_sessions_conv_country.empty and 'conversions' in df_geo_sessions_conv_country :
                    df_map_data = df_geo_sessions_conv_country[df_geo_sessions_conv_country['conversions'].fillna(0) == 0]
                    if not df_map_data.empty:
                         fig_geo_map = get_fig_geo_opportunity_map(df_map_data)


                table_geo_opp_city_content = html.P(f"No clear geographic opportunities found (cities with >= {session_threshold:.0f} sessions and 0 conversions).", className="text-center")
                if not df_opportunities_city.empty:
                    df_opportunities_city_display = df_opportunities_city.sort_values(by='sessions', ascending=False).head(15)
                    table_geo_opp_city_content = dash_table.DataTable(
                        data=df_opportunities_city_display.to_dict('records'),
                        columns=[{'name': 'City', 'id': 'city'}, {'name': 'Sessions (0 conv.)', 'id': 'sessions'}],
                        style_table={'overflowX': 'auto', 'marginTop': '20px', 'marginBottom': '20px'}, page_size=10,
                        sort_action='native', filter_action='native'
                    )
                    context_parts.append(f"Geo-Opportunities (Cities with sessions >= {session_threshold:.0f}, 0 conversions): {df_opportunities_city_display.head(3).to_string(index=False)}")

                geo_explanation_text = f"""
                **What is this Geo-Opportunities Map/Table?** It identifies countries and cities that generate significant traffic (sessions)
                but do not result in any recorded conversions for the selected period.
                **Potential Actions:** Investigate marketing campaigns targeting these areas, review content localization,
                or analyze product/service suitability for these markets. Threshold for 'significant traffic' is >= {session_threshold:.0f} sessions.
                """
                geo_opp_layout_elements.extend([
                    dbc.Card(dbc.CardBody(dcc.Markdown(geo_explanation_text)), color="info", outline=True, className="mb-3 mt-3"),
                    dcc.Graph(id='ga-geo-opportunity-map', figure=fig_geo_map),
                    html.H5(f"Top Cities with Opportunities (Sessions >= {session_threshold:.0f}, Conversions = 0)", className="mt-4 text-center"),
                    table_geo_opp_city_content
                ])

            ai_text_demog = default_no_data_ai_text
            if context_parts:
                context_demog = "\n".join(context_parts)
                prompt_demog = "Analyze demographic data (gender, age) AND geo-opportunities (traffic without conversion by country/city). Which segments stand out or might be underserved/misfocused? Provide a combined diagnosis and a powerful, actionable recommendation."
                ai_text_demog = get_openai_response(prompt_demog, context_demog)

            return html.Div([
                html.H4("Demographic Analysis & Segmentation 🌍📍", className="text-center mt-4"),
                dbc.Row(graphs_content) if graphs_content else html.P("No sufficient demographic data to display.", className="text-center"),
                html.Hr(className="my-4"),
                html.Div(geo_opp_layout_elements),
                create_ai_insight_card('ga-demography-ai-card', title="💡 AI Diagnosis & Action (Demographics & Geo)"),
                html.Div(ai_text_demog, id='ga-demography-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_demography_chat')
            ])

        # Funnels & Paths Tab
        elif subtab_value == 'funnels':
            df_events = ga_data.get_events_df(start_date_str, end_date_str, event_names=KPI_EVENTS_FOR_EVOLUTION_CHART)
            kpi_table_content = html.P("No event data for KPIs.", className="text-center")
            fig_event_evol = create_empty_figure("Event Conversion Evolution")
            if not df_events.empty and 'date' in df_events.columns:
                df_events_pivot = df_events.pivot_table(index="date", columns="eventName", values="eventCount", aggfunc='sum').fillna(0).reset_index()
                for col in KPI_EVENTS_FOR_EVOLUTION_CHART:
                    if col not in df_events_pivot.columns: df_events_pivot[col] = 0

                event_totals = {col: int(df_events_pivot[col].sum()) for col in KPI_EVENTS_FOR_EVOLUTION_CHART if col in df_events_pivot}
                kpi_table_rows = [html.Tr([html.Td(k), html.Td(f"{v:,.0f}")]) for k,v in event_totals.items()]
                if kpi_table_rows:
                    kpi_table_content = dbc.Table([html.Thead(html.Tr([html.Th("Event Channel"), html.Th("Total Conversions")])),
                                           html.Tbody(kpi_table_rows)], bordered=True, hover=True, striped=True, className="mt-2")
                fig_event_evol = get_fig_event_evolution(df_events_pivot, KPI_EVENTS_FOR_EVOLUTION_CHART)

            df_acq_channels = ga_data.get_top_channels_df(start_date_str, end_date_str)
            fig_acq_channels_chart = get_fig_acquisition_channels(df_acq_channels)

            df_page_perf = ga_data.get_page_path_performance_df(start_date_str, end_date_str)
            fig_page_visits_chart = get_fig_page_visits(df_page_perf)
            fig_page_bounce_chart = get_fig_page_bounce_rate(df_page_perf)
            fig_page_duration_chart = get_fig_page_duration(df_page_perf)

            labels_w, counts_w = ga_data.get_funnel_plot_lists(FUNNEL_CONFIG_WHATSAPP, start_date_str, end_date_str)
            labels_f, counts_f = ga_data.get_funnel_plot_lists(FUNNEL_CONFIG_FORM, start_date_str, end_date_str)
            labels_l, counts_l = ga_data.get_funnel_plot_lists(FUNNEL_CONFIG_CALLS, start_date_str, end_date_str)

            fig_funnel_whatsapp = get_fig_funnel_chart(labels_w, counts_w, "WhatsApp Funnel")
            fig_funnel_form = get_fig_funnel_chart(labels_f, counts_f, "Form Submission Funnel")
            fig_funnel_calls = get_fig_funnel_chart(labels_l, counts_l, "Call Funnel")

            total_initial_visits = counts_w[0] if counts_w and len(counts_w)>0 else 0
            total_final_conversions = (counts_w[-1] if len(counts_w) == len(FUNNEL_CONFIG_WHATSAPP) and counts_w else 0) + \
                                      (counts_f[-1] if len(counts_f) == len(FUNNEL_CONFIG_FORM) and counts_f else 0) + \
                                      (counts_l[-1] if len(counts_l) == len(FUNNEL_CONFIG_CALLS) and counts_l else 0)
            fig_total_funnel_chart = get_fig_funnel_chart(
                ["Total Initial Visits (Funnel Start)", "Total Final Conversions (Funnel End)"],
                [total_initial_visits, total_final_conversions],
                "Overall Funnel Conversion"
            )

            funnels_layout_section = html.Div([
                html.H4("Acquisition, KPIs & Evolution", className="mt-4 text-center"),
                dbc.Row([dbc.Col(kpi_table_content, width=12, lg=4), dbc.Col(dcc.Graph(figure=fig_event_evol), width=12, lg=8)]),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_acq_channels_chart), width=12, lg=10)], className="mt-4", justify="center"),
                html.Hr(), html.H4("Page Behavior", className="mt-4 text-center"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_page_visits_chart), width=12, lg=4),
                    dbc.Col(dcc.Graph(figure=fig_page_bounce_chart), width=12, lg=4),
                    dbc.Col(dcc.Graph(figure=fig_page_duration_chart), width=12, lg=4)
                ]),
                html.Hr(), html.H4("Specific & Overall Funnels", className="mt-4 text-center"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_funnel_whatsapp), width=12, lg=4),
                    dbc.Col(dcc.Graph(figure=fig_funnel_form), width=12, lg=4),
                    dbc.Col(dcc.Graph(figure=fig_funnel_calls), width=12, lg=4)
                ]),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_total_funnel_chart), width=12, lg=8, className="mx-auto mt-3")])
            ])

            sankey_key_events = ['page_view', 'form_start', 'Clic_Whatsapp', 'Lleno Formulario', 'Clic_Boton_Llamanos']
            df_sankey_source = ga_data.query_ga(metrics=['sessions', 'eventCount'], dimensions=['sessionSourceMedium', 'eventName'], start_date=start_date_str, end_date=end_date_str)
            fig_sankey_chart = get_fig_sankey_user_flow(df_sankey_source, sankey_key_events)

            sankey_explanation_text = """
            **Interpreting the Sankey Diagram:** This chart visualizes user flows from traffic sources/mediums to key website events.
            Thicker lines indicate more common paths. It helps identify which channels effectively drive desired actions.
            **Limitation:** This is a simplified flow model and does not represent strict sequential user pathing for all users.
            """
            sankey_context_for_ai = "Sankey diagram data not available for AI context."
            if not df_sankey_source.empty and not df_sankey_source[df_sankey_source['eventName'].isin(sankey_key_events)].empty:
                sankey_context_for_ai = (
                    f"Sankey Diagram shows flows from "
                    f"'{df_sankey_source['sessionSourceMedium'].nunique()}' sources/mediums to "
                    f"'{df_sankey_source[df_sankey_source['eventName'].isin(sankey_key_events)]['eventName'].nunique()}' key events. "
                    f"Top contributing sources: {df_sankey_source.groupby('sessionSourceMedium')['sessions'].sum().nlargest(3).to_string()}."
                )

            sankey_layout_section = html.Div([
                html.H4("User Flow Analysis (Sankey Diagram)", className="mt-5 text-center"),
                dbc.Card(dbc.CardBody(dcc.Markdown(sankey_explanation_text)), color="info", outline=True, className="mb-3 mt-3"),
                dcc.Graph(id='ga-sankey-graph', figure=fig_sankey_chart),
            ])

            ai_text_funnels = default_no_data_ai_text
            if total_initial_visits > 0 or (not df_sankey_source.empty and not df_sankey_source[df_sankey_source['eventName'].isin(sankey_key_events)].empty) :
                context_funnels = (
                    f"Funnel Data: WhatsApp counts {counts_w}, Form counts {counts_f}, Call counts {counts_l}. "
                    f"Overall Conversion: Initial Visits={total_initial_visits}, Final Conversions={total_final_conversions}. "
                    f"{sankey_context_for_ai}"
                )
                prompt_funnels = "Analyze the performance of conversion funnels AND user paths from the Sankey diagram. Identify the main bottleneck in the funnels and the most important (or inefficient) user paths in the Sankey. Provide a combined diagnosis and a powerful, actionable recommendation to improve overall conversion and path efficiency."
                ai_text_funnels = get_openai_response(prompt_funnels, context_funnels)

            return html.Div([
                funnels_layout_section,
                html.Hr(className="my-4"),
                sankey_layout_section,
                create_ai_insight_card('ga-funnels-ai-card', title="💡 AI Diagnosis & Action (Funnels & Paths)"),
                html.Div(ai_text_funnels, id='ga-funnels-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_funnels_chat')
            ])

        # What If Simulator Tab
        elif subtab_value == 'what_if':
            ai_text_what_if = "Adjust the sliders to simulate scenarios and see the AI analysis."
            return html.Div([
                html.H4('Scenario Simulator "What If" 🧪', className="mt-4 text-center"),
                dbc.Row([
                    dbc.Col([html.Label("Increase % in Total Sessions:", className="form-label"),
                             dcc.Slider(id='ga-what-if-sessions-slider', min=0, max=100, step=5, value=0, marks={i: f'{i}%' for i in range(0, 101, 20)}, tooltip={"placement": "bottom", "always_visible": True})], md=6, className="mb-3"),
                    dbc.Col([html.Label("Change % in Overall Conversion Rate:", className="form-label"),
                             dcc.Slider(id='ga-what-if-cr-slider', min=-50, max=50, step=5, value=0, marks={i: f'{i}%' for i in range(-50, 51, 25)}, tooltip={"placement": "bottom", "always_visible": True})], md=6, className="mb-3"),
                ]),
                dbc.Button("Simulate Scenario", id="ga-what-if-simulate-button", color="primary", className="mt-3 mb-3"),
                html.Div(id='ga-what-if-results-display'),
                create_ai_insight_card('ga-what-if-ai-card', title="💡 Scenario Interpretation & Suggestions"),
                html.Div(ai_text_what_if, id='ga-what-if-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_what_if_chat')
            ])

        # Temporal Analysis Tab
        elif subtab_value == 'temporal':
            df_temporal_src = ga_data.get_traffic_overview_df(start_date_str, end_date_str)

            fig_temporal_chart = create_empty_figure('Temporal Decomposition & Anomalies (Insufficient data)')
            ai_text_temporal = "At least 14 days of data are needed for meaningful temporal analysis."

            if not df_temporal_src.empty and 'date' in df_temporal_src.columns and 'sessions' in df_temporal_src.columns and len(df_temporal_src) >= 14 :
                series_for_decomp = df_temporal_src.set_index(pd.to_datetime(df_temporal_src['date'])).sort_index()['sessions'].asfreq('D').fillna(0)
                if len(series_for_decomp) >= 14:
                    period = min(7, len(series_for_decomp) // 2 if len(series_for_decomp) // 2 > 0 else 1)
                    fig_temporal_chart = get_fig_temporal_decomposition(series_for_decomp, period=period)

                    if fig_temporal_chart.layout.title.text != DEFAULT_NO_DATA_FIGURE_TITLE and not "Error" in fig_temporal_chart.layout.title.text :
                        context_temporal_data = f"Temporal decomposition analysis performed on daily sessions. Period used: {period} days."
                        prompt_temporal_analysis = "Diagnose patterns in trend, seasonality, and any anomalies found in daily sessions. Suggest a powerful, actionable recommendation based on these findings."
                        ai_text_temporal = get_openai_response(prompt_temporal_analysis, context_temporal_data)
                    else:
                         ai_text_temporal = f"Could not perform temporal decomposition. Figure title: {fig_temporal_chart.layout.title.text}"
                else:
                    ai_text_temporal = "Not enough contiguous daily data points after resampling for temporal analysis (need at least 14)."

            return html.Div([
                dcc.Graph(id='ga-temporal-graph', figure=fig_temporal_chart),
                create_ai_insight_card('ga-temporal-ai-card', title="💡 AI Diagnosis & Action (Temporal Analysis)"),
                html.Div(ai_text_temporal, id='ga-temporal-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_temporal_chat')
            ])

        # Correlations Tab
        elif subtab_value == 'correlations':
            df_corr_src = ga_data.get_correlation_data_df(start_date_str, end_date_str)
            df_age_conv_src = ga_data.get_demographics_df(start_date_str, end_date_str, dimension='userAgeBracket')

            metrics_for_scatter = ['sessions', 'activeUsers', 'averageSessionDuration', 'bounceRate', 'conversions']
            fig_scatter_matrix = get_fig_correlation_scatter_matrix(df_corr_src, metrics_for_scatter, 'deviceCategory')
            fig_boxplot_device = get_fig_conversions_boxplot(df_corr_src, 'deviceCategory', 'Device Category')
            fig_boxplot_age = get_fig_conversions_boxplot(df_age_conv_src, 'userAgeBracket', 'Age Bracket')

            ai_text_corr = default_no_data_ai_text
            corr_matrix_summary_for_ai = "Correlation matrix data not available."
            if not df_corr_src.empty and all(m in df_corr_src for m in metrics_for_scatter):
                try:
                    numeric_cols_for_corr = df_corr_src[metrics_for_scatter].select_dtypes(include=np.number).columns.tolist()
                    if len(numeric_cols_for_corr) >=2:
                         corr_matrix_summary_for_ai = df_corr_src[numeric_cols_for_corr].corr(numeric_only=True).to_string()
                except Exception as e:
                    logger.error(f"Error calculating correlation matrix for AI context: {e}")
                    corr_matrix_summary_for_ai = "Error calculating correlation matrix."

            context_correlation = (
                f"Correlation Matrix Summary:\n{corr_matrix_summary_for_ai}\n"
                f"Also consider boxplots of conversions by device and age."
            )
            prompt_correlation = "Identify strong correlations or significant differences in conversions by group (device, age). Diagnose these patterns and suggest a powerful, actionable recommendation."
            ai_text_corr = get_openai_response(prompt_correlation, context_correlation)

            return html.Div([
                dcc.Graph(id='ga-correlation-scatter-matrix', figure=fig_scatter_matrix),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_boxplot_device), md=6),
                    dbc.Col(dcc.Graph(figure=fig_boxplot_age), md=6)
                ], className="mt-3"),
                create_ai_insight_card('ga-correlations-ai-card', title="💡 AI Diagnosis & Action (Correlations)"),
                html.Div(ai_text_corr, id='ga-correlations-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_correlations_chat')
            ])

        # Cohort Analysis Tab
        elif subtab_value == 'cohort':
            df_cohort_src = ga_data.get_cohorts_df(start_date_str, end_date_str)

            fig_cohort_heatmap = create_empty_figure("Cohort Analysis – User Retention (%) (Insufficient data)")
            cohort_explanation_text = "Insufficient data for cohort analysis. Requires at least two cohorts with retention data."
            ai_text_cohort = cohort_explanation_text

            if not df_cohort_src.empty and 'firstSessionDate' in df_cohort_src and 'nthDay' in df_cohort_src and 'activeUsers' in df_cohort_src and len(df_cohort_src['firstSessionDate'].unique()) >= 2:
                try:
                    cohort_pivot = df_cohort_src.pivot_table(index='firstSessionDate', columns='nthDay', values='activeUsers')
                    if not cohort_pivot.empty and not cohort_pivot.iloc[:, 0].empty:
                        cohort_pivot = cohort_pivot.sort_index(ascending=False).reindex(sorted(cohort_pivot.columns), axis=1)
                        cohort_sizes = cohort_pivot.iloc[:, 0]

                        retention_matrix = cohort_pivot.apply(lambda row: (row / cohort_sizes.loc[row.name] * 100) if cohort_sizes.loc[row.name] > 0 else 0.0, axis=1).fillna(0.0)
                        retention_matrix_display = retention_matrix.head(15).iloc[:, :15]

                        if not retention_matrix_display.empty:
                            fig_cohort_heatmap = get_fig_cohort_retention_heatmap(retention_matrix_display)
                            cohort_explanation_text = """
                            **Interpreting Cohort Analysis:** This groups users by their first visit date (cohort) and tracks their retention over subsequent days/weeks.
                            It helps understand how well you retain users and the impact of changes over time.
                            Rows = Cohorts (first visit date), Columns = Time since first visit (e.g., Day 0, Day 1...), Color/Number = Retention Percentage.
                            """

                            avg_ret_day1 = retention_matrix_display.iloc[:, 1].mean() if len(retention_matrix_display.columns) > 1 else 'N/A'
                            avg_ret_day7 = retention_matrix_display.iloc[:, 7].mean() if len(retention_matrix_display.columns) > 7 else 'N/A'
                            context_cohort_data = (
                                f"Cohort Analysis Summary: Average Day 1 Retention: {avg_ret_day1 if isinstance(avg_ret_day1, str) else avg_ret_day1:.1f}% (if available). "
                                f"Average Day 7 Retention: {avg_ret_day7 if isinstance(avg_ret_day7, str) else avg_ret_day7:.1f}% (if available)."
                            )
                            prompt_cohort_analysis = "Analyze the user retention trend from the cohort data. Does any cohort stand out (positively or negatively)? Are there general patterns? Diagnose these findings and suggest a powerful, actionable recommendation to improve user retention."
                            ai_text_cohort = get_openai_response(prompt_cohort_analysis, context_cohort_data)
                        else:
                            ai_text_cohort = "Cohort retention matrix is empty after processing."
                    else:
                        ai_text_cohort = "Could not build pivot table for cohort analysis (e.g. no Day 0 data)."
                except Exception as e:
                    logger.error(f"Error in cohort analysis processing: {e}", exc_info=True)
                    ai_text_cohort = f"Error processing data for cohort analysis: {str(e)}"

            return html.Div([
                dbc.Card(dbc.CardBody(dcc.Markdown(cohort_explanation_text)), color="info", outline=True, className="mb-3"),
                dcc.Graph(id='ga-cohort-heatmap', figure=fig_cohort_heatmap),
                create_ai_insight_card('ga-cohort-ai-card', title="💡 AI Diagnosis & Action (Cohort Analysis)"),
                html.Div(ai_text_cohort, id='ga-cohort-ai-data', style={'display': 'none'}),
                create_ai_chat_interface('ga_cohort_chat')
            ])

        return html.Div(html.P(f"Google Analytics Sub-tab '{subtab_value}' not implemented or data unavailable.", className="text-center mt-5 text-danger"))

    ga_subtabs_for_ai_cards = ['overview', 'demography', 'funnels', 'what_if', 'temporal', 'correlations', 'cohort']
    for tab_key in ga_subtabs_for_ai_cards:
        @app.callback(
            Output(f'ga-{tab_key}-ai-card', 'children'),
            Input(f'ga-{tab_key}-ai-data', 'children'),
        )
        def update_specific_ga_ai_card(ai_text_content: str | None, card_id_suffix=tab_key):
            # card_id_suffix helps differentiate callbacks if needed, not strictly used here but good practice for complex scenarios
            default_msg = "AI analysis will appear here once data is processed."
            no_data_indicators = [
                "Insufficient data for AI analysis.", "No data available for GA Overview",
                "No geographic data for opportunity analysis.", "No event data for KPIs.",
                "Sankey diagram data not available", "At least 14 days of data are needed",
                "Could not perform temporal decomposition", "Correlation matrix data not available.",
                "Insufficient data for cohort analysis.", "Cohort retention matrix is empty",
                "Could not build pivot table for cohort analysis", "Error processing data for cohort analysis",
                "Adjust the sliders to simulate scenarios"
            ]
            if not ai_text_content or any(indicator in ai_text_content for indicator in no_data_indicators):
                return html.P(default_msg, className="text-muted")
            return html.P(ai_text_content)

    @app.callback(
        Output('ga-what-if-results-display', 'children'),
        Output('ga-what-if-ai-data', 'children'),
        Input('ga-what-if-simulate-button', 'n_clicks'),
        State('ga-date-picker', 'start_date'),
        State('ga-date-picker', 'end_date'),
        State('ga-what-if-sessions-slider', 'value'),
        State('ga-what-if-cr-slider', 'value'),
        prevent_initial_call=True
    )
    def handle_what_if_simulation(
        n_clicks: int | None,
        start_date_str: str | None, end_date_str: str | None,
        sessions_increase_pct: float, cr_change_pct: float
    ) -> tuple[html.Div, str]:

        if not n_clicks or not start_date_str or not end_date_str :
            return html.Div(html.P("Click 'Simulate Scenario' after selecting dates and adjusting sliders.", className="text-info")), "Adjust sliders and click simulate."

        df_baseline_metrics = ga_data.get_baseline_metrics_df(start_date_str, end_date_str)

        if df_baseline_metrics.empty or 'sessions' not in df_baseline_metrics.columns or 'conversions' not in df_baseline_metrics.columns:
            return html.Div(html.P("Could not fetch baseline data for simulation. Ensure GA is connected.", className="text-danger")), "Baseline data unavailable."

        baseline_sessions = df_baseline_metrics['sessions'].iloc[0]
        baseline_conversions = df_baseline_metrics['conversions'].iloc[0]

        if baseline_sessions == 0:
            return html.Div(html.P("Baseline sessions are zero. Cannot simulate meaningful impact.", className="text-warning")), "Baseline sessions are zero."

        baseline_cr_pct = (baseline_conversions / baseline_sessions * 100) if baseline_sessions > 0 else 0
        
        new_sessions = baseline_sessions * (1 + sessions_increase_pct / 100)
        new_cr_abs_pct = max(0, min(baseline_cr_pct * (1 + cr_change_pct / 100), 100))
        predicted_conversions = new_sessions * (new_cr_abs_pct / 100)

        change_in_conversions = predicted_conversions - baseline_conversions
        pct_change_in_conversions = ((predicted_conversions / baseline_conversions - 1) * 100) if baseline_conversions > 0 else float('inf') if predicted_conversions > 0 else 0

        results_card_content = dbc.CardBody([
            html.H5("Simulation Results", className="card-title text-primary"),
            dbc.Row([
                dbc.Col([
                    html.H6("Baseline:"),
                    html.P(f"Sessions: {baseline_sessions:,.0f}"),
                    html.P(f"Conversion Rate: {baseline_cr_pct:.2f}%"),
                    html.P(f"Conversions: {baseline_conversions:,.0f}"),
                ], md=6),
                dbc.Col([
                    html.H6("Projected Scenario:"),
                    html.P(f"Sessions: {new_sessions:,.0f} ({sessions_increase_pct:+}%)"),
                    html.P(f"Conversion Rate: {new_cr_abs_pct:.2f}% ({cr_change_pct:+}% relative change applied)"),
                    html.P(f"Conversions: {predicted_conversions:,.0f}"),
                ], md=6),
            ]),
            html.Hr(),
            html.P(f"Change in Conversions: {change_in_conversions:,.0f}", className="fw-bold mt-2"),
            html.P(f"Percentage Change in Conversions: {pct_change_in_conversions:.2f}%" if baseline_conversions > 0 and pct_change_in_conversions != float('inf') else "N/A (baseline conversions were zero or change is infinite)", className="fw-bold")
        ])

        context_what_if_sim = (
            f"Simulation Input: Baseline Sessions={baseline_sessions:,.0f}, Baseline CR={baseline_cr_pct:.2f}%, Baseline Conversions={baseline_conversions:,.0f}. "
            f"Applied Changes: Session Increase={sessions_increase_pct}%, CR Relative Change={cr_change_pct}%. "
            f"Projected Scenario: Sessions={new_sessions:,.0f}, CR={new_cr_abs_pct:.2f}%, Conversions={predicted_conversions:,.0f}."
        )
        prompt_what_if_sim = "Interpret this 'What If' scenario. Discuss its realism, the primary impact (benefits/risks), and suggest one key action to try and achieve the projected positive outcome, or mitigate risks if negative."
        ai_interpretation_text = get_openai_response(prompt_what_if_sim, context_what_if_sim)

        return html.Div(dbc.Card(results_card_content, className="mt-3 shadow-sm")), ai_interpretation_text

    ga_chat_subtab_ids = ['ga_overview_chat', 'ga_demography_chat', 'ga_funnels_chat', 'ga_what_if_chat', 'ga_temporal_chat', 'ga_correlations_chat', 'ga_cohort_chat']
    for chat_tab_prefix in ga_chat_subtab_ids:
        @app.callback(
            Output(f'{chat_tab_prefix}-chat-history', 'children'),
            Input(f'{chat_tab_prefix}-chat-submit', 'n_clicks'),
            State(f'{chat_tab_prefix}-chat-input', 'value'),
            State(f'{chat_tab_prefix}-chat-history', 'children'),
            State('ga-subtabs', 'value'),
            prevent_initial_call=True,
        )
        def update_ga_chat_interface(
            n_clicks: int | None, user_input: str | None,
            chat_history: list | html.Div | None, active_ga_subtab: str,
            # To make the callback unique for each chat instance, use its prefix
            # This is a simplified way; for very complex apps, consider function factories or more explicit IDing
            # Dash's default behavior should handle this as long as Input/Output IDs are unique.
            # The `current_chat_prefix` is implicitly the one that triggered this instance.
            # We get it from the list `ga_chat_subtab_ids` used to generate these.
            # This is a common pattern, but can be tricky. Let's make it explicit.
            # This callback is defined inside a loop, so we need to ensure `chat_tab_prefix` is captured correctly.
            # The default argument trick is a common way:
            # callback_chat_prefix=chat_tab_prefix
        ):
            # The loop variable `chat_tab_prefix` is not directly available in the callback's scope
            # in the way one might expect due to Python's late binding in closures.
            # However, Dash's callback mechanism might handle this by creating distinct callbacks
            # if the Input/Output objects are distinct.
            # For robustness, it's better to use `dash.callback_context` or ensure distinct function names
            # or use a helper that generates these callbacks.
            # For now, let's assume the active_ga_subtab gives enough context.

            triggered_input_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0] # e.g. 'ga_overview_chat-chat-submit'
            current_chat_prefix_from_ctx = triggered_input_id.replace('-chat-submit', '')


            if not n_clicks or not user_input:
                return chat_history or []

            current_history = list(chat_history) if isinstance(chat_history, list) else ([chat_history] if chat_history else [])

            context_tab_name = current_chat_prefix_from_ctx.replace('_chat','').replace('ga_','') # e.g. 'overview'

            context_for_ai = f"User is on the Google Analytics '{context_tab_name}' sub-tab (full active sub-tab ID: '{active_ga_subtab}'). User asks: {user_input}"
            ai_response_text = get_openai_response(user_input, context_for_ai)

            new_entry_user = html.P([html.B("You: ", style={'color': '#007bff'}), user_input], style={'margin': '5px 0'})
            new_entry_ai = html.P([html.B("Omesh AI: ", style={'color': '#28a745'}), ai_response_text], style={'background': '#f0f0f0', 'padding': '8px', 'borderRadius': '5px', 'margin': '5px 0'})

            current_history.extend([new_entry_user, new_entry_ai])
            return current_history

```
