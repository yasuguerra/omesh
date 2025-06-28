from __future__ import annotations
import logging
from dash import dcc, html, Input, Output, dash_table, State, callback # Added State and callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import typing

# Project dependencies
from ..ui.components import create_sales_dashboard_layout # Changed from create_ops_sales_layout
from ..ai import get_openai_response
from ..data.data_processing import unify_data, clean_df, safe_sorted_unique

logger = logging.getLogger(__name__)

# Standard margin for all figures
FIGURE_MARGIN = dict(t=40, l=20, r=20, b=40)

# Helper function to create an empty figure
def create_empty_figure(title: str = "No data to display") -> go.Figure:
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
def get_fig_flights_by_month(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Flights by Month")
    data = df_plot.groupby(['Año', 'MonthName'], observed=False).size().reset_index(name='Flights')
    fig = px.line(data, x='MonthName', y='Flights', color='Año', markers=True, title='Flights by Month')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Month", yaxis_title="Number of Flights")
    return fig

def get_fig_revenue_by_month(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Revenue by Month")
    data = df_plot.groupby(['Año', 'MonthName'], observed=False)['Monto total a cobrar'].sum().reset_index()
    fig = px.line(data, x='MonthName', y='Monto total a cobrar', color='Año', markers=True, title='Revenue by Month')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Month", yaxis_title="Total Revenue")
    return fig

def get_fig_profit_by_month(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Profit by Month")
    data = df_plot.groupby(['Año', 'MonthName'], observed=False)['Ganancia'].sum().reset_index()
    fig = px.line(data, x='MonthName', y='Ganancia', color='Año', markers=True, title='Profit by Month')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Month", yaxis_title="Total Profit")
    return fig

def get_fig_total_monthly_profit_timeline(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Total Monthly Profit + Trend")
    data = df_plot.groupby('year_month')['Ganancia'].sum().reset_index().sort_values('year_month')
    fig = go.Figure(go.Scatter(x=data['year_month'], y=data['Ganancia'], mode='lines+markers', name='Monthly Profit'))
    if data.shape[0] > 1:
        x_fit = np.arange(len(data)); y_fit = data['Ganancia'].values
        try:
            z = np.polyfit(x_fit, y_fit, 1); p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=data['year_month'], y=p(x_fit), mode='lines', name='Trend', line=dict(dash='dash')))
        except Exception as e:
            logger.warning(f"Could not compute trendline for total monthly profit: {e}")
    fig.update_layout(title='Total Monthly Profit + Trend', xaxis_title='Month', yaxis_title='Profit', xaxis=dict(tickangle=45), margin=FIGURE_MARGIN)
    return fig

def get_fig_total_monthly_operations_timeline(df_plot: pd.DataFrame, meses_order: list) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Total Monthly Operations + Trend")
    data = df_plot.groupby('MonthName', observed=False).size().reset_index(name='Flights')
    data['MonthName_cat'] = pd.Categorical(data['MonthName'], categories=meses_order, ordered=True)
    data = data.sort_values('MonthName_cat')
    fig = go.Figure(go.Scatter(x=data['MonthName'], y=data['Flights'], mode='lines+markers', name='Monthly Operations'))
    if data.shape[0] > 1:
        y_fit_ops = data['Flights'].values; x_fit_ops = np.arange(len(y_fit_ops))
        poly_degree_ops = min(2, len(y_fit_ops)-1) if len(y_fit_ops) > 2 else 1
        try:
            z_ops = np.polyfit(x_fit_ops, y_fit_ops, poly_degree_ops); p_ops = np.poly1d(z_ops)
            fig.add_trace(go.Scatter(x=data['MonthName'], y=p_ops(x_fit_ops), mode='lines', name='Trend', line=dict(dash='dash')))
        except Exception as e:
            logger.warning(f"Could not compute trendline for total monthly operations: {e}")
    fig.update_layout(title='Total Monthly Operations + Trend', margin=FIGURE_MARGIN, xaxis_title="Month", yaxis_title="Number of Flights")
    return fig

def get_fig_weekly_timeseries(df_plot: pd.DataFrame, column: str, title: str) -> go.Figure:
    if df_plot.empty: return create_empty_figure(title)
    fig = go.Figure()
    for year_val_unique in df_plot['Año'].unique():
        df_year_ts = df_plot[df_plot['Año'] == year_val_unique].set_index('Fecha y hora del vuelo').sort_index()
        if not df_year_ts.empty:
            if column == 'Vuelos': # Special case for counting flights
                resampled_data = df_year_ts.resample('W').size().reset_index(name=column)
            else:
                resampled_data = df_year_ts[column].resample('W').sum().reset_index()

            fig.add_scatter(x=resampled_data['Fecha y hora del vuelo'], y=resampled_data[column], mode='lines', name=f'Year {year_val_unique}')
    fig.update_layout(title=title, margin=FIGURE_MARGIN, xaxis_title="Week", yaxis_title=column)
    return fig

def get_fig_top_destinations_by_flights(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Top Destinations by Number of Flights")
    data = df_plot.groupby(['Año', 'Destino'], observed=False).size().reset_index(name='Count').sort_values(['Count'], ascending=False)
    fig = px.bar(data, x='Destino', y='Count', color='Año', barmode='group', title='Top Destinations by Number of Flights')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Destination", yaxis_title="Number of Flights")
    return fig

def get_fig_top_destinations_by_profit(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Top Destinations by Profit")
    data = df_plot.groupby(['Año', 'Destino'], observed=False)['Ganancia'].sum().reset_index().sort_values(['Ganancia'], ascending=False)
    fig = px.bar(data, x='Destino', y='Ganancia', color='Año', barmode='group', title='Top Destinations by Profit')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Destination", yaxis_title="Total Profit")
    return fig

def get_fig_top_destinations_by_passengers(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Top Destinations by Passengers")
    data = df_plot.groupby(['Año', 'Destino'], observed=False)['Número de pasajeros'].sum().reset_index().sort_values(['Número de pasajeros'], ascending=False)
    fig = px.bar(data, x='Destino', y='Número de pasajeros', color='Año', barmode='group', title='Top Destinations by Passengers')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Destination", yaxis_title="Number of Passengers")
    return fig

def get_fig_flights_per_operator(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Flights per Operator")
    data = df_plot.groupby(['Año', 'Operador'], observed=False).size().reset_index(name='Flights').sort_values(['Año', 'Flights'], ascending=[True, False])
    fig = px.bar(data, x='Operador', y='Flights', color='Año', barmode='group', title='Flights per Operator')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Operator", yaxis_title="Number of Flights")
    return fig

def get_fig_profit_by_aircraft(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Profit by Aircraft")
    data = df_plot.groupby(['Año', 'Aeronave'], observed=False)['Ganancia'].sum().reset_index().sort_values(['Año', 'Ganancia'], ascending=[True, False])
    fig = px.bar(data, x='Aeronave', y='Ganancia', color='Año', barmode='group', title='Profit by Aircraft')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Aircraft", yaxis_title="Total Profit")
    return fig

def get_fig_top_operators_by_total_profit(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Top Operators by Total Profit")
    data = df_plot.groupby('Operador', observed=False)['Ganancia'].sum().reset_index().sort_values('Ganancia', ascending=False)
    fig = px.bar(data, x='Operador', y='Ganancia', title="Top Operators by Total Profit")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Operator", yaxis_title="Total Profit")
    return fig

def get_fig_top_aircraft_by_total_profit(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Top Aircraft by Total Profit")
    data = df_plot.groupby('Aeronave', observed=False)['Ganancia'].sum().reset_index().sort_values('Ganancia', ascending=False)
    fig = px.bar(data, x='Aeronave', y='Ganancia', title="Top Aircraft by Total Profit")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Aircraft", yaxis_title="Total Profit")
    return fig

def get_fig_average_ticket_by_destination_year(df_plot: pd.DataFrame) -> go.Figure:
    if df_plot.empty: return create_empty_figure("Average Ticket Price by Destination and Year")
    data = df_plot.groupby(['Año', 'Destino'], observed=False)['Monto total a cobrar'].mean().reset_index()
    fig = px.bar(data, x='Destino', y='Monto total a cobrar', color='Año', barmode='group', title='Average Ticket Price by Destination and Year')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Destination", yaxis_title="Average Ticket Price")
    return fig

def get_fig_heatmap(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str, y_order: list | None = None) -> go.Figure:
    if data.empty: return create_empty_figure(title)
    if y_order and y_col in data.columns:
        data[y_col] = pd.Categorical(data[y_col], categories=y_order, ordered=True)
    fig = px.density_heatmap(data, x=x_col, y=y_col, z=z_col, title=title, nbinsx=13, nbinsy=7, color_continuous_scale='rdylbu')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title=x_col.capitalize(), yaxis_title=y_col.capitalize())
    return fig

# --- Layout Definition ---
def layout() -> html.Div:
    # Uses create_sales_dashboard_layout from ui.components
    # Note: create_sales_dashboard_layout was not refactored by me due to tool limits,
    # so it will still contain Spanish text and old IDs.
    # The IDs used in Outputs of the callback below MUST match those in create_sales_dashboard_layout.
    # If create_sales_dashboard_layout was successfully updated to use prefixed IDs like 'sales-upload-data',
    # then the Output IDs here must match. Assuming old IDs for now based on previous failure.
    return create_sales_dashboard_layout() # This component comes from ui.components

# --- Callbacks ---
def register_callbacks(app):
    # Assuming IDs from the original create_ops_sales_layout in ui.components.py
    # If ui.components.create_sales_dashboard_layout was successfully updated with new IDs, these must match.
    # For safety, using the original IDs found in the provided initial code.
    # If the layout component uses 'sales-upload-data', then Input should be 'sales-upload-data'

    # IDs based on the provided initial `ui/components.py` for `create_ops_sales_layout`
    # These might need to change if `create_sales_dashboard_layout` in `ui.components.py`
    # was successfully refactored with new IDs like 'sales-upload-data', 'sales-filter-destination' etc.
    # If the `overwrite_file_with_block` for `ui.components.py` in step 5 failed, these IDs are correct.
    # If it succeeded with new IDs, these must be updated.
    # Given the failure in step 5, I will assume original IDs are still in `ui.components.py`.

    output_ids = [
        Output('output-kpis', 'children'), Output('destino-filter', 'options'),
        Output('operador-filter', 'options'), Output('mes-filter', 'options'),
        Output('vuelos-mes', 'figure'), Output('ingresos-mes', 'figure'),
        Output('ganancia-mes', 'figure'), Output('ganancia-total-mes', 'figure'),
        Output('ops-total-mes', 'figure'), Output('vuelos-tiempo', 'figure'),
        Output('ingresos-tiempo', 'figure'), Output('ganancia-tiempo', 'figure'),
        Output('top-destinos-vuelos', 'figure'), Output('top-destinos-ganancia', 'figure'),
        Output('pasajeros-destino', 'figure'), Output('vuelos-operador', 'figure'),
        Output('ganancia-aeronave', 'figure'), Output('top-ganancia-operador', 'figure'),
        Output('top-ganancia-aeronave', 'figure'), Output('destino-heatmap', 'options'),
        Output('heatmap-gain-destino-dia', 'figure'), Output('heatmap-count-destino-dia', 'figure'),
        Output('heatmap-dia-hora', 'figure'), Output('ticket-promedio', 'figure'),
        Output('tabla-detallada', 'data'), Output('tabla-detallada', 'columns'),
        Output('error-message', 'children'),
        Output('ai-insight-comparativo-general', 'children'),
        Output('ai-insight-vuelos-destinos', 'children'),
        Output('ai-insight-operadores-aeronaves', 'children'),
        Output('ai-insight-analisis-avanzado', 'children')
    ]
    input_ids = [
        Input('upload-data', 'contents'), Input('upload-data', 'filename'),
        Input('destino-filter', 'value'), Input('operador-filter', 'value'),
        Input('mes-filter', 'value'), Input('destino-heatmap', 'value')
    ]

    @app.callback(output_ids, input_ids)
    def update_sales_dashboard(contents, filenames, destino_filter_val, operador_filter_val, mes_filter_val, destino_heatmap_val):
        no_ai_insight_text = "Insufficient data to generate AI analysis."

        # Initialize all figure outputs to empty figures with titles
        initial_figures = {
            'vuelos-mes': create_empty_figure("Flights by Month"),
            'ingresos-mes': create_empty_figure("Revenue by Month"),
            'ganancia-mes': create_empty_figure("Profit by Month"),
            'ganancia-total-mes': create_empty_figure("Total Monthly Profit + Trend"),
            'ops-total-mes': create_empty_figure("Total Monthly Operations + Trend"),
            'vuelos-tiempo': create_empty_figure("Weekly Flights Time Series"),
            'ingresos-tiempo': create_empty_figure("Weekly Revenue Time Series"),
            'ganancia-tiempo': create_empty_figure("Weekly Profit Time Series"),
            'top-destinos-vuelos': create_empty_figure("Top Destinations by Flights"),
            'top-destinos-ganancia': create_empty_figure("Top Destinations by Profit"),
            'pasajeros-destino': create_empty_figure("Top Destinations by Passengers"),
            'vuelos-operador': create_empty_figure("Flights per Operator"),
            'ganancia-aeronave': create_empty_figure("Profit by Aircraft"),
            'top-ganancia-operador': create_empty_figure("Top Operators by Profit"),
            'top-ganancia-aeronave': create_empty_figure("Top Aircraft by Profit"),
            'heatmap-gain-destino-dia': create_empty_figure("Profit Heatmap (Destination)"),
            'heatmap-count-destino-dia': create_empty_figure("Operations Heatmap (Destination)"),
            'heatmap-dia-hora': create_empty_figure("Overall Flights Heatmap (Day/Hour)"),
            'ticket-promedio': create_empty_figure("Average Ticket Price")
        }

        initial_return_state = [
            [], [], [], [], # KPIs, filter options
            initial_figures['vuelos-mes'], initial_figures['ingresos-mes'], initial_figures['ganancia-mes'],
            initial_figures['ganancia-total-mes'], initial_figures['ops-total-mes'],
            initial_figures['vuelos-tiempo'], initial_figures['ingresos-tiempo'], initial_figures['ganancia-tiempo'],
            initial_figures['top-destinos-vuelos'], initial_figures['top-destinos-ganancia'], initial_figures['pasajeros-destino'],
            initial_figures['vuelos-operador'], initial_figures['ganancia-aeronave'],
            initial_figures['top-ganancia-operador'], initial_figures['top-ganancia-aeronave'],
            [], # destino-heatmap options
            initial_figures['heatmap-gain-destino-dia'], initial_figures['heatmap-count-destino-dia'],
            initial_figures['heatmap-dia-hora'], initial_figures['ticket-promedio'],
            [], [], '', # table data, columns, error message
            no_ai_insight_text, no_ai_insight_text, no_ai_insight_text, no_ai_insight_text # AI insights
        ]

        if not contents or not filenames:
            initial_return_state[-4] = "Please upload data files to begin analysis." # Error message index
            return initial_return_state

        df, err = unify_data(contents, filenames)
        if err:
            initial_return_state[-4] = f"Error processing files: {err}"
            return initial_return_state

        if df is None or df.empty:
            initial_return_state[-4] = "No data to display after processing files."
            return initial_return_state

        df = clean_df(df)
        df_plot_original = df.copy() # For populating filters

        # Apply filters
        filtered_df = df.copy()
        if destino_filter_val: filtered_df = filtered_df[filtered_df['Destino'].astype(str).isin(destino_filter_val)]
        if operador_filter_val: filtered_df = filtered_df[filtered_df['Operador'].astype(str).isin(operador_filter_val)]
        if mes_filter_val:
             # Ensure mes_filter_val contains strings if 'Mes' column is string
            str_mes_filter_val = [str(m) for m in mes_filter_val]
            filtered_df = filtered_df[filtered_df['Mes'].astype(str).isin(str_mes_filter_val)]


        # Prepare filter options based on the original full dataset
        destino_options = [{'label': d, 'value': d} for d in safe_sorted_unique(df_plot_original['Destino'])]
        operador_options = [{'label': o, 'value': o} for o in safe_sorted_unique(df_plot_original['Operador'])]
        mes_options = [{'label': m, 'value': m} for m in safe_sorted_unique(df_plot_original['Mes'].astype(str))] # Ensure Mes is string for options
        destino_heatmap_options = [{'label': d, 'value': d} for d in safe_sorted_unique(df_plot_original['Destino'])]

        if filtered_df.empty:
            initial_return_state[1] = destino_options
            initial_return_state[2] = operador_options
            initial_return_state[3] = mes_options
            initial_return_state[19] = destino_heatmap_options # Index for 'destino-heatmap' options
            initial_return_state[-4] = "No data matches the selected filters."
            return initial_return_state

        # KPIs
        kpi_cards_list = []
        for year_val in sorted(filtered_df['Año'].unique()):
            df_year = filtered_df[filtered_df['Año'] == year_val]
            if df_year.empty: continue
            avg_ticket = df_year['Monto total a cobrar'].mean() if not df_year.empty and df_year['Monto total a cobrar'].sum() > 0 else 0
            kpi_cards_list.append(dbc.Col(dbc.Card([
                dbc.CardHeader(f"Summary Year {year_val}", className="text-white", style={'backgroundColor': '#002859'}), # Translated
                dbc.CardBody([
                    html.H5("Total Flights", className="card-title"), html.P(f"{df_year.shape[0]}", className="card-text fs-4 fw-bold"), # Translated
                    html.H5("Total Passengers", className="card-title mt-2"), html.P(f"{int(df_year['Número de pasajeros'].sum())}", className="card-text fs-4 fw-bold"), # Translated
                    html.H5("Total Revenue", className="card-title mt-2"), html.P(f"${df_year['Monto total a cobrar'].sum():,.2f}", className="card-text fs-4 fw-bold"), # Translated
                    html.H5("Total Profit", className="card-title mt-2"), html.P(f"${df_year['Ganancia'].sum():,.2f}", className="card-text fs-4 fw-bold"), # Translated
                    html.H5("Average Ticket Price", className="card-title mt-2"), html.P(f"${avg_ticket:,.2f}", className="card-text fs-4 fw-bold"), # Translated
                ])
            ], className="shadow-sm mb-4 h-100"), xs=12, sm=6, md=4, lg=3))
        output_kpis_children = dbc.Row(kpi_cards_list)

        # Prepare data for plots
        df_plot = filtered_df.copy()
        # Ensure 'Fecha y hora del vuelo' is datetime
        if 'Fecha y hora del vuelo' not in df_plot.columns or not pd.api.types.is_datetime64_any_dtype(df_plot['Fecha y hora del vuelo']):
             df_plot['Fecha y hora del vuelo'] = pd.to_datetime(df_plot['Fecha y hora del vuelo'], errors='coerce')
        
        df_plot = df_plot.dropna(subset=['Fecha y hora del vuelo']) # Drop rows where date conversion failed

        meses_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        df_plot['MonthName'] = df_plot['Fecha y hora del vuelo'].dt.strftime('%B')
        df_plot['MonthName'] = pd.Categorical(df_plot['MonthName'], categories=meses_order, ordered=True)
        df_plot['year_month'] = df_plot['Fecha y hora del vuelo'].dt.to_period('M').astype(str)
        
        # Generate figures using helper functions
        fig_vuelos_mes = get_fig_flights_by_month(df_plot)
        fig_ingresos_mes = get_fig_revenue_by_month(df_plot)
        fig_ganancia_mes = get_fig_profit_by_month(df_plot)
        fig_ganancia_total_mes = get_fig_total_monthly_profit_timeline(df_plot)
        fig_ops_total_mes = get_fig_total_monthly_operations_timeline(df_plot, meses_order)
        
        fig_vuelos_tiempo = get_fig_weekly_timeseries(df_plot, 'Vuelos', 'Weekly Flights Time Series')
        fig_ingresos_tiempo = get_fig_weekly_timeseries(df_plot, 'Monto total a cobrar', 'Weekly Revenue Time Series')
        fig_ganancia_tiempo = get_fig_weekly_timeseries(df_plot, 'Ganancia', 'Weekly Profit Time Series')

        fig_top_destinos_vuelos = get_fig_top_destinations_by_flights(df_plot)
        fig_top_destinos_ganancia = get_fig_top_destinations_by_profit(df_plot)
        fig_pasajeros_destino = get_fig_top_destinations_by_passengers(df_plot)
        fig_vuelos_operador = get_fig_flights_per_operator(df_plot)
        fig_ganancia_aeronave = get_fig_profit_by_aircraft(df_plot)
        fig_top_ganancia_operador = get_fig_top_operators_by_total_profit(df_plot)
        fig_top_ganancia_aeronave = get_fig_top_aircraft_by_total_profit(df_plot)
        fig_ticket_promedio = get_fig_average_ticket_by_destination_year(df_plot)

        # Heatmap figures
        fig_heatmap_gain_destino_res = create_empty_figure("Profit Heatmap (Destination)")
        fig_heatmap_count_destino_res = create_empty_figure("Operations Heatmap (Destination)")
        fig_heatmap_overall_res = create_empty_figure("Overall Flights Heatmap")
        
        context_aa_parts = []
        if not fig_ticket_promedio.data: # Check if data was plotted
             context_aa_parts.append(f"Average ticket price by destination/year (top 5): {df_plot.groupby(['Año', 'Destino'], observed=False)['Monto total a cobrar'].mean().reset_index().head().to_string()}")

        if 'Año' in df_plot.columns and df_plot['Año'].nunique() > 0 and 'hora' in df_plot.columns and 'nombre_dia' in df_plot.columns:
            year_latest = sorted(df_plot['Año'].astype(str).unique())[-1]
            # Ensure 'hora' is numeric and 'nombre_dia' exists before filtering
            df_hm_base = df_plot[(df_plot['Año'].astype(str) == year_latest)].copy()
            df_hm_base['hora'] = pd.to_numeric(df_hm_base['hora'], errors='coerce')
            df_hm_base = df_hm_base.dropna(subset=['hora'])
            df_hm = df_hm_base[(df_hm_base['hora'] >= 6) & (df_hm_base['hora'] <= 18)]

            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            if not df_hm.empty:
                heatmap_data = df_hm.groupby(['nombre_dia', 'hora'], observed=False).size().reset_index(name='Flights')
                if not heatmap_data.empty:
                    fig_heatmap_overall_res = get_fig_heatmap(heatmap_data, 'hora', 'nombre_dia', 'Flights', f'Overall Flights by Day & Hour ({year_latest}, 6AM-6PM)', dias_orden)
                    context_aa_parts.append(f"Overall flights heatmap (summary): {heatmap_data.describe().to_string()}")

                if destino_heatmap_val and destino_heatmap_val in df_hm['Destino'].unique():
                    df_sel = df_hm[df_hm['Destino'] == destino_heatmap_val]
                    if not df_sel.empty:
                        heatmap_gain_data = df_sel.groupby(['nombre_dia', 'hora'], observed=False)['Ganancia'].sum().reset_index()
                        if not heatmap_gain_data.empty:
                             fig_heatmap_gain_destino_res = get_fig_heatmap(heatmap_gain_data, 'hora', 'nombre_dia', 'Ganancia', f'Profit Heatmap - {destino_heatmap_val}', dias_orden)
                             context_aa_parts.append(f"Profit heatmap for {destino_heatmap_val} (summary): {heatmap_gain_data.describe().to_string()}")

                        heatmap_count_data = df_sel.groupby(['nombre_dia', 'hora'], observed=False).size().reset_index(name='Flights')
                        if not heatmap_count_data.empty:
                            fig_heatmap_count_destino_res = get_fig_heatmap(heatmap_count_data, 'hora', 'nombre_dia', 'Flights', f'Operations Heatmap - {destino_heatmap_val}', dias_orden)
                            context_aa_parts.append(f"Operations heatmap for {destino_heatmap_val} (summary): {heatmap_count_data.describe().to_string()}")

        # AI Insights (translated prompts)
        # Need to ensure data passed to to_string() is not excessively large.
        ai_insight_comparativo = get_openai_response(
            "Analyze comparative trends in flights, revenue, and profit. Provide a diagnosis and a powerful action item.",
            f"Comparative data: Flights/month summary: {fig_vuelos_mes.data[0] if fig_vuelos_mes.data else 'N/A'}\nRevenue/month summary: {fig_ingresos_mes.data[0] if fig_ingresos_mes.data else 'N/A'}\nProfit/month summary: {fig_ganancia_mes.data[0] if fig_ganancia_mes.data else 'N/A'}"
        ) if not df_plot.empty else no_ai_insight_text

        ai_insight_vuelos = get_openai_response(
            "Analyze top destinations by flights and profit. Diagnose and suggest an action to optimize routes or profitability.",
            f"Top destinations by flights (summary): {df_plot.groupby('Destino')['Vuelos'].count().nlargest(5).to_string() if 'Vuelos' in df_plot else df_plot.groupby('Destino').size().nlargest(5).to_string()}\nTop destinations by profit (summary): {df_plot.groupby('Destino')['Ganancia'].sum().nlargest(5).to_string()}"
        ) if not df_plot.empty else no_ai_insight_text

        ai_insight_operadores = get_openai_response(
            "Analyze performance by operator and aircraft. What diagnosis and powerful action item do you suggest?",
            f"Flights by operator (summary): {df_plot.groupby('Operador').size().nlargest(5).to_string()}\nProfit by aircraft (summary): {df_plot.groupby('Aeronave')['Ganancia'].sum().nlargest(5).to_string()}"
        ) if not df_plot.empty else no_ai_insight_text

        ai_insight_avanzado = get_openai_response(
            "Based on heatmaps and average ticket prices, what demand patterns or opportunities are observed? Provide a diagnosis and a powerful action item.",
            "\n".join(context_aa_parts)
        ) if context_aa_parts else no_ai_insight_text

        # Table data
        display_columns = ['Año', 'Mes', 'Fecha y hora del vuelo', 'Destino', 'Operador', 'Aeronave', 'Número de pasajeros', 'Monto total a cobrar', 'Ganancia', 'Cliente', 'Fase actual']
        # Filter df_plot to only include columns that actually exist, to prevent KeyErrors
        existing_display_columns = [col for col in display_columns if col in df_plot.columns]
        df_table_display = df_plot[existing_display_columns].copy()

        if 'Fecha y hora del vuelo' in df_table_display.columns:
            df_table_display['Fecha y hora del vuelo'] = df_table_display['Fecha y hora del vuelo'].dt.strftime('%Y-%m-%d %H:%M')

        tabla_data = df_table_display.to_dict('records')
        tabla_columns = [{'name': col.replace('_', ' ').title(), 'id': col} for col in df_table_display.columns] # Prettify column names

        return (
            output_kpis_children, destino_options, operador_options, mes_options,
            fig_vuelos_mes, fig_ingresos_mes, fig_ganancia_mes, fig_ganancia_total_mes, fig_ops_total_mes,
            fig_vuelos_tiempo, fig_ingresos_tiempo, fig_ganancia_tiempo, fig_top_destinos_vuelos,
            fig_top_destinos_ganancia, fig_pasajeros_destino, fig_vuelos_operador, fig_ganancia_aeronave,
            fig_top_ganancia_operador, fig_top_ganancia_aeronave, destino_heatmap_options,
            fig_heatmap_gain_destino_res, fig_heatmap_count_destino_res, fig_heatmap_overall_res, fig_ticket_promedio,
            tabla_data, tabla_columns, '', # No error message if successful
            ai_insight_comparativo, ai_insight_vuelos, ai_insight_operadores, ai_insight_avanzado
        )
