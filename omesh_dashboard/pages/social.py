from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, dash_table, callback # Added callback
import dash_bootstrap_components as dbc
import logging
import typing

# Project dependencies
from ..settings import FACEBOOK_ID, INSTAGRAM_ID # These should be configured in .env and settings.py
from ..ai import get_openai_response
from ..ui.components import (
    create_ai_insight_card,
    create_ai_chat_interface,
    add_trendline,
    generate_wordcloud_base64
    # If generate_wordcloud was not renamed in ui.components.py due to previous issues, use:
    # from ..ui.components import generate_wordcloud as generate_wordcloud_base64
)
# Correcting the import if ui.components wasn't updated:
try:
    from ..ui.components import generate_wordcloud_base64
except ImportError:
    from ..ui.components import generate_wordcloud as generate_wordcloud_base64


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
def get_fig_instagram_impressions_trend(df_ig: pd.DataFrame) -> go.Figure:
    if df_ig.empty or 'timestamp' not in df_ig.columns or 'impressions' not in df_ig.columns or len(df_ig) < 2:
        return create_empty_figure("Instagram Daily Impressions Trend (No data)")

    df_ig_trend_data = df_ig.sort_values('timestamp').set_index('timestamp')['impressions'].resample('D').sum().reset_index()
    if len(df_ig_trend_data) < 2:
        return create_empty_figure("Instagram Daily Impressions Trend (Not enough data points)")

    fig = px.line(df_ig_trend_data, x='timestamp', y='impressions', title='Instagram Daily Impressions Trend', markers=True)
    # add_trendline is from ui.components; ensure it's robust or handle potential errors here
    try:
        fig = add_trendline(fig, df_ig_trend_data, 'timestamp', 'impressions', trendline_name_prefix="Trend")
    except Exception as e:
        logger.warning(f"Could not add trendline to Instagram impressions: {e}")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Date", yaxis_title="Impressions")
    return fig

def get_fig_instagram_engagement_by_type(df_ig_eng_summary: pd.DataFrame, y_col: str, title: str) -> go.Figure:
    if df_ig_eng_summary.empty or 'media_type' not in df_ig_eng_summary.columns or y_col not in df_ig_eng_summary.columns:
        return create_empty_figure(f"{title} (No data)")

    fig = px.bar(df_ig_eng_summary, x='media_type', y=y_col, title=title, text_auto=True, color='media_type')
    yaxis_title = "Total Engagement" if y_col == 'total_engagement' else "Avg. Engagement Rate (%)"
    if y_col == 'avg_engagement_rate':
        fig.update_yaxes(ticksuffix="%")
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Media Type", yaxis_title=yaxis_title)
    return fig

# --- Layout Definition ---
def layout() -> html.Div:
    return html.Div([
        dcc.Tabs(id="social-media-subtabs", value="general", children=[
            dcc.Tab(label="Overall Metrics", value="general"),
            dcc.Tab(label="Instagram Engagement", value="ig_engagement"),
            dcc.Tab(label="Content Word Cloud", value="wordcloud"),
            dcc.Tab(label="Top Performing Posts", value="top_posts"),
        ]),
        dcc.Loading(id="social-loading-indicator", type="default", children=[
            html.Div(id="social-subtabs-content-area")
        ]),
    ])

# --- Callbacks ---
def register_callbacks(app: dash.Dash) -> None:
    """Registers all callbacks for the Social Media page."""

    @app.callback(
        Output('social-subtabs-content-area', 'children'),
        Input('social-media-subtabs', 'value'),
        State('date-picker', 'start_date'),
        State('date-picker', 'end_date')
    )
    def render_social_media_subtab_content(
        subtab_value: str,
        start_date_str: str | None,
        end_date_str: str | None
    ) -> html.Div:

        if not start_date_str or not end_date_str:
            return html.Div(html.P("Please select a date range to view social media analytics.", className="text-center mt-5 text-warning"))

        start_date_dt = pd.to_datetime(start_date_str, errors='coerce').tz_localize(None)
        end_date_dt = pd.to_datetime(end_date_str, errors='coerce').tz_localize(None)

        if pd.NaT in [start_date_dt, end_date_dt]:
             return html.Div(html.P("Invalid date format selected. Please check the dates.", className="text-center mt-5 text-danger"))

        default_ai_text = "Insufficient data for AI analysis."
        ai_card_id = f"social-{subtab_value}-ai-card"
        ai_data_id = f"social-{subtab_value}-ai-data"
        chat_interface_id_prefix = f"social_{subtab_value}_chat"

        if not FACEBOOK_ID or not INSTAGRAM_ID:
            logger.warning("FACEBOOK_ID or INSTAGRAM_ID is not configured in settings.")
            return html.Div([
                html.P("Social media account IDs (Facebook, Instagram) are not configured in the settings.", className="text-danger text-center mt-3"),
                create_ai_insight_card(ai_card_id, title="🤖 AI Analysis"),
                html.Div(default_ai_text, id=ai_data_id, style={'display': 'none'}),
                create_ai_chat_interface(chat_interface_id_prefix)
            ])

        try:
            from ..data.data_processing import get_facebook_posts, get_instagram_posts, process_facebook_posts, process_instagram_posts
            fb_posts_raw = get_facebook_posts(FACEBOOK_ID)
            ig_posts_raw = get_instagram_posts(INSTAGRAM_ID)
            df_fb_full = process_facebook_posts(fb_posts_raw)
            df_ig_full = process_instagram_posts(ig_posts_raw)
        except Exception as e:
            logger.error(f"Error fetching or processing social media data: {e}", exc_info=True)
            return html.Div([
                html.P(f"Error fetching social media data: {str(e)}", className="text-danger text-center mt-3"),
                create_ai_insight_card(ai_card_id, title="🤖 AI Analysis"),
                html.Div(default_ai_text, id=ai_data_id, style={'display': 'none'}),
                create_ai_chat_interface(chat_interface_id_prefix)
            ])

        df_fb = df_fb_full.copy()
        df_ig = df_ig_full.copy()

        if not df_fb.empty and 'created_time' in df_fb.columns:
            df_fb['created_time'] = pd.to_datetime(df_fb['created_time'], errors='coerce').dt.tz_localize(None)
            df_fb = df_fb.dropna(subset=['created_time'])
            df_fb = df_fb[(df_fb['created_time'] >= start_date_dt) & (df_fb['created_time'] <= end_date_dt)]
        if not df_ig.empty and 'timestamp' in df_ig.columns:
            df_ig['timestamp'] = pd.to_datetime(df_ig['timestamp'], errors='coerce').dt.tz_localize(None)
            df_ig = df_ig.dropna(subset=['timestamp'])
            df_ig = df_ig[(df_ig['timestamp'] >= start_date_dt) & (df_ig['timestamp'] <= end_date_dt)]

        no_data_message_div = html.Div([
            html.P("No social media data available for the selected period.", className="text-center mt-3"),
            create_ai_insight_card(ai_card_id, title="🤖 AI Analysis"),
            html.Div(default_ai_text, id=ai_data_id, style={'display': 'none'}),
            create_ai_chat_interface(chat_interface_id_prefix)
        ])

        if subtab_value == 'general':
            if df_fb.empty and df_ig.empty:
                return no_data_message_div

            metrics_display_map = {
                "impressions": "Impressions", "reach": "Reach", "engagement": "Engagement",
                "likes_count": "Likes", "like_count": "Likes",
                "video_views": "Video Views", "comments_count": "Comments", "shares_count": "Shares"
            }

            fb_metrics_data = {}
            if not df_fb.empty:
                fb_metrics_data = {
                    metrics_display_map.get(col, col.replace("_"," ").title()): df_fb[col].sum()
                    for col in ['impressions', 'likes_count', 'comments_count', 'shares_count'] if col in df_fb and pd.api.types.is_numeric_dtype(df_fb[col])
                }

            ig_metrics_data = {}
            if not df_ig.empty:
                ig_metrics_data = {
                    metrics_display_map.get(col, col.replace("_"," ").title()): df_ig[col].sum()
                    for col in ['impressions', 'reach', 'engagement', 'like_count', 'video_views', 'comments_count'] if col in df_ig and pd.api.types.is_numeric_dtype(df_ig[col])
                }

            fig_ig_impressions = get_fig_instagram_impressions_trend(df_ig)

            ai_text_general = default_ai_text
            if fb_metrics_data or ig_metrics_data:
                context_general = f"Overall Social Metrics: Facebook - {fb_metrics_data}. Instagram - {ig_metrics_data}. Instagram daily impressions trend also shown."
                prompt_general = "Analyze the overall performance metrics for Facebook and Instagram. Which platform stands out and in which key metrics? Diagnose the general performance and suggest one powerful, actionable recommendation."
                ai_text_general = get_openai_response(prompt_general, context_general)

            return html.Div([
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H5("Facebook Key Metrics", className="card-title text-primary"),
                        *[html.P(f"{name}: {value:,.0f}") for name, value in fb_metrics_data.items()]
                    ])), md=6, className="mb-3"),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H5("Instagram Key Metrics", className="card-title text-danger"),
                        *[html.P(f"{name}: {value:,.0f}") for name, value in ig_metrics_data.items()]
                    ])), md=6, className="mb-3"),
                ]),
                dcc.Graph(figure=fig_ig_impressions) if not df_ig.empty else html.P("Instagram impressions trend data not available.", className="text-center"),
                create_ai_insight_card(ai_card_id, title="💡 AI Insight & Action (Overall Social Metrics)"),
                html.Div(ai_text_general, id=ai_data_id, style={'display': 'none'}),
                create_ai_chat_interface(chat_interface_id_prefix)
            ])

        elif subtab_value == 'ig_engagement':
            if df_ig.empty or not all(k in df_ig for k in ['media_type', 'engagement', 'reach']):
                return html.Div([
                    html.P("Not enough Instagram data to analyze engagement by media type.", className="text-center mt-3"),
                    create_ai_insight_card(ai_card_id, title="🤖 AI Analysis (IG Engagement)"), html.Div(default_ai_text, id=ai_data_id, style={'display': 'none'}),
                    create_ai_chat_interface(chat_interface_id_prefix)
                ])

            df_ig_eng = df_ig.copy()
            df_ig_eng['engagement_rate'] = (df_ig_eng['engagement'].fillna(0) / df_ig_eng['reach'].replace(0, np.nan).fillna(1) * 100).fillna(0)

            eng_by_type_summary = df_ig_eng.groupby('media_type', as_index=False).agg(
                total_engagement=('engagement', 'sum'),
                avg_engagement_rate=('engagement_rate', 'mean')
            ).sort_values('total_engagement', ascending=False)

            fig_total_engagement = get_fig_instagram_engagement_by_type(eng_by_type_summary, 'total_engagement', 'Total Engagement by Media Type (Instagram)')
            fig_avg_engagement_rate = get_fig_instagram_engagement_by_type(eng_by_type_summary, 'avg_engagement_rate', 'Average Engagement Rate (%) by Media Type (Instagram)')

            ai_text_ig_eng = default_ai_text
            if not eng_by_type_summary.empty:
                context_ig_eng = f"Instagram Engagement by Media Type: {eng_by_type_summary.to_string(index=False)}"
                prompt_ig_eng = "Analyze total engagement and average engagement rate by media type on Instagram. Which format is most effective for engagement? Diagnose and suggest one powerful, actionable recommendation to improve Instagram engagement."
                ai_text_ig_eng = get_openai_response(prompt_ig_eng, context_ig_eng)

            return html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_total_engagement), md=6),
                    dbc.Col(dcc.Graph(figure=fig_avg_engagement_rate), md=6)
                ]),
                create_ai_insight_card(ai_card_id, title="💡 AI Diagnosis & Action (Instagram Engagement)"),
                html.Div(ai_text_ig_eng, id=ai_data_id, style={'display': 'none'}),
                create_ai_chat_interface(chat_interface_id_prefix)
            ])

        elif subtab_value == 'wordcloud':
            if df_fb.empty and df_ig.empty:
                return no_data_message_div

            text_fb_msgs = " ".join(df_fb['message'].dropna().astype(str)) if not df_fb.empty and 'message' in df_fb else ""
            text_ig_captions = " ".join(df_ig['caption'].dropna().astype(str)) if not df_ig.empty and 'caption' in df_ig else ""
            full_text_content = (text_fb_msgs + " " + text_ig_captions).strip()

            wordcloud_image_src = generate_wordcloud_base64(full_text_content)

            ai_text_wordcloud = default_ai_text
            if wordcloud_image_src:
                context_wordcloud = "A word cloud has been generated from the most frequent words in Facebook and Instagram posts."
                prompt_wordcloud = "Observing the word cloud from posts, what general themes appear predominant? Diagnose if these themes align with the content strategy and suggest one powerful, actionable recommendation to optimize messaging."
                ai_text_wordcloud = get_openai_response(prompt_wordcloud, context_wordcloud)
            else:
                ai_text_wordcloud = "No text content found in posts to generate a word cloud or AI analysis."

            return html.Div([
                html.H5("Word Cloud from Facebook & Instagram Posts", className="text-center my-3"),
                html.Img(src=wordcloud_image_src, style={'width': '100%', 'maxWidth': '700px', 'display': 'block', 'margin': 'auto', 'border': '1px solid #ddd'}) if wordcloud_image_src else html.P('Could not generate the Word Cloud. Ensure there is text in posts.', className="text-center"),
                create_ai_insight_card(ai_card_id, title="💡 AI Insight & Action (Content Word Cloud)"),
                html.Div(ai_text_wordcloud, id=ai_data_id, style={'display': 'none'}),
                create_ai_chat_interface(chat_interface_id_prefix)
            ])

        elif subtab_value == 'top_posts':
            # Use full data before date filtering for identifying top posts, then filter display
            df_fb_all = df_fb_full.copy()
            df_ig_all = df_ig_full.copy()

            if df_fb_all.empty and df_ig_all.empty: return no_data_message_div

            df_fb_std = pd.DataFrame(); df_ig_std = pd.DataFrame()
            if not df_fb_all.empty:
                df_fb_std = df_fb_all[['id', 'message', 'created_time', 'likes_count', 'comments_count', 'impressions']].copy()
                df_fb_std.rename(columns={'message': 'content', 'created_time': 'time', 'likes_count': 'likes', 'comments_count':'comments'}, inplace=True)
                df_fb_std['platform'] = 'Facebook'

            if not df_ig_all.empty:
                df_ig_std = df_ig_all[['id', 'caption', 'timestamp', 'like_count', 'comments_count', 'impressions', 'permalink', 'media_type']].copy()
                df_ig_std.rename(columns={'caption': 'content', 'timestamp': 'time', 'like_count': 'likes', 'comments_count':'comments'}, inplace=True)
                df_ig_std['platform'] = 'Instagram'

            df_all_posts_unfiltered = pd.concat([df_fb_std, df_ig_std], ignore_index=True)
            if df_all_posts_unfiltered.empty:
                 return html.Div([html.P("No posts found from Facebook or Instagram.", className="text-center"), create_ai_insight_card(ai_card_id), html.Div(default_ai_text, id=ai_data_id, style={'display': 'none'}), create_ai_chat_interface(chat_interface_id_prefix)])

            df_all_posts_unfiltered['time'] = pd.to_datetime(df_all_posts_unfiltered['time'], errors='coerce').dt.tz_localize(None)
            df_all_posts_unfiltered.dropna(subset=['time'], inplace=True) # Critical for next step

            # Calculate total impact for sorting on unfiltered data
            df_all_posts_unfiltered['total_impact'] = df_all_posts_unfiltered['likes'].fillna(0) + \
                                                      df_all_posts_unfiltered['comments'].fillna(0) + \
                                                      df_all_posts_unfiltered['impressions'].fillna(0)

            # Get top 10 overall, then filter these top 10 by date range for display
            top_10_overall_posts = df_all_posts_unfiltered.sort_values('total_impact', ascending=False).head(10)
            top_posts_to_display = top_10_overall_posts[
                (top_10_overall_posts['time'] >= start_date_dt) & (top_10_overall_posts['time'] <= end_date_dt)
            ]


            if top_posts_to_display.empty:
                return html.Div([html.P("No top posts found in the selected date range based on overall impact.", className="text-center"), create_ai_insight_card(ai_card_id), html.Div(default_ai_text, id=ai_data_id, style={'display': 'none'}), create_ai_chat_interface(chat_interface_id_prefix)])

            table_data_top_posts = []
            for _, row in top_posts_to_display.iterrows():
                post_content = str(row.get('content', 'N/A'))
                display_content = (post_content[:75] + "...") if len(post_content) > 75 else post_content
                if row.get('platform') == 'Instagram' and pd.notna(row.get('permalink')):
                    display_content = f"[{display_content}]({row['permalink']})"

                table_data_top_posts.append({
                    'Platform': row.get('platform', 'N/A'),
                    'Content': display_content,
                    'Date': pd.to_datetime(row.get('time')).strftime('%Y-%m-%d') if pd.notna(row.get('time')) else 'N/A',
                    'Likes': f"{row.get('likes', 0):,.0f}",
                    'Comments': f"{row.get('comments', 0):,.0f}",
                    'Impressions': f"{row.get('impressions', 0):,.0f}",
                    'Total Impact': f"{row.get('total_impact', 0):,.0f}"
                })

            datatable_columns = [{'name': c, 'id': c, 'presentation': 'markdown' if c=='Content' else 'input'} for c in table_data_top_posts[0].keys()] if table_data_top_posts else []

            ai_text_top_posts = default_ai_text
            if not top_posts_to_display.empty:
                context_top_posts = f"Top posts by engagement impact (likes+comments+impressions) within date range: Summary - {top_posts_to_display[['platform', 'content', 'total_impact']].head(3).to_string(index=False)}"
                prompt_top_posts = "Analyze common characteristics of the top-performing posts. What type of content or platform works best? Diagnose these patterns and suggest one powerful, actionable recommendation to replicate this success."
                ai_text_top_posts = get_openai_response(prompt_top_posts, context_top_posts)

            return html.Div([
                html.H5("Top Performing Posts by Engagement Impact (in selected date range)", className="text-center my-3"),
                dash_table.DataTable(
                    id='social-top-posts-table',
                    data=table_data_top_posts,
                    columns=datatable_columns,
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_cell={'textAlign': 'left', 'padding': '10px', 'minWidth': '100px', 'width': 'auto', 'maxWidth': '300px', 'whiteSpace': 'normal', 'height': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    markdown_options={'html': True},
                    page_size=10, sort_action='native', filter_action='native'
                ),
                create_ai_insight_card(ai_card_id, title="💡 AI Diagnosis & Action (Top Posts)"),
                html.Div(ai_text_top_posts, id=ai_data_id, style={'display': 'none'}),
                create_ai_chat_interface(chat_interface_id_prefix)
            ])

        return html.Div(html.P(f"Social Media Sub-tab '{subtab_value}' not implemented or data unavailable.", className="text-center mt-5 text-danger"))

    social_subtab_keys_for_ai = ['general', 'ig_engagement', 'wordcloud', 'top_posts']
    for tab_key in social_subtab_keys_for_ai:
        @app.callback(
            Output(f'social-{tab_key}-ai-card', 'children'),
            Input(f'social-{tab_key}-ai-data', 'children')
        )
        def update_social_media_ai_card(ai_text_content: str | None, card_tab_key=tab_key):
            default_msg = "AI analysis will appear here once data is processed."
            no_data_indicators = [
                "Insufficient data for AI analysis.", "No social media data available",
                "Not enough Instagram data to analyze", "No text content found in posts",
                "No posts found", "No posts with engagement impact found"
            ]
            if not ai_text_content or any(indicator in ai_text_content for indicator in no_data_indicators):
                return html.P(default_msg, className="text-muted")
            return html.P(ai_text_content)

    social_chat_prefixes = [f"social_{key}_chat" for key in social_subtab_keys_for_ai]
    for chat_prefix in social_chat_prefixes:
        @app.callback(
            Output(f'{chat_prefix}-chat-history', 'children'),
            Input(f'{chat_prefix}-chat-submit', 'n_clicks'),
            State(f'{chat_prefix}-chat-input', 'value'),
            State(f'{chat_prefix}-chat-history', 'children'),
            State('social-media-subtabs', 'value'),
            prevent_initial_call=True
        )
        def update_social_media_chat(
            n_clicks: int | None, user_input: str | None,
            chat_history: list | html.Div | None, active_social_subtab: str
        ):
            triggered_input_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

            if not n_clicks or not user_input:
                return chat_history or []

            current_history = list(chat_history) if isinstance(chat_history, list) else ([chat_history] if chat_history else [])

            context_tab_name = active_social_subtab

            context_for_ai = f"User is on the Social Media page, '{context_tab_name}' sub-tab. User asks: {user_input}"
            ai_response_text = get_openai_response(user_input, context_for_ai)

            new_entry_user = html.P([html.B("You: ", style={'color': '#007bff'}), user_input], style={'margin': '5px 0'})
            new_entry_ai = html.P([html.B("Omesh AI: ", style={'color': '#28a745'}), ai_response_text], style={'background': '#f0f0f0', 'padding': '8px', 'borderRadius': '5px', 'margin': '5px 0'})

            current_history.extend([new_entry_user, new_entry_ai])
            return current_history

```
