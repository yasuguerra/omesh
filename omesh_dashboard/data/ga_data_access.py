from __future__ import annotations

"""
Data Access Layer for Google Analytics 4.

This module provides high-level functions to fetch and process data from
the Google Analytics 4 Data API, returning pandas DataFrames.
It utilizes the generic `query_ga` function from `ga_client.py`.
"""

import pandas as pd
import logging
from .ga_client import query_ga # Correct import for query_ga

logger = logging.getLogger(__name__)

def get_traffic_overview_df(start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches a traffic overview DataFrame (sessions, users, conversions by date).

    Args:
        start_date: The start date for the query (YYYY-MM-DD or relative e.g., "30daysAgo").
        end_date: The end date for the query (YYYY-MM-DD or relative e.g., "today").
        property_id: Optional GA4 property ID to override settings.

    Returns:
        A pandas DataFrame with columns: ['date', 'sessions', 'activeUsers', 'conversions'].
        Returns an empty DataFrame with these columns if no data is found or an error occurs.
    """
    metrics = ['sessions', 'activeUsers', 'conversions']
    dimensions = ['date']
    expected_cols = dimensions + metrics
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None # Pass None to use default from settings
        )
        if df.empty:
            logger.warning(f"No traffic overview data returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)

        # Ensure correct dtypes for metrics, query_ga should mostly handle this
        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'], errors='coerce') # query_ga should handle this too

        return df[expected_cols] # Ensure column order and presence
    except Exception as e:
        logger.error(f"Error fetching traffic overview DF: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)


def get_top_channels_df(start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches a DataFrame of top traffic channels (e.g., by sessionSourceMedium).

    Args:
        start_date: The start date for the query.
        end_date: The end date for the query.
        property_id: Optional GA4 property ID.

    Returns:
        A pandas DataFrame with columns: ['sessionSourceMedium', 'sessions', 'conversions', 'conversionRate'].
        Returns an empty DataFrame with these columns if no data or error.
    """
    metrics = ['sessions', 'conversions']
    dimensions = ['sessionSourceMedium']
    expected_cols = dimensions + metrics + ['conversionRate']
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No top channels data returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)

        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)

        df['conversionRate'] = (df['conversions'] / df['sessions'].replace(0, pd.NA) * 100).fillna(0)
        return df[expected_cols]
    except Exception as e:
        logger.error(f"Error fetching top channels DF: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)


def get_cohorts_df(start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches data for cohort analysis (user retention by first session date and nth day).

    Args:
        start_date: The start date for the query.
        end_date: The end date for the query.
        property_id: Optional GA4 property ID.

    Returns:
        A pandas DataFrame with columns: ['firstSessionDate', 'nthDay', 'activeUsers'].
        Returns an empty DataFrame with these columns if no data or error.
    """
    metrics = ['activeUsers']
    dimensions = ['firstSessionDate', 'nthDay']
    expected_cols = dimensions + metrics
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No cohort data returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)

        # Dtype conversion should be handled by query_ga, but ensure for safety
        if 'firstSessionDate' in df.columns:
             df['firstSessionDate'] = pd.to_datetime(df['firstSessionDate'], errors='coerce')
        if 'nthDay' in df.columns:
            df['nthDay'] = pd.to_numeric(df['nthDay'], errors='coerce').fillna(0).astype(int)
        if 'activeUsers' in df.columns:
            df['activeUsers'] = pd.to_numeric(df['activeUsers'], errors='coerce').fillna(0)

        return df[expected_cols]
    except Exception as e:
        logger.error(f"Error fetching cohorts DF: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)


def get_funnel_data_df(steps_config: list[dict], start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches and processes data for funnel analysis based on a list of event steps.
    This is a reimplementation of the logic previously in data_processing.py,
    now using the local query_ga.

    Args:
        steps_config: A list of dictionaries, each defining a funnel step.
                      Example: [{"label": "Visited Site", "type": "event", "dimension": "eventName", "value": "page_view"}, ...]
        start_date: The start date for the query.
        end_date: The end date for the query.
        property_id: Optional GA4 property ID.

    Returns:
        A pandas DataFrame with columns: ['step_label', 'event_value', 'count'].
        Returns an empty DataFrame if no data or error.
    """
    funnel_results = []
    expected_cols = ['step_label', 'event_value', 'count']

    try:
        for step in steps_config:
            step_label = step.get('label', 'Unknown Step')
            event_value = step.get('value', '')
            count = 0

            if event_value == 'page_view': # Typically 'page_view' might correspond to 'sessions' for the first step
                # This logic might need adjustment based on actual GA4 event setup for "page_view" as a funnel start
                # For simplicity, let's assume 'sessions' for a 'page_view' event type if it's the first step.
                # Or it could be eventCount for eventName = 'page_view'. The original code had a special case.
                # Let's stick to eventCount for page_view for consistency with other events unless specified otherwise.
                df_step = query_ga(
                    metrics=['eventCount'], # Could also be 'sessions' depending on definition
                    dimensions=['eventName'], # Could be empty if just getting total sessions
                    start_date=start_date,
                    end_date=end_date,
                    property_id=property_id if property_id else None
                )
                if not df_step.empty and 'eventName' in df_step and 'eventCount' in df_step:
                    # If eventName is 'page_view' and we want its count
                    page_view_df = df_step[df_step['eventName'] == 'page_view']
                    if not page_view_df.empty:
                        count = int(page_view_df['eventCount'].sum())
                    # If 'page_view' meant total sessions for the period as funnel start:
                    # df_sessions = query_ga(metrics=['sessions'], dimensions=[], start_date=start_date, end_date=end_date, property_id=property_id)
                    # count = int(df_sessions['sessions'].sum()) if not df_sessions.empty and 'sessions' in df_sessions else 0

            elif step.get('type') == 'event' and 'dimension' in step and event_value:
                metric_to_use = 'eventCount' # Default for events
                dimension_to_use = step['dimension']

                df_step = query_ga(
                    metrics=[metric_to_use],
                    dimensions=[dimension_to_use],
                    start_date=start_date,
                    end_date=end_date,
                    property_id=property_id if property_id else None
                )
                if not df_step.empty and dimension_to_use in df_step.columns and metric_to_use in df_step.columns:
                    df_step_filtered = df_step[df_step[dimension_to_use] == event_value]
                    if not df_step_filtered.empty:
                        count = int(df_step_filtered[metric_to_use].sum())
            else:
                logger.warning(f"Skipping funnel step due to incomplete configuration: {step}")

            funnel_results.append({'step_label': step_label, 'event_value': event_value, 'count': count})

        if not funnel_results:
            return pd.DataFrame(columns=expected_cols)

        return pd.DataFrame(funnel_results)

    except Exception as e:
        logger.error(f"Error fetching funnel_data_df: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)

# Example helper to get data for the Plotly Funnel chart which expects separate lists
def get_funnel_plot_lists(steps_config: list[dict], start_date: str, end_date: str, property_id: str | None = None) -> tuple[list[str], list[int]]:
    """
    Wrapper around get_funnel_data_df to return labels and counts as separate lists,
    similar to the original get_funnel_data in data_processing.py.
    """
    df = get_funnel_data_df(steps_config, start_date, end_date, property_id)
    if df.empty:
        return [step.get('label', 'Unknown') for step in steps_config], [0] * len(steps_config)

    # Ensure order matches steps_config
    labels_map = {row['event_value']: row['count'] for index, row in df.iterrows()}
    ordered_counts = [labels_map.get(step['value'], 0) for step in steps_config]
    ordered_labels = [step['label'] for step in steps_config]

    return ordered_labels, ordered_counts

# Add more GA data access functions as needed:
# e.g., get_page_path_performance_df, get_demographics_df, etc.

def get_page_path_performance_df(start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches page path performance (sessions, bounceRate, averageSessionDuration).
    """
    metrics = ['sessions', 'bounceRate', 'averageSessionDuration']
    dimensions = ['pagePath']
    expected_cols = dimensions + metrics
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No page path data returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)
        # query_ga should handle dtypes, but good to be explicit for what pages/google_analytics.py expects
        if 'bounceRate' in df.columns: df['bounceRate'] = pd.to_numeric(df['bounceRate'], errors='coerce').fillna(0) # GA4 sends as ratio
        if 'averageSessionDuration' in df.columns: df['averageSessionDuration'] = pd.to_numeric(df['averageSessionDuration'], errors='coerce').fillna(0)
        if 'sessions' in df.columns: df['sessions'] = pd.to_numeric(df['sessions'], errors='coerce').fillna(0)
        return df[expected_cols]
    except Exception as e:
        logger.error(f"Error fetching page path performance DF: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)

def get_demographics_df(start_date: str, end_date: str, dimension: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches demographics data (activeUsers, conversions by a given dimension like userGender, userAgeBracket, country, city).
    """
    metrics = ['activeUsers', 'conversions']
    dimensions = [dimension]
    expected_cols = dimensions + metrics
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No demographics data for dimension {dimension} returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)
        for metric_col in metrics:
            if metric_col in df.columns: df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce').fillna(0)
        return df[expected_cols]
    except Exception as e:
        logger.error(f"Error fetching demographics DF for dimension {dimension}: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)

def get_events_df(start_date: str, end_date: str, event_names: list[str] | None = None, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches event counts by date and eventName.
    """
    metrics = ['eventCount']
    dimensions = ['date', 'eventName']
    expected_cols = dimensions + metrics
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No events data returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)
        if event_names and 'eventName' in df.columns:
            df = df[df['eventName'].isin(event_names)]

        if df.empty and event_names:
            logger.warning(f"No events data for specified event_names {event_names} returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)

        if 'eventCount' in df.columns: df['eventCount'] = pd.to_numeric(df['eventCount'], errors='coerce').fillna(0)
        if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df[expected_cols]
    except Exception as e:
        logger.error(f"Error fetching events DF: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)

def get_correlation_data_df(start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches data for correlation analysis (sessions, activeUsers, averageSessionDuration, bounceRate, conversions by date and deviceCategory).
    """
    metrics = ['sessions', 'activeUsers', 'averageSessionDuration', 'bounceRate', 'conversions']
    dimensions = ['date', 'deviceCategory']
    expected_cols = dimensions + metrics
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No correlation data returned for {start_date} - {end_date}")
            return pd.DataFrame(columns=expected_cols)

        for metric_col in metrics:
            if metric_col in df.columns: df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce').fillna(0)
        if 'bounceRate' in df.columns: df['bounceRate'] = df['bounceRate'] # GA4 sends as ratio, not %, multiply by 100 in page if needed
        if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df[expected_cols]
    except Exception as e:
        logger.error(f"Error fetching correlation data DF: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_cols)

def get_baseline_metrics_df(start_date: str, end_date: str, property_id: str | None = None) -> pd.DataFrame:
    """
    Fetches baseline metrics (sessions, conversions) for "What If" simulator.
    """
    metrics = ['sessions', 'conversions']
    dimensions = [] # No dimensions, just totals for the period
    expected_cols = metrics # No date or other dimensions needed for this specific use case in what-if
    try:
        df = query_ga(
            metrics=metrics,
            dimensions=dimensions,
            start_date=start_date,
            end_date=end_date,
            property_id=property_id if property_id else None
        )
        if df.empty:
            logger.warning(f"No baseline metrics data returned for {start_date} - {end_date}")
            # Return DataFrame with 0 values for metrics if empty
            return pd.DataFrame([[0] * len(expected_cols)], columns=expected_cols)

        for metric_col in metrics:
            if metric_col in df.columns:
                df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce').fillna(0)
            else: # Ensure column exists even if not returned by GA (should not happen for non-dimensioned query)
                df[metric_col] = 0
        return df[expected_cols] # Select only the metric columns
    except Exception as e:
        logger.error(f"Error fetching baseline metrics DF: {e}", exc_info=True)
        return pd.DataFrame([[0] * len(expected_cols)], columns=expected_cols)
