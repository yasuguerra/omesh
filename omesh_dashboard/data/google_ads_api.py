# google_ads_api.py
# -----------------------------------------------------------
# Helper para consultar métricas de Google Ads en SkyIntel
# -----------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.auth.exceptions import RefreshError

# ────────────────────────────────────────────────────────────
# 1) Localizar el YAML de configuración (developer token, etc.)
# ────────────────────────────────────────────────────────────
PROJECT_YAML: Path = Path(__file__).resolve().parent / "google-ads.yaml"


# ────────────────────────────────────────────────────────────
# 2) Cliente Google Ads
# ────────────────────────────────────────────────────────────
def load_client(config_path: str | os.PathLike | None = None) -> GoogleAdsClient:
    """
    Carga un GoogleAdsClient con la siguiente prioridad:
      1) Ruta explícita       -> parámetro `config_path`
      2) Variable de entorno  -> GOOGLE_ADS_CONFIGURATION_FILE_PATH
      3) YAML local al repo   -> google-ads.yaml
      4) YAML en HOME         -> ~/.google-ads.yaml (comportamiento por defecto)
    """
    if config_path:
        return GoogleAdsClient.load_from_storage(str(config_path))

    env_path = os.getenv("GOOGLE_ADS_CONFIGURATION_FILE_PATH")
    if env_path:
        return GoogleAdsClient.load_from_storage(env_path)

    if PROJECT_YAML.exists():
        return GoogleAdsClient.load_from_storage(str(PROJECT_YAML))

    # Fallback → que el SDK busque en ~/.google-ads.yaml
    return GoogleAdsClient.load_from_storage()


def load_client_safe(config_path: str | os.PathLike | None = None) -> GoogleAdsClient:
    """Versión segura que muestra un error claro si expira el refresh-token."""
    try:
        return load_client(config_path)
    except RefreshError as exc:
        raise RuntimeError(
            "⛔ Error al refrescar el token de Google Ads. "
            "Revisa tu refresh_token o client_id/secret."
        ) from exc


# ────────────────────────────────────────────────────────────
# 3) GAQL queries (ajusta columnas si lo necesitas)
# ────────────────────────────────────────────────────────────
GAQL_ADS_METRICS = """
SELECT
  segments.date,
  campaign.name,
  metrics.clicks,
  metrics.impressions,
  metrics.conversions,
  metrics.cost_micros
FROM campaign
WHERE segments.date BETWEEN '{start}' AND '{end}'
  AND campaign.status = 'ENABLED'
"""

GAQL_GEO = """
SELECT
  segments.date,
  segments.geo_target_city,
  metrics.clicks,
  metrics.impressions,
  metrics.conversions,
  metrics.cost_micros
FROM customer
WHERE segments.date BETWEEN '{start}' AND '{end}'
  AND segments.geo_target_city IS NOT NULL
"""

# (Si quieres keyword metrics añade otro GAQL similar)


# ────────────────────────────────────────────────────────────
# 4) Helper genérico: ejecuta GAQL y devuelve DataFrame
# ────────────────────────────────────────────────────────────
def _run_gaql(
    client: GoogleAdsClient,
    customer_id: str,
    query: str,
) -> List[dict]:
    service = client.get_service("GoogleAdsService")
    stream = service.search_stream(customer_id=customer_id, query=query)
    rows: List[dict] = []
    for batch in stream:
        for r in batch.results:
            rows.append(r)
    return rows


def fetch_ads_metrics(
    client: GoogleAdsClient,
    customer_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Devuelve métricas crudas por campaña + fecha."""
    raw = _run_gaql(client, customer_id, GAQL_ADS_METRICS.format(start=start, end=end))

    if not raw:
        return pd.DataFrame(columns=["date", "campaign", "clicks", "conversions", "cost"])

    rows: list[dict] = []
    for r in raw:
        rows.append(
            {
                # ⬇️  <---  ya son strings; no uses .value
                "date": r.segments.date,
                "campaign": r.campaign.name,
                "clicks": r.metrics.clicks,
                "conversions": r.metrics.conversions,
                "cost": r.metrics.cost_micros / 1_000_000,  # micros ➜ unidades monetarias
            }
        )
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# 5) Métricas específicas que usa google_ads_tab.py
# ────────────────────────────────────────────────────────────
def _get_customer_id(config_path: str | os.PathLike | None = None) -> str:
    """Recupera el customer_id desde env o YAML."""
    cid = os.getenv("GOOGLE_ADS_CUSTOMER_ID")
    if cid:
        return str(cid)

    path = Path(config_path or PROJECT_YAML)
    if path.exists():
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            cid_yaml = data.get("customer_id")
            if cid_yaml:
                return str(cid_yaml)
        except Exception:
            pass
    return ""


# ---------- 5.1  Rendimiento diario ----------
def fetch_daily_performance(start_date: date, end_date: date) -> pd.DataFrame:
    """Spend, clicks y conversions por día."""
    client = load_client_safe()
    customer_id = _get_customer_id()
    if not customer_id:
        raise RuntimeError("❗ GOOGLE_ADS_CUSTOMER_ID no configurado.")

    df = fetch_ads_metrics(client, customer_id, start_date.isoformat(), end_date.isoformat())
    if df.empty:
        return pd.DataFrame(columns=["date", "spend", "clicks", "conversions"])

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            spend=("cost", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .sort_values("date")
    )
    return daily


# ---------- 5.2  Rendimiento por campaña ----------
def fetch_campaign_performance(start_date: date, end_date: date) -> pd.DataFrame:
    """Agrega métricas por campaña."""
    client = load_client_safe()
    customer_id = _get_customer_id()
    if not customer_id:
        raise RuntimeError("❗ GOOGLE_ADS_CUSTOMER_ID no configurado.")

    df = fetch_ads_metrics(client, customer_id, start_date.isoformat(), end_date.isoformat())
    if df.empty:
        return pd.DataFrame(
            columns=["campaign", "spend", "clicks", "conversions", "cpc", "cpa", "roas"]
        )

    agg = (
        df.groupby("campaign", as_index=False)
        .agg(
            spend=("cost", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .sort_values("spend", ascending=False)
    )
    agg["cpc"] = agg.spend / agg.clicks.replace({0: None})
    agg["cpa"] = agg.spend / agg.conversions.replace({0: None})
    agg["roas"] = agg.conversions / agg.spend.replace({0: None})
    return agg


# ---------- 5.3  Resumen + deltas ----------
def fetch_overview(start_date: date, end_date: date) -> Dict[str, float]:
    """Totales y variación porcentual vs periodo anterior."""
    client = load_client_safe()
    customer_id = _get_customer_id()
    if not customer_id:
        raise RuntimeError("❗ GOOGLE_ADS_CUSTOMER_ID no configurado.")

    # Periodo actual
    df_now = fetch_ads_metrics(client, customer_id, start_date.isoformat(), end_date.isoformat())

    current = {
        "spend": df_now.cost.sum(),
        "clicks": df_now.clicks.sum(),
        "conversions": df_now.conversions.sum(),
    } if not df_now.empty else {"spend": 0.0, "clicks": 0, "conversions": 0}

    # Periodo anterior (mismos días inmediatamente previos)
    period_days = (end_date - start_date).days + 1
    prev_start = start_date - timedelta(days=period_days)
    prev_end = start_date - timedelta(days=1)
    df_prev = fetch_ads_metrics(client, customer_id, prev_start.isoformat(), prev_end.isoformat())

    prev = {
        "spend": df_prev.cost.sum(),
        "clicks": df_prev.clicks.sum(),
        "conversions": df_prev.conversions.sum(),
    } if not df_prev.empty else {"spend": 0.0, "clicks": 0, "conversions": 0}

    # Helpers
    def pct(curr: float, old: float) -> float:
        return 0.0 if old == 0 else (curr - old) / old * 100

    # KPIs derivados
    avg_cpc = current["spend"] / current["clicks"] if current["clicks"] else 0.0
    prev_avg_cpc = prev["spend"] / prev["clicks"] if prev["clicks"] else 0.0
    roas = current["conversions"] / current["spend"] if current["spend"] else 0.0
    prev_roas = prev["conversions"] / prev["spend"] if prev["spend"] else 0.0

    return {
        "spend": current["spend"],
        "delta_spend_pct": pct(current["spend"], prev["spend"]),
        "clicks": current["clicks"],
        "delta_clicks_pct": pct(current["clicks"], prev["clicks"]),
        "cpc": avg_cpc,
        "delta_cpc_pct": pct(avg_cpc, prev_avg_cpc),
        "roas": roas,
        "delta_roas_pct": pct(roas, prev_roas),
    }

# -----------------------------------------------------------
# END OF MODULE
# -----------------------------------------------------------
