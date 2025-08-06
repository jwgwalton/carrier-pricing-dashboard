import numpy as np
import pandas as pd
from scipy.stats import iqr, gaussian_kde
from scipy.signal import find_peaks
from typing import Optional, Union


def identify_optimal_price(
    df: pd.DataFrame,
    route_key: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    target_bandwidth: float = 5
) -> Union[float, None]:
    """
    Identifies the optimal (baseline) carrier price for a given route_key and vehicle_type
    using KDE peak detection or fallback to most common price band.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'carrier_price', 'route_key', and 'vehicle_type'.
        route_key (str, optional): Route key to filter. If None, assumes df is pre-filtered.
        vehicle_type (str, optional): Vehicle type to filter. If None, assumes df is pre-filtered.
        target_bandwidth (float): Target bandwidth for KDE smoothing.

    Returns:
        float or None: Optimal carrier price (baseline) or None if not enough data.
    """
    if route_key:
        df = df[df["route_key"] == route_key]
    if vehicle_type:
        df = df[df["vehicle_type"] == vehicle_type]

    df = df.copy()
    df["carrier_price"] = df["carrier_price"].astype(float)
    costs = df["carrier_price"].dropna().values

    if len(costs) < 5:
        return None

    cost_std = np.std(costs)
    cost_iqr = iqr(costs)
    spread = cost_iqr if cost_iqr > 0 else cost_std
    if spread == 0:
        return None

    bw_method = target_bandwidth / spread
    kde = gaussian_kde(costs, bw_method=bw_method)

    cost_range = np.linspace(costs.min(), costs.max(), 1000)
    density = kde(cost_range)

    peaks, _ = find_peaks(density)
    density_threshold = 0.1 * len(costs) / (cost_range.max() - cost_range.min())

    # First valid KDE peak above threshold
    for idx in peaks:
        if density[idx] >= density_threshold:
            return float(cost_range[idx])

    # Fallback: most common price band
    df["cost_group"] = (df["carrier_price"] // target_bandwidth) * target_bandwidth
    group_counts = df["cost_group"].value_counts().sort_index()
    if not group_counts.empty:
        return float(group_counts.idxmax())

    return None
