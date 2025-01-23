"""The Inventory Optimizer."""

import numpy as np
from scipy import stats


class InventoryOptimizer:
    """Calculates optimal inventory parameters."""

    def __init__(self, lead_time: int = 7, service_level: float = 0.95) -> None:
        """Initializes inventory optimization parameters.

        Args:
            lead_time: Days between order and delivery
            service_level: Desired probability of avoiding stockout
        """
        self.lead_time = lead_time
        self.service_level = service_level

    def calculate_safety_stock(self, demand_std: float) -> float:
        """Calculates safety stock using service level z-score.

        Parameters
        ----------
        demand_mean:
            Average expected demand.
        demand_std:
            Standard deviation of demand.

        Returns:
            Safety stock level
        """
        z_score = stats.norm.ppf(self.service_level)
        safety_stock_level = z_score * demand_std * np.sqrt(self.lead_time)
        return safety_stock_level

    def calculate_reorder_point(
        self, average_sales: float, safety_stock: float
    ) -> float:
        """Calculates the reorder point considering lead time.

        Parameters
        ----------
        average_sales: float
            Average daily or weekly sales.
        safety_stock: float
            Calculated safety stock level.

        Returns
        -------
            reorder_point
        """
        reorder_point = (average_sales * self.lead_time) + safety_stock
        return reorder_point
