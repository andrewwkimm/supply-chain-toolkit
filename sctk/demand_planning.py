"""The Demand Forecaster."""

from typing import Optional, self

import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class DemandForecaster:
    """Demand forecasting model."""

    def __init__(self, n_estimators: int = 100, lead_time: int = 7) -> None:
        """Initializes the demand forecasting model."""
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.scaler = StandardScaler()
        self.lead_time = lead_time
        self._is_fitted = False

    def _prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepares time series features for modeling."""
        return df.with_columns(
            [
                pl.col("quantity").shift(1).alias("lag_1"),
                pl.col("quantity").shift(2).alias("lag_2"),
                pl.col("date").dt.month().alias("month"),
                pl.col("date").dt.weekday().alias("day_of_week"),
            ]
        ).drop_nulls()

    def fit(
        self,
        x: pl.DataFrame,
        y: Optional[pl.Series],
    ) -> self:
        """Fits the demand forecasting model."""
        if y is None:
            if "quantity" not in x.columns:
                raise ValueError(
                    "Target column 'quantity' must be present if y is not provided"
                )
            y = x["quantity"]
            x = x.drop("quantity")

        prepared_data = self._prepare_features(x.with_columns(y=y))

        feature_cols = ["lag_1", "lag_2", "month", "day_of_week"]
        X_train = prepared_data[feature_cols].to_numpy()
        y_train = prepared_data["y"].to_numpy()

        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._is_fitted = True

        return self

    def predict(self, x: pl.DataFrame, periods: int = 30) -> pl.Series:
        """Generates demand forecasts.

        Parameters
        ----------
        x: pl.DataFrame
            Recent historical data for generating forecasts.
        periods: int
            Number of periods to forecast.

        Returns:
            Series of forecasted demand values.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        current_data = self._prepare_features(x)
        predictions = []

        for _ in range(periods):
            last_features = current_data.slice(-1)[
                ["lag_1", "lag_2", "month", "day_of_week"]
            ]
            X_pred = self.scaler.transform(last_features.to_numpy())

            prediction = self.model.predict(X_pred)[0]
            predictions.append(prediction)

            new_row = pl.DataFrame(
                {
                    "date": [current_data["date"].max() + pl.duration(days=1)],
                    "quantity": [prediction],
                    "lag_1": [current_data["quantity"].max()],
                    "lag_2": [current_data["quantity"].slice(-2, 1).max()],
                    "month": [current_data["date"].max().month()],
                    "day_of_week": [current_data["date"].max().weekday()],
                }
            )
            current_data = pl.concat([current_data, new_row])

        return pl.Series(predictions)
