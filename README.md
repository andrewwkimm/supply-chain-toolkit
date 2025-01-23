<h1 align="center">
  <img
    src="https://raw.githubusercontent.com/andrewwkimm/supply-chain-toolkit/main/assets/sctk.png" alt="sctk logo">
  <br>
</h1>

**supply-chain-toolkit** (sctk) is a package that offers various tools for supply chain analytics.

## Quick Start
Usage will vary depending on which module is used. Using the `InventoryOptimizer` as an example, a data file with the quanity for a given product is required at minimum.

```python
import polars as pl

from sctk.inventory import InventoryOptimizer


# Load data
df = pl.read_csv('sales_data.csv')

# Calculate demand statistics
demand_std = df['quantity'].std()

# Initialize the inventory optimizer
optimizer = InventoryOptimizer(lead_time=7, service_level=0.95)

# Calculate safety stock
safety_stock = optimizer.calculate_safety_stock(demand_std)

# Calculate reorder point (assuming average daily demand)
average_sales = df['quantity'].mean()
reorder_point = optimizer.calculate_reorder_point(average_sales, safety_stock)

print(f"Safety Stock: {safety_stock}")
print(f"Reorder Point: {reorder_point}")
```
