# seminar:

# Allgemeine Informationen zum Datensatz: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

- 4.500.000 Rows in Train
- 350.000 Rows in Eval
- 90 Tage
- verschiedene SKUs

# Bedeutung Spalten:
IDs: selbsterklärend -> series_id: Eindeutige ID für jede Kombination aus Geschäft und Produkt
dt: Datum
day_idx: Der Tagesindex (Anzahl Tage seit Beginn der Aufzeichnung)

sale_amount: The daily sales amount after global normalization (Multiplied by a specific coefficient)
hours_sale: The hourly sales amount after global normalization (Multiplied by a specific coefficient) between 0:00 and 24:00
stock_hour6_22_cnt: The number of out-of-stock hours between 6:00 and 22:00
hours_stock_status: The hourly stock status (0 for out-of-stock, 1 for in-stock) between 0:00 and 24:00

discount: The discount rate (1.0 means no discount, 0.9 means 10% off)
holiday_flag	int32	Holiday indicator -> Chinesische Arbeitstage 0 und Feiertage 1
activity_flag	int32	Activity indicator
precpt	float64	The total precipitation
avg_temperature	float64	The average temperature
avg_humidity	float64	The average humidity
avg_wind_level	float64	The average wind force

is_censored: The censoring indicator (0 for uncensored, 1 for censored)
censoring_severity: Prozent der Zensierung (0.0 bis 1.0)
sales_lag1: Lag-1-Verkauf (Verkauf am vorherigen Tag)
sales_lag7: Lag-7-Verkauf (Verkauf am gleichen Tag der Vorwoche)
sales_roll7: 7-Tage gleitender Durchschnitt der Verkäufe
sales_roll28: 28-Tage gleitender Durchschnitt der Verkäufe
