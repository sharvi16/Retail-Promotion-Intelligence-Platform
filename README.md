# Retail Promotion Intelligence Platform

Project scaffold for a retail promotion intelligence workflow built around Dunnhumby-style retail data.

## Structure

- `data/raw/`: source CSVs
- `data/processed/`: cleaned parquet outputs from the pipeline
- `notebooks/`: exploration and analysis scripts
- `src/`: reusable pipeline, segmentation, elasticity, simulation, and optimization modules
- `app/`: Streamlit dashboard
- `outputs/figures/`: generated charts and visuals

## Next steps

1. Run `python run_pipeline.py` to refresh processed tables and charts.
2. Launch the dashboard with `streamlit run app/streamlit_app.py`.
3. Adjust the budget optimizer or demand-driver view if you want a different planning lens.
