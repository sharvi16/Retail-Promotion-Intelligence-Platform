# CPG Pricing Optimization

Project scaffold for a pricing optimization workflow built around Dunnhumby-style retail data.

## Structure

- `data/raw/`: source CSVs
- `data/processed/`: cleaned parquet outputs from the pipeline
- `notebooks/`: exploration and analysis scripts
- `src/`: reusable pipeline, segmentation, elasticity, and simulation modules
- `app/`: Streamlit dashboard
- `outputs/figures/`: generated charts and visuals

## Next steps

1. Implement the ETL in `src/data_pipeline.py`.
2. Add segmentation, elasticity, and promotion logic in `src/`.
3. Connect the Streamlit app to the processed outputs.
4. Run `python run_pipeline.py` once the modules are implemented.
