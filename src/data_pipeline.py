"""
Data Pipeline for Dunnhumby - The Complete Journey
Loads, cleans, merges, and engineers features from raw CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw_tables():
    """Load all raw Dunnhumby CSV files."""
    print("Loading raw tables...")

    transactions = pd.read_csv(RAW_DIR / "transaction_data.csv")
    products = pd.read_csv(RAW_DIR / "product.csv")
    demographics = pd.read_csv(RAW_DIR / "hh_demographic.csv")
    coupons = pd.read_csv(RAW_DIR / "coupon.csv")
    coupon_redemptions = pd.read_csv(RAW_DIR / "coupon_redempt.csv")
    causal = pd.read_csv(RAW_DIR / "causal_data.csv")
    campaigns = pd.read_csv(RAW_DIR / "campaign_table.csv")
    campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")

    print(f"  Transactions:        {transactions.shape}")
    print(f"  Products:            {products.shape}")
    print(f"  Demographics:        {demographics.shape}")
    print(f"  Coupons:             {coupons.shape}")
    print(f"  Coupon Redemptions:  {coupon_redemptions.shape}")
    print(f"  Causal (promo):      {causal.shape}")
    print(f"  Campaigns:           {campaigns.shape}")
    print(f"  Campaign Desc:       {campaign_desc.shape}")

    return {
        "transactions": transactions,
        "products": products,
        "demographics": demographics,
        "coupons": coupons,
        "coupon_redemptions": coupon_redemptions,
        "causal": causal,
        "campaigns": campaigns,
        "campaign_desc": campaign_desc,
    }


def clean_transactions(df):
    """Clean transaction data."""
    df = df.copy()

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df)} duplicate transaction rows")

    # Drop rows with negative or zero sales (returns/errors)
    df = df[df["SALES_VALUE"] > 0].copy()
    df = df[df["QUANTITY"] > 0].copy()

    # Compute unit price
    df["UNIT_PRICE"] = df["SALES_VALUE"] / df["QUANTITY"]

    # Compute total discount per transaction line
    df["TOTAL_DISCOUNT"] = (
        df["RETAIL_DISC"].abs() + df["COUPON_DISC"].abs() + df["COUPON_MATCH_DISC"].abs()
    )

    # Compute pre-discount price (shelf price)
    df["SHELF_PRICE"] = df["SALES_VALUE"] + df["TOTAL_DISCOUNT"]
    df["SHELF_UNIT_PRICE"] = df["SHELF_PRICE"] / df["QUANTITY"]

    # Discount depth as a fraction
    df["DISCOUNT_DEPTH"] = np.where(
        df["SHELF_PRICE"] > 0,
        df["TOTAL_DISCOUNT"] / df["SHELF_PRICE"],
        0,
    )

    # Flag: was this item bought on any discount?
    df["ON_PROMO"] = (df["TOTAL_DISCOUNT"] > 0).astype(int)

    # Discount source flags
    df["HAS_RETAIL_DISC"] = (df["RETAIL_DISC"].abs() > 0).astype(int)
    df["HAS_COUPON_DISC"] = (df["COUPON_DISC"].abs() > 0).astype(int)

    return df


def clean_products(df):
    """Clean product table."""
    df = df.copy()
    df = df.drop_duplicates(subset=["PRODUCT_ID"])

    # Fill missing descriptions
    for col in ["DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]:
        df[col] = df[col].fillna("UNKNOWN")

    # Standardize text
    for col in ["DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]:
        df[col] = df[col].str.strip().str.upper()

    return df


def clean_demographics(df):
    """Clean household demographics."""
    df = df.copy()
    df = df.drop_duplicates(subset=["household_key"])

    # Create numeric income mapping for modeling
    income_map = {
        "Under 15K": 10_000,
        "15-24K": 20_000,
        "25-34K": 30_000,
        "35-49K": 42_000,
        "50-74K": 62_000,
        "75-99K": 87_000,
        "100-124K": 112_000,
        "125-149K": 137_000,
        "150-174K": 162_000,
        "175-199K": 187_000,
        "200-249K": 225_000,
        "250K+": 275_000,
    }
    df["INCOME_NUMERIC"] = df["INCOME_DESC"].map(income_map)

    # Household size as numeric
    df["HH_SIZE_NUMERIC"] = pd.to_numeric(
        df["HOUSEHOLD_SIZE_DESC"].str.replace("+", "", regex=False),
        errors="coerce",
    )

    return df


def merge_master_table(tables):
    """Create the master analysis table by merging transactions + products + demographics."""
    txn = tables["transactions"]
    prod = tables["products"]
    demo = tables["demographics"]
    causal = tables["causal"]

    # Merge transactions with product info
    master = txn.merge(prod, on="PRODUCT_ID", how="left")

    # Merge with demographics (left join — not all households have demographics)
    master = master.merge(demo, on="household_key", how="left")

    # Merge with causal/promo data
    causal_clean = causal.drop_duplicates(subset=["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
    master = master.merge(
        causal_clean[["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"]],
        on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
        how="left",
    )

    # Fill missing display/mailer with 0 (no promo exposure)
    master["display"] = master["display"].fillna(0)
    master["mailer"] = master["mailer"].fillna(0)
    master["display"] = pd.to_numeric(master["display"], errors="coerce").fillna(0)
    master["mailer"] = pd.to_numeric(master["mailer"], errors="coerce").fillna(0)

    # Promo exposure flag
    master["HAD_DISPLAY"] = (master["display"] > 0).astype(int)
    master["HAD_MAILER"] = (master["mailer"] > 0).astype(int)

    print(f"  Master table: {master.shape}")
    return master


def engineer_time_features(df):
    """Add time-based features."""
    df = df.copy()

    # Week number is relative (1-102 over 2 years)
    # Create quarter and month proxies
    df["QUARTER"] = ((df["WEEK_NO"] - 1) // 13) + 1
    df["MONTH_PROXY"] = ((df["WEEK_NO"] - 1) // 4) + 1
    df["YEAR_HALF"] = np.where(df["WEEK_NO"] <= 52, 1, 2)

    # Is this a "festive" period? (approximate Q4 equivalent = weeks 40-52, 92-102)
    festive_weeks = list(range(40, 53)) + list(range(92, 103))
    df["IS_FESTIVE"] = df["WEEK_NO"].isin(festive_weeks).astype(int)

    return df


def build_rfm_table(master):
    """Build RFM (Recency, Frequency, Monetary) table per household."""
    max_week = master["WEEK_NO"].max()

    rfm = master.groupby("household_key").agg(
        recency=("WEEK_NO", lambda x: max_week - x.max()),
        frequency=("BASKET_ID", "nunique"),
        monetary=("SALES_VALUE", "sum"),
        total_items=("QUANTITY", "sum"),
        unique_products=("PRODUCT_ID", "nunique"),
        unique_departments=("DEPARTMENT", "nunique"),
        avg_basket_value=("SALES_VALUE", "mean"),
        avg_discount_depth=("DISCOUNT_DEPTH", "mean"),
        promo_purchase_rate=("ON_PROMO", "mean"),
        total_discount_received=("TOTAL_DISCOUNT", "sum"),
        weeks_active=("WEEK_NO", "nunique"),
    ).reset_index()

    # Derived metrics
    rfm["avg_spend_per_visit"] = rfm["monetary"] / rfm["frequency"]
    rfm["discount_sensitivity"] = rfm["total_discount_received"] / rfm["monetary"].replace(0, 1)

    return rfm


def run_pipeline():
    """Execute the full data pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    tables = load_raw_tables()

    # Step 2: Clean
    print("\nCleaning tables...")
    tables["transactions"] = clean_transactions(tables["transactions"])
    tables["products"] = clean_products(tables["products"])
    tables["demographics"] = clean_demographics(tables["demographics"])

    # Step 3: Merge
    print("\nMerging master table...")
    master = merge_master_table(tables)

    # Step 4: Time features
    print("\nEngineering time features...")
    master = engineer_time_features(master)

    # Step 5: RFM
    print("\nBuilding RFM table...")
    rfm = build_rfm_table(master)

    # Step 6: Save
    print("\nSaving processed data...")
    master.to_parquet(PROCESSED_DIR / "master_transactions.parquet", index=False)
    rfm.to_parquet(PROCESSED_DIR / "rfm_table.parquet", index=False)

    # Save demographics separately for quick access
    tables["demographics"].to_parquet(PROCESSED_DIR / "demographics.parquet", index=False)

    # Save coupon redemption data for promo analysis
    tables["coupon_redemptions"].to_parquet(
        PROCESSED_DIR / "coupon_redemptions.parquet", index=False
    )

    print("\nPipeline complete.")
    print(f"  Master table:  {master.shape}")
    print(f"  RFM table:     {rfm.shape}")

    return master, rfm


if __name__ == "__main__":
    run_pipeline()