#!/usr/bin/env python3
"""
Backend ETL for NCR ride bookings dataset.

- Cleans and loads `ncr_ride_bookings.csv` into a SQLite database (default: ride_bookings.db).
- Creates indexes and a convenience view for common analytics.

Usage:
    python backend.py --csv ncr_ride_bookings.csv --db ride_bookings.db

"""
import argparse
import os
import sqlite3
from datetime import datetime

import pandas as pd


NA_VALUES = ["null", "NULL", "NaN", "nan", "",]


def read_and_clean(csv_path: str) -> pd.DataFrame:
    # Read CSV with Python engine to better tolerate odd quoting
    df = pd.read_csv(
        csv_path,
        engine="python",
        na_values=NA_VALUES,
        keep_default_na=True
    )

    # Standardize column names
    df.columns = [c.strip().replace(" ", "_").replace("/", "_") for c in df.columns]

    # Normalize text fields
    text_cols = [
        'Booking_ID','Booking_Status','Customer_ID','Vehicle_Type','Pickup_Location','Drop_Location',
        'Reason_for_cancelling_by_Customer','Driver_Cancellation_Reason','Incomplete_Rides_Reason','Payment_Method'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA})

    # Parse Date and Time; build a single datetime column
    if 'Date' in df.columns:
        # Try multiple date formats
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M:%S').dt.time

    # Combine Date and Time to a single timestamp
    if 'Date' in df.columns:
        if 'Time' in df.columns:
            df['Booking_Timestamp'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce'
            )
        else:
            df['Booking_Timestamp'] = df['Date']

    # Numeric conversions
    numeric_map = {
        'Avg_VTAT': float,
        'Avg_CTAT': float,
        'Cancelled_Rides_by_Customer': 'Int64',
        'Cancelled_Rides_by_Driver': 'Int64',
        'Incomplete_Rides': 'Int64',
        'Booking_Value': float,
        'Ride_Distance': float,
        'Driver_Ratings': float,
        'Customer_Rating': float,
    }
    for col, dtype in numeric_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if dtype == 'Int64':
                df[col] = df[col].astype('Int64')

    # Derivations for analytics
    if 'Booking_Timestamp' in df.columns:
        ts = df['Booking_Timestamp']
        df['Year'] = ts.dt.year
        df['Month'] = ts.dt.to_period('M').astype(str)
        df['Date_Only'] = ts.dt.date
        df['Hour'] = ts.dt.hour
        df['Weekday'] = ts.dt.day_name()
    else:
        # Fallback if timestamp missing
        df['Year'] = pd.NA
        df['Month'] = pd.NA
        df['Date_Only'] = pd.NA
        df['Hour'] = pd.NA
        df['Weekday'] = pd.NA

    # Status normalization groups
    if 'Booking_Status' in df.columns:
        def map_status(x):
            if not isinstance(x, str):
                return 'Unknown'
            x = x.strip()
            xl = x.lower()
            if xl.startswith('cancelled by customer'):
                return 'Cancelled_by_Customer'
            if xl.startswith('cancelled by driver'):
                return 'Cancelled_by_Driver'
            if xl.startswith('no driver found'):
                return 'No_Driver_Found'
            if xl.startswith('incomplete'):
                return 'Incomplete'
            if xl.startswith('completed'):
                return 'Completed'
            return x.replace(' ', '_')
        df['Status_Group'] = df['Booking_Status'].apply(map_status)

    # Trim overly long text values to keep SQLite efficient
    for col in ['Pickup_Location', 'Drop_Location', 'Reason_for_cancelling_by_Customer',
                'Driver_Cancellation_Reason', 'Incomplete_Rides_Reason']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.slice(0, 200)

    return df


def load_to_sqlite(df: pd.DataFrame, db_path: str):
    con = sqlite3.connect(db_path)
    try:
        # Write to SQLite (replace existing table)
        df.to_sql('bookings', con, if_exists='replace', index=False)

        # Create indexes
        cur = con.cursor()
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bookings_ts ON bookings(Booking_Timestamp)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(Status_Group)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bookings_vehicle ON bookings(Vehicle_Type)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bookings_payment ON bookings(Payment_Method)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bookings_pickup ON bookings(Pickup_Location)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bookings_drop ON bookings(Drop_Location)')

        # Convenience view (daily aggregates)
        cur.execute("""
            CREATE VIEW IF NOT EXISTS v_daily_summary AS
            SELECT
                date(Booking_Timestamp) as date,
                COUNT(*) as total_bookings,
                SUM(CASE WHEN Status_Group='Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN Status_Group='Cancelled_by_Customer' THEN 1 ELSE 0 END) as cancelled_by_customer,
                SUM(CASE WHEN Status_Group='Cancelled_by_Driver' THEN 1 ELSE 0 END) as cancelled_by_driver,
                SUM(CASE WHEN Status_Group='No_Driver_Found' THEN 1 ELSE 0 END) as no_driver_found,
                SUM(CASE WHEN Status_Group='Incomplete' THEN 1 ELSE 0 END) as incomplete,
                AVG(Booking_Value) as avg_booking_value,
                AVG(Avg_VTAT) as avg_vtat,
                AVG(Avg_CTAT) as avg_ctat
            FROM bookings
            GROUP BY date(Booking_Timestamp)
            ORDER BY date(Booking_Timestamp)
        """)
        con.commit()
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description='Load NCR ride bookings CSV into SQLite DB')
    parser.add_argument('--csv', default='ncr_ride_bookings.csv', help='Path to CSV file')
    parser.add_argument('--db', default='ride_bookings.db', help='SQLite DB output path')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    print('Reading and cleaning CSV...')
    df = read_and_clean(args.csv)
    print(f"Rows read: {len(df):,}")

    print(f'Loading into SQLite: {args.db} ...')
    load_to_sqlite(df, args.db)
    print('Done. Table `bookings` and view `v_daily_summary` are ready.')


if __name__ == '__main__':
    main()
