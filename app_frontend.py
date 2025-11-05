#!/usr/bin/env python3
"""
Streamlit dashboard for NCR ride bookings.
Run with:
    streamlit run app_frontend.py -- --db ride_bookings.db

Optional flags:
    --db <path> : SQLite DB produced by backend.py (default: ride_bookings.db)
"""
import argparse
import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------- Utilities ----------------------
@st.cache_data(show_spinner=False)
def load_frame(query: str, db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, con, parse_dates=['Booking_Timestamp'])
    finally:
        con.close()


def get_min_max_dates(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT MIN(date(Booking_Timestamp)), MAX(date(Booking_Timestamp)) FROM bookings")
        row = cur.fetchone()
        return row[0], row[1]
    finally:
        con.close()
        

# Safe SQL list constructor for IN (...) clauses
def _sql_list(values):
    """Return a SQL IN-list like ('A','B'), with single quotes escaped for SQLite."""
    if not values:
        return "(NULL)"  # never matches, useful fallback
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ",".join(escaped) + ")"



# ---------------------- App ----------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--db', default='ride_bookings.db')
args, _ = parser.parse_known_args()
DB_PATH = args.db

st.set_page_config(page_title='NCR Ride Bookings Dashboard', layout='wide')

st.title('🚖 NCR Ride Bookings — Analytics Dashboard')

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at '{DB_PATH}'. Run backend.py first to generate it.")
    st.stop()

min_d, max_d = get_min_max_dates(DB_PATH)
if min_d is None or max_d is None:
    st.warning('No data in database.')
    st.stop()

min_d = pd.to_datetime(min_d).date()
max_d = pd.to_datetime(max_d).date()

# Sidebar filters
st.sidebar.header('Filters')
start_date, end_date = st.sidebar.date_input('Date range', value=(min_d, max_d), min_value=min_d, max_value=max_d)

vehicle_types = load_frame("SELECT DISTINCT Vehicle_Type FROM bookings WHERE Vehicle_Type IS NOT NULL ORDER BY 1", DB_PATH)['Vehicle_Type'].dropna().tolist()
sel_vehicle = st.sidebar.multiselect('Vehicle Type', vehicle_types, default=vehicle_types)

status_groups = load_frame("SELECT DISTINCT Status_Group FROM bookings ORDER BY 1", DB_PATH)['Status_Group'].dropna().tolist()
sel_status = st.sidebar.multiselect('Status', status_groups, default=status_groups)

payments = load_frame("SELECT DISTINCT Payment_Method FROM bookings WHERE Payment_Method IS NOT NULL ORDER BY 1", DB_PATH)['Payment_Method'].dropna().tolist()
sel_payment = st.sidebar.multiselect('Payment Method', payments, default=payments)

# Core query with filters
where_clauses = [
    f"date(Booking_Timestamp) BETWEEN '{start_date}' AND '{end_date}'"
]
if sel_vehicle:
    where_clauses.append(f"Vehicle_Type IN {_sql_list(sel_vehicle)}")

if sel_status:
    where_clauses.append(f"Status_Group IN {_sql_list(sel_status)}")

if sel_payment:
    where_clauses.append(f"(Payment_Method IS NULL OR Payment_Method IN {_sql_list(sel_payment)})")

where_sql = " WHERE " + " AND ".join(where_clauses)

base_query = f"""
SELECT *
FROM bookings
{where_sql}
"""

df = load_frame(base_query, DB_PATH)

if df.empty:
    st.warning('No rows match the selected filters.')
    st.stop()

# Ensure numeric types for metrics
for col in ['Booking_Value', 'Avg_VTAT', 'Avg_CTAT', 'Ride_Distance', 'Driver_Ratings', 'Customer_Rating']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# KPIs
total = len(df)
completed = (df['Status_Group'] == 'Completed').sum()
cancelled_cust = (df['Status_Group'] == 'Cancelled_by_Customer').sum()
cancelled_dr = (df['Status_Group'] == 'Cancelled_by_Driver').sum()
ndf = (df['Status_Group'] == 'No_Driver_Found').sum()
incomplete = (df['Status_Group'] == 'Incomplete').sum()
completion_rate = completed / total if total else 0

avg_value = df['Booking_Value'].mean(skipna=True)
avg_vtat = df['Avg_VTAT'].mean(skipna=True)
avg_ctat = df['Avg_CTAT'].mean(skipna=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric('Total Bookings', f"{total:,}")
c2.metric('Completion Rate', f"{completion_rate:.1%}")
c3.metric('Cancelled (Customer)', f"{cancelled_cust:,}")
c4.metric('Cancelled (Driver)', f"{cancelled_dr:,}")
c5.metric('No Driver Found', f"{ndf:,}")
c6.metric('Incomplete', f"{incomplete:,}")

c7, c8, c9 = st.columns(3)
c7.metric('Avg Booking Value', f"₹{avg_value:,.0f}" if pd.notna(avg_value) else '—')
c8.metric('Avg VTAT (min)', f"{avg_vtat:.1f}" if pd.notna(avg_vtat) else '—')
c9.metric('Avg CTAT (min)', f"{avg_ctat:.1f}" if pd.notna(avg_ctat) else '—')

st.divider()

# Time series
if 'Booking_Timestamp' in df.columns:
    ts = df.copy()
    ts['Date'] = pd.to_datetime(ts['Booking_Timestamp']).dt.date
    by_day = ts.groupby('Date').size().reset_index(name='Bookings')
    fig_ts = px.line(by_day, x='Date', y='Bookings', title='Bookings Over Time', markers=True)
    st.plotly_chart(fig_ts, use_container_width=True)

# Status distribution
status_counts = df['Status_Group'].value_counts(dropna=False).reset_index()
status_counts.columns = ['Status', 'Count']
fig_status = px.bar(status_counts, x='Status', y='Count', color='Status', title='Status Breakdown')
st.plotly_chart(fig_status, use_container_width=True)

# Vehicle type distribution
if 'Vehicle_Type' in df.columns:
    veh_counts = df['Vehicle_Type'].value_counts(dropna=False).reset_index()
    veh_counts.columns = ['Vehicle Type', 'Count']
    fig_veh = px.pie(veh_counts, names='Vehicle Type', values='Count', title='Vehicle Type Mix', hole=0.3)
    st.plotly_chart(fig_veh, use_container_width=True)

# Heatmap by hour x weekday
if 'Hour' in df.columns and 'Weekday' in df.columns:
    # Order weekdays
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    hm = df[['Hour', 'Weekday']].dropna().copy()
    hm['Weekday'] = pd.Categorical(hm['Weekday'], categories=weekday_order, ordered=True)
    pivot = hm.value_counts().reset_index(name='Bookings')
    fig_hm = px.density_heatmap(pivot, x='Hour', y='Weekday', z='Bookings', title='Bookings Heatmap (Hour x Weekday)', color_continuous_scale='Blues')
    st.plotly_chart(fig_hm, use_container_width=True)

# Top locations
if 'Pickup_Location' in df.columns:
    top_pickup = df['Pickup_Location'].value_counts().head(10).reset_index()
    top_pickup.columns = ['Pickup Location', 'Trips']
    fig_pk = px.bar(top_pickup, x='Trips', y='Pickup Location', orientation='h', title='Top 10 Pickup Locations')
    st.plotly_chart(fig_pk, use_container_width=True)

if 'Drop_Location' in df.columns:
    top_drop = df['Drop_Location'].value_counts().head(10).reset_index()
    top_drop.columns = ['Drop Location', 'Trips']
    fig_dp = px.bar(top_drop, x='Trips', y='Drop Location', orientation='h', title='Top 10 Drop Locations')
    st.plotly_chart(fig_dp, use_container_width=True)

# Booking value and distance distributions
dist_cols = []
if 'Booking_Value' in df.columns:
    dist_cols.append('Booking_Value')
if 'Ride_Distance' in df.columns:
    dist_cols.append('Ride_Distance')

if dist_cols:
    tabs = st.tabs([c.replace('_',' ') for c in dist_cols])
    for t, col in zip(tabs, dist_cols):
        with t:
            fig_hist = px.histogram(df, x=col, nbins=40, title=f'Distribution of {col.replace("_"," ")}', marginal='box')
            st.plotly_chart(fig_hist, use_container_width=True)

# Ratings
if 'Driver_Ratings' in df.columns or 'Customer_Rating' in df.columns:
    rcols = []
    if 'Driver_Ratings' in df.columns:
        rcols.append('Driver_Ratings')
    if 'Customer_Rating' in df.columns:
        rcols.append('Customer_Rating')
    tabs = st.tabs(['Driver Ratings', 'Customer Ratings'][:len(rcols)])
    for t, col in zip(tabs, rcols):
        with t:
            fig = px.violin(df, y=col, box=True, points='all', title=f'{col.replace("_"," ")}')
            st.plotly_chart(fig, use_container_width=True)

# Cancellation reasons
if 'Status_Group' in df.columns:
    canc = df[df['Status_Group'].isin(['Cancelled_by_Customer','Cancelled_by_Driver'])]
    if not canc.empty:
        # Choose the appropriate reason column per row
        canc = canc.copy()
        canc['Reason'] = np.where(
            canc['Status_Group'] == 'Cancelled_by_Customer',
            canc.get('Reason_for_cancelling_by_Customer'),
            canc.get('Driver_Cancellation_Reason')
        )
        top_reasons = canc['Reason'].dropna().value_counts().head(10).reset_index()
        if not top_reasons.empty:
            top_reasons.columns = ['Reason', 'Count']
            fig_cr = px.bar(top_reasons, x='Count', y='Reason', orientation='h', title='Top Cancellation Reasons')
            st.plotly_chart(fig_cr, use_container_width=True)

st.caption('Data source: uploaded CSV; backend stores a cleaned copy in SQLite. Built with Streamlit + Plotly.')
