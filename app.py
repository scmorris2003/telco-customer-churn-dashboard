import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm


# Cache the data load so it doesn't rerun every time you change something
@st.cache_data
def load_data():
    # Adjust path if your CSV is in a subfolder (e.g. data/WA_Fn-UseC_-Telco-Customer-Churn.csv)
    df = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Fix common data issues in this dataset
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])  # Drop the ~11 rows with empty TotalCharges

    # Optional: create a few useful derived columns early
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=[0, 30, 50, 70, 90, 120],
                                      labels=['0-30', '30-50', '50-70', '70-90', '90+'])

    return df


# Load once
df = load_data()

# Title & basic info
st.title("Telco Customer Churn Analytics Dashboard")
st.markdown("Exploring customer acquisition, retention, and churn patterns")

st.header("Dataset Overview")
st.write(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# Show sample
st.subheader("First 5 rows")
st.dataframe(df.head())

# Quick stats
st.subheader("Key Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
col3.metric("Average Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")

# Sidebar filters
st.sidebar.header("Filters")

# Multi-select for categorical columns
internet_service = st.sidebar.multiselect(
    "Internet Service",
    options=df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

contract = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

# Slider for tenure
tenure_range = st.sidebar.slider(
    "Tenure (months)",
    0, int(df['tenure'].max()),
    (0, int(df['tenure'].max()))
)

# Apply filters
filtered_df = df[
    (df['InternetService'].isin(internet_service)) &
    (df['Contract'].isin(contract)) &
    (df['tenure'].between(tenure_range[0], tenure_range[1]))
]

st.write(f"Filtered customers: {len(filtered_df):,}")

import plotly.express as px

st.header("Churn by Contract Type")
fig_contract = px.bar(
    filtered_df.groupby('Contract')['Churn'].mean().reset_index(),
    x='Contract',
    y='Churn',
    title="Churn Rate by Contract",
    labels={'Churn': 'Churn Rate'},
    color='Churn'
)
st.plotly_chart(fig_contract)

st.header("Tenure Distribution by Churn")
fig_tenure = px.histogram(
    filtered_df,
    x='tenure',
    color='Churn',
    barmode='overlay',
    title="Tenure Distribution – Churn vs Non-Churn",
    labels={'tenure': 'Months with Company'}
)
st.plotly_chart(fig_tenure)


st.subheader("Approximate Retention Curve (Survival Proxy)")

# Precompute retention rates for each tenure threshold
max_tenure = int(filtered_df['tenure'].max())
retention_rates = []

for t in range(max_tenure + 1):
    subset = filtered_df[filtered_df['tenure'] >= t]
    if len(subset) > 0:
        rate = 1 - subset['Churn'].mean()
    else:
        rate = np.nan
    retention_rates.append(rate)

retention_df = pd.DataFrame({
    'Tenure Months': range(max_tenure + 1),
    'Retention Rate': retention_rates
})

# Drop NaN rows if any at the end
retention_df = retention_df.dropna()

fig_ret = px.line(
    retention_df,
    x='Tenure Months',
    y='Retention Rate',
    title="Estimated Retention Over Time (Proportion Still Active)",
    labels={'Retention Rate': 'Retention Rate'},
    markers=True
)
fig_ret.update_layout(yaxis_range=[0, 1])  # Force 0-1 scale
st.plotly_chart(fig_ret, width='stretch')


st.header("Key Statistical Insights & Recommendations")

from scipy import stats

# Example: Compare churn between fiber optic and DSL users
fiber = filtered_df[filtered_df['InternetService'] == 'Fiber optic']['Churn']
dsl = filtered_df[filtered_df['InternetService'] == 'DSL']['Churn']

if len(fiber) > 5 and len(dsl) > 5:
    t_stat, p_val = stats.ttest_ind(fiber, dsl, equal_var=False)
    st.write(f"**Fiber Optic vs DSL Churn Comparison**")
    st.write(f"- Fiber churn rate: {fiber.mean():.1%}")
    st.write(f"- DSL churn rate: {dsl.mean():.1%}")
    st.write(f"- p-value (Welch's t-test): {p_val:.4f}")
    if p_val < 0.05:
        st.warning("Statistically significant difference (p < 0.05) — Fiber users churn more.")
        st.markdown("**Action for Engineering/Business**: Investigate Fiber onboarding or service quality issues.")
    else:
        st.success("No significant difference detected.")
else:
    st.info("Not enough data in one or more groups for statistical test.")

st.header("Churn Monitoring & Alerts")

# Simple anomaly example: flag if churn in last "period" (using tenure as proxy) is much higher
recent = filtered_df[filtered_df['tenure'] <= 6]  # "newer" customers
overall_churn = df['Churn'].mean()
recent_churn = recent['Churn'].mean()

delta = recent_churn - overall_churn

st.metric(
    label="New Customer (≤6 mo) Churn Rate",
    value=f"{recent_churn:.1%}",
    delta=f"{delta:+.1%} vs overall",
    delta_color="inverse" if delta > 0 else "normal"
)

if delta > 0.05:
    st.error(
        "🚨 ALERT: Elevated churn in newest customers — potential regression in acquisition/onboarding experience.")
    st.markdown("**Recommended Action**: Review recent changes to signup flow or first-month support.")

from prophet import Prophet

st.header("Customer Acquisition / Survival Forecast (Proxy via Tenure)")

# Aggregate "new customers" by tenure month as proxy
acquisition_ts = filtered_df.groupby('tenure').size().reset_index(name='NewCustomers')
acquisition_ts = acquisition_ts.rename(columns={'tenure': 'ds', 'NewCustomers': 'y'})
acquisition_ts['ds'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(acquisition_ts['ds'], unit='D')  # Fake dates

if len(acquisition_ts) > 10:
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False)
    m.fit(acquisition_ts)
    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)

    # Rename for clear plotting
    plot_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    plot_forecast = plot_forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted New Customers',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })

    historical_plot = acquisition_ts[['ds', 'y']].copy()
    historical_plot = historical_plot.rename(columns={
        'ds': 'Date',
        'y': 'Actual New Customers (by tenure)'
    })

    fig_forecast = px.line(
        plot_forecast,
        x='Date',
        y='Predicted New Customers',
        title="Forecast: Estimated Future New Customers (Tenure-Based Proxy)",
        labels={'Predicted New Customers': 'Predicted New Customers'}
    )

    fig_forecast.add_scatter(
        x=historical_plot['Date'],
        y=historical_plot['Actual New Customers (by tenure)'],
        mode='markers',
        name='Historical (tenure proxy)'
    )

    # Optional: add uncertainty band
    fig_forecast.add_scatter(
        x=plot_forecast['Date'],
        y=plot_forecast['Lower Bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    )
    fig_forecast.add_scatter(
        x=plot_forecast['Date'],
        y=plot_forecast['Upper Bound'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0),
        name='Uncertainty Range'
    )

    st.plotly_chart(fig_forecast, width='stretch')

    st.caption("Note: This uses customer tenure (months) as a rough proxy for time since signup. "
               "In real data, replace tenure with actual acquisition dates for accurate monthly forecasts. Also if you are filtering to the full data set, there are outliers so it is recommended to filter to 5-65 Tenure for more reliable forecasting.")
else:
    st.info("Not enough distinct tenure values for reliable forecasting.")

st.header("Regime Change Detection (Structural Break Analysis)")


# Use Chow test on retention_df (from earlier code)
def chow_test(data, break_point):
    y = data['Retention Rate']
    x = data['Tenure Months']
    n = len(data)

    y1, y2 = y[:break_point], y[break_point:]
    x1, x2 = x[:break_point], x[break_point:]

    model1 = sm.OLS(y1, sm.add_constant(x1)).fit()
    model2 = sm.OLS(y2, sm.add_constant(x2)).fit()
    model_full = sm.OLS(y, sm.add_constant(x)).fit()

    ssr_full = model_full.ssr
    ssr1 = model1.ssr
    ssr2 = model2.ssr
    k = 2
    chow_stat = ((ssr_full - (ssr1 + ssr2)) / k) / ((ssr1 + ssr2) / (n - 2 * k))
    p_val = 1 - stats.f.cdf(chow_stat, k, n - 2 * k)

    return chow_stat, p_val


potential_breaks = range(5, len(retention_df) - 5)
chow_results = [(bp, *chow_test(retention_df, bp)) for bp in potential_breaks]
chow_df = pd.DataFrame(chow_results, columns=['Break Point (Months)', 'Chow Stat', 'p-value'])
chow_df = chow_df.sort_values('Chow Stat', ascending=False).head(5)

st.subheader("Top Detected Breaks")
st.dataframe(chow_df.style.highlight_max(subset=['Chow Stat'], color='lightgreen'))

top_break = int(chow_df.iloc[0]['Break Point (Months)'])

# Highlight on the retention plot
fig_ret.add_vline(x=top_break, line_dash="dash", line_color="red",
                  annotation_text="Detected Break", annotation_position="top left")

st.plotly_chart(fig_ret, width='stretch')  # Re-show updated plot

# Slopes before/after
before = retention_df.iloc[:top_break]
after = retention_df.iloc[top_break:]
slope_before = np.polyfit(before['Tenure Months'], before['Retention Rate'], 1)[0]
slope_after = np.polyfit(after['Tenure Months'], after['Retention Rate'], 1)[0]

st.write(
    f"**Key Insight**: Break at ~{top_break} months. Slope before: {slope_before:.4f}/month; after: {slope_after:.4f}/month.")
st.markdown(
    "**Business Value**: If forecast flips positive post-break, focus retention efforts just before this point (e.g., loyalty offers at 6-8 months).")