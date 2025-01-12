import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
from datetime import datetime

def preprocess_data(df):
    """
    Applies all preprocessing steps used in training:
    - Drops specified columns
    - Creates percent change feature
    - Creates crash indicator
    - Adds lagged features for specified columns
    """
    # Drop unwanted columns
    if 'LLL1 Index' in df.columns:
        df.drop(columns=['LLL1 Index'], inplace=True)
   
    
    # Create lagged features
    columns_to_lag = ['VIX Index', 'MXWO Index']
    num_lags = 3
    
    for col in columns_to_lag:
        for lag in range(1, num_lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN values created by lagging
    df = df.dropna()
    
    return df

# Load the model
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load your saved model
xgb_model = load_model('xgb_model.pkl')

# Set page title
st.title("Market Crash Prediction Dashboard")

# Load your data
df = pd.read_csv('FinancialMarketDataFormatted.csv')

# Convert the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as the index
df.set_index('Date', inplace=True)

df = preprocess_data(df)

# Now create the selectbox with the proper dates
available_dates = df.index.tolist()
print("Available dates check:", available_dates[:5])  # Let's verify the dates look right

selected_date = st.selectbox(
    "Select a week",
    options=available_dates,
    format_func=lambda x: x.strftime('%Y-%m-%d'),
    index=0
)

# Now selected_date will be directly usable as an index
selected_data = df.loc[selected_date]

# Find the closest date in our dataset
closest_date = df.index[df.index.get_indexer([pd.Timestamp(selected_date)], method='nearest')[0]]
selected_data = df.loc[closest_date]

# Create three columns for better layout
col1, col2, col3 = st.columns([1, 1, 1])

# Key Market Indicators in first column
with col1:
    st.subheader("Key Market Indicators")
    st.metric("VIX Index", f"{selected_data['VIX Index']:.2f}")
    st.metric("VIX 3 Week Lag", f"{selected_data['VIX Index_lag_3']:.2f}")
    st.metric("CRY Index", f"{selected_data['CRY Index']:.2f}")
    st.metric("EONIA Index", f"{selected_data['EONIA Index']:.2f}")
    st.metric("JPY Currncy", f"{selected_data['JPY Curncy']:.2f}")

# Make prediction
prediction_proba = xgb_model.predict_proba(selected_data.values.reshape(1, -1))[0]

# Absolute Crash Probability in second column
with col2:
    st.subheader("Crash Probability")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba[1] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': ""},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True)

# First, get predictions for all data
all_predictions = xgb_model.predict_proba(df.values)[:, 1]  # Get probability of crash for all rows
min_prob = all_predictions.min()
max_prob = all_predictions.max()

# Relative Risk in third column
with col3:
    st.subheader("Relative Risk")
    relative_risk = ((prediction_proba[1] - min_prob) / (max_prob - min_prob)) * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=relative_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': ""},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=75, b=20),
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True)

# Create a new row for the feature importance
st.subheader("Top 3 Contributing Factors")
feature_importance = pd.DataFrame({
    'feature': df.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(3)

# Use a container to center the bar chart
container = st.container()
with container:
    chart = st.bar_chart(feature_importance.set_index('feature'))