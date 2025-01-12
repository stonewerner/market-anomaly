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
    
    # Create percent change feature
    df['Pct_Change'] = df['MXWO Index'].pct_change(periods=-1) * 100
    
    # Create crash indicator
    df['CRASH'] = (df['Pct_Change'] <= -5).astype(int)
    
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

# Display key metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Key Market Indicators")
    # Adjust these based on your most important columns
    st.metric("VIX Index", f"{selected_data['VIX Index']:.2f}")
    # st.metric("VIX 3 Week Lag", f"{selected_data['VIX_Index_lag_3']:.2f}")
    st.metric("CRY Index", f"{selected_data['CRY Index']:.2f}")
    st.metric("EONIA Index", f"{selected_data['EONIA Index']:.2f}")
    st.metric("JPY Currncy", f"{selected_data['JPY Curncy']:.2f}")

# Make prediction
prediction_proba = xgb_model.predict_proba(selected_data.values.reshape(1, -1))[0]

# Create gauge chart for crash probability
with col2:
    st.subheader("Crash Probability")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probability of Crash"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig)

# Display feature importance
st.subheader("Top 3 Contributing Factors")
feature_importance = pd.DataFrame({
    'feature': df.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(3)

st.bar_chart(feature_importance.set_index('feature'))