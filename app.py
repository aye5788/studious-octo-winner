import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Auto-cleaning function
def auto_clean_data(df, correlation_threshold=0.95):
    """Automatically cleans the dataset based on detected issues."""
    
    # Detect missing values and fill accordingly
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())  # Numeric: Median
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])  # Categorical: Mode
            elif np.issubdtype(df[col].dtype, np.datetime64):
                df[col] = df[col].ffill()  # Date: Forward fill

    # Convert dates to datetime format
    for col in df.select_dtypes(include=["object"]):
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass  # Ignore if not convertible

    # Handle high correlation
    corr_matrix = df.corr()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                to_drop.add(corr_matrix.columns[i])
    df = df.drop(columns=list(to_drop), errors='ignore')

    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

# Streamlit UI
st.title("🚀 Auto Data Cleaning App")
st.write("Upload a CSV file, and the app will clean it automatically.")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Show raw data
    st.subheader("📂 Raw Dataset")
    st.write(df.head())

    # Generate profiling report
    st.subheader("📊 Data Profiling Report")
    profile = ProfileReport(df, explorative=True)
    st_profile_report(profile)

    # Clean the dataset
    cleaned_df = auto_clean_data(df)

    # Show cleaned data
    st.subheader("🧹 Cleaned Dataset")
    st.write(cleaned_df.head())

    # Download cleaned dataset
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

