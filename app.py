import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for a premium look
st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and calculate the scaler dynamically
@st.cache_resource
def load_model_and_scaler():
    from sklearn.preprocessing import StandardScaler
    try:
        model = joblib.load('ridge_model.pkl')
        df = pd.read_csv('Life Expectancy Data.csv')
        
        FEATURES = [
            'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 
            'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 
            'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 
            ' thinness  1-19 years', ' thinness 5-9 years', 
            'Income composition of resources', 'Schooling'
        ]
        
        # The model requires scaled features. Fit a StandardScaler to the original data.
        X_all = df[FEATURES].fillna(df[FEATURES].median())
        scaler = StandardScaler()
        scaler.fit(X_all)
        
        return model, scaler, FEATURES
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None

model, scaler, FEATURES = load_model_and_scaler()

st.title("🌍 Life Expectancy Prediction Portal")
st.markdown("Predict the life expectancy of a demographic profile using our advanced Ridge Regression Model.")

tab1, tab2 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction (CSV)"])

with tab1:
    st.subheader("Enter Key Demographic & Health Indicators")
    
    # Pre-fill all features with their median values from the dataset
    default_values = {
        'Adult Mortality': 144.0, 'infant deaths': 3.0, 'Alcohol': 3.755, 'percentage expenditure': 64.91, 
        'Hepatitis B': 92.0, 'Measles ': 17.0, ' BMI ': 43.5, 'under-five deaths ': 4.0, 'Polio': 93.0, 
        'Total expenditure': 5.755, 'Diphtheria ': 93.0, ' HIV/AIDS': 0.1, 'GDP': 1766.95, 
        ' thinness  1-19 years': 3.3, ' thinness 5-9 years': 3.3, 'Income composition of resources': 0.677, 'Schooling': 12.3
    }
    input_data = default_values.copy()
    
    # Create a nice form with columns for the top inputs
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data['GDP'] = col1.number_input("GDP (in USD)", value=1766.95, format="%.2f", help="Gross Domestic Product per capita (in USD)")
        input_data[' BMI '] = col2.number_input("Average BMI", value=43.5, format="%.1f", help="Average Body Mass Index of entire population")
        
        input_data['Schooling'] = col1.number_input("Years of Schooling", value=12.3, format="%.1f", help="Number of years of Schooling (years)")
        input_data['Alcohol'] = col2.number_input("Alcohol Consumption (litres)", value=3.75, format="%.2f", help="Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)")
        
        input_data['Adult Mortality'] = col1.number_input("Adult Mortality (per 1000)", value=144.0, format="%.1f", help="Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)")
        input_data[' HIV/AIDS'] = col2.number_input("HIV/AIDS (deaths per 1000 live births)", value=0.1, format="%.2f", help="Deaths per 1000 live births HIV/AIDS (0-4 years)")
            
        submit_button = st.form_submit_button("Predict Life Expectancy")
        
    if submit_button:
        if model is not None:
            # Convert input data to dataframe and strictly enforce the column order
            input_df = pd.DataFrame([input_data])[FEATURES]
            
            try:
                # Scale the input data and Predict
                X_scaled = scaler.transform(input_df)
                prediction = model.predict(X_scaled)[0]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #6c757d; margin-bottom: 0;">Predicted Life Expectancy</h3>
                    <h1 style="color: #2b2d42; font-size: 3.5rem; margin-top: 0.5rem;">{prediction:.1f} <span style="font-size: 1.5rem">Years</span></h1>
                </div>
                """, unsafe_allow_html=True)
                
                if prediction > 100 or prediction < 30:
                    st.warning("Note: The prediction is outside the typical human life expectancy range. This could happen if the inputs are on a different scale than what the model was trained on (e.g. if the model expects standardized/scaled features).")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Upload Dataset for Batch Predictions")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Generate Predictions"):
            if model is not None:
                # Ensure all features exist in the uploaded file
                missing_cols = [col for col in FEATURES if col not in df.columns]
                
                if missing_cols:
                    st.error(f"The uploaded CSV is missing the following required columns: {missing_cols}")
                else:
                    try:
                        # Extract the required columns and fill NaNs with 0 for prediction
                        X_batch = df[FEATURES].fillna(0)
                        X_batch_scaled = scaler.transform(X_batch)
                        predictions = model.predict(X_batch_scaled)
                        
                        # Add predictions to the dataframe
                        result_df = df.copy()
                        result_df['Predicted Life Expectancy'] = predictions
                        
                        st.success("Predictions generated successfully!")
                        st.dataframe(result_df[['Predicted Life Expectancy'] + FEATURES].head(10))
                        
                        # Download button
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Full Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
