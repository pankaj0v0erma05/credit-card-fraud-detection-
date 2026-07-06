import os
import streamlit as st
import pandas as pd
import numpy as np

# Safely import prediction logic from predict.py in the same folder
try:
    from predict import predict_transaction, preprocessor, model
except ImportError:
    # Fallback to handle import issues if run from different working directories
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from predict import predict_transaction, preprocessor, model

# Define directory structures relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'static'))

import shap
import matplotlib.pyplot as plt

# Cache the explainer resource to avoid re-constructing it on each Streamlit rerun
@st.cache_resource
def get_shap_explainer(_model):
    # TreeExplainer is extremely fast and optimized for tree models like XGBoost
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

def main():
    # Set page configuration for a premium dashboard look
    st.set_page_config(
        page_title="Financial Sentinel - Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # -------------------------------------------------------------------------
    # Custom CSS Styling: High-Contrast Premium Solid Theme (SaaS Style)
    # -------------------------------------------------------------------------
    st.markdown("""
    <style>
    /* Import modern Google fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Set solid high-contrast dark background and default font family */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    
    /* Standard text colors overrides for crisp readability */
    .stMarkdown, .stMarkdown p, .stMarkdown li, div[data-testid="stMarkdownContainer"] p, label, .stSlider label {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* Form inputs and dropdown labels */
    div[data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Headings styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
    }
    
    /* Custom main title styling with gradient text */
    .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 10%, #8b5cf6 50%, #ec4899 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    /* Custom subheader styling */
    .subheader-text {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Solid High-Contrast Cards for layout items */
    .glass-card {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
    }
    
    /* Metric container solid dark card */
    div[data-testid="stMetric"] {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #4b5563 !important;
        transform: translateY(-2px) !important;
    }
    
    /* Custom label overrides for metric boxes */
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Navigation tabs styling - solid tab layout */
    div[data-baseweb="tab-list"] {
        background: #111827 !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid #1f2937 !important;
        margin-bottom: 24px !important;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 12px 24px !important;
        color: #94a3b8 !important;
        background-color: transparent !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: #1f2937 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Custom classification result banners for solid readability */
    .approved-banner {
        background-color: #064e3b !important;
        border: 2px solid #059669 !important;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);
    }
    
    .fraud-banner {
        background-color: #7f1d1d !important;
        border: 2px solid #dc2626 !important;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.35);
        animation: warningPulse 2.5s infinite;
    }
    @keyframes warningPulse {
        0% { box-shadow: 0 4px 15px rgba(220, 38, 38, 0.2); }
        50% { box-shadow: 0 4px 25px rgba(220, 38, 38, 0.45); }
        100% { box-shadow: 0 4px 15px rgba(220, 38, 38, 0.2); }
    }
    
    /* Input widgets (numbers, selectors, files) styling for solid dark theme */
    div[data-baseweb="input"], select, textarea {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    
    /* File uploader high contrast */
    div[data-testid="stFileUploader"] {
        background-color: #111827 !important;
        border: 2px dashed #4b5563 !important;
        border-radius: 14px !important;
        padding: 24px !important;
    }
    
    /* Collapsible Expander styling */
    div[data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
    }
    
    /* Accent action buttons */
    button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
    }
    
    /* Styled HR */
    hr {
        border-color: #1f2937 !important;
        margin: 24px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # UI Header: Enterprise Title
    # -------------------------------------------------------------------------
    st.markdown('<div class="title-text">🛡️ Financial Sentinel</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-text">Real-Time Credit Card Fraud Detection Engine</div>', unsafe_allow_html=True)
    
    # Create the three-tab navigation layout
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Transaction Scan", 
        "📁 Batch Processing Pipeline", 
        "📊 Exploratory Data Insights"
    ])
    
    # =========================================================================
    # TAB 1: Single Transaction Scan
    # =========================================================================
    with tab1:
        st.markdown("""
        Adjust the transaction properties below to evaluate the likelihood of fraud in real time.
        The system exposes the most critical features as direct inputs and folds other parameters away.
        """)
        
        # Split inputs into two columns: Principal Inputs vs. Details
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Key Transactional Fields")
            # Time slider: seconds elapsed over a 48-hour period
            time_val = st.slider(
                "Transaction Time (Seconds elapsed since baseline)", 
                min_value=0.0, 
                max_value=172800.0, 
                value=42000.0, 
                step=100.0,
                help="Seconds elapsed since the start of the transaction dataset recording window (up to 48 hours)."
            )
            
            # Amount input: raw dollars
            amount_val = st.number_input(
                "Transaction Amount ($)", 
                min_value=0.0, 
                max_value=25000.0, 
                value=25.0, 
                step=1.0,
                help="Raw transaction value in USD."
            )
            
            st.subheader("Critical Correlated PCA Features")
            st.markdown("*These features were identified during EDA as having the strongest negative correlation with fraud.*")
            
            # Direct inputs for the most vital fraud indicating features (V17, V14, V12)
            v17_val = st.slider("V17 (Strongest Negative Correlation)", -30.0, 10.0, 0.0, 0.1, help="PCA feature V17. Highly negative values correlate strongly with fraud.")
            v14_val = st.slider("V14 (Second Strongest Negative)", -30.0, 10.0, 0.0, 0.1, help="PCA feature V14. Highly negative values correlate strongly with fraud.")
            v12_val = st.slider("V12 (Third Strongest Negative)", -30.0, 10.0, 0.0, 0.1, help="PCA feature V12. Highly negative values correlate strongly with fraud.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Advanced Anonymized Features")
            # Collapsible expander to prevent cluttering the interface with 28 sliders
            with st.expander("Adjust Anonymized Components (V1 to V28)", expanded=False):
                st.markdown("These anonymous components are standard-scaled. Default is set to 0.0 (dataset mean).")
                
                # Dictionary to hold the dynamic values of V1-V28
                v_inputs = {}
                for i in range(1, 29):
                    # We skip V12, V14, and V17 as they are defined above as primary inputs
                    if i in [12, 14, 17]:
                        continue
                    v_inputs[f"V{i}"] = st.slider(f"V{i}", -10.0, 10.0, 0.0, 0.1)
            
            # Map primary inputs back into the dictionary structure
            v_inputs["V12"] = v12_val
            v_inputs["V14"] = v14_val
            v_inputs["V17"] = v17_val
            
            # Assemble the complete feature dictionary
            transaction_payload = {
                "Time": time_val,
                "Amount": amount_val,
                **{f"V{i}": v_inputs[f"V{i}"] for i in range(1, 29)}
            }
            
            # -----------------------------------------------------------------
            # Real-Time Scoring Execution
            # -----------------------------------------------------------------
            st.subheader("Sentinel Classification Output")
            
            if st.button("Analyze Transaction", type="primary"):
                # Call our inference engine
                with st.spinner("Scoring transaction..."):
                    result = predict_transaction(transaction_payload)
                    
                prob = result["fraud_probability"]
                is_fraud = result["is_fraud"]
                
                # Visual output blocks
                if is_fraud == 1:
                    st.markdown(f"""
                    <div class="fraud-banner">
                        <h3 style='color: #fca5a5; margin-top: 0;'>🚨 HIGH RISK TRANSACTION DETECTED</h3>
                        <p style='color: #fee2e2; font-size: 1.1rem;'>The transaction exhibits extreme anomalous characteristics. The probability of fraud is estimated at <strong>{prob * 100:.2f}%</strong>.</p>
                        <p style='margin-bottom: 0; color: #fee2e2; font-weight: bold;'>Action Recommendation: DECLINE immediately and trigger security SMS verification for the cardholder.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Fraud Risk Probability", f"{prob * 100:.2f}%", delta="SUSPICIOUS")
                else:
                    st.markdown(f"""
                    <div class="approved-banner">
                        <h3 style='color: #6ee7b7; margin-top: 0;'>🟢 TRANSACTION APPROVED</h3>
                        <p style='color: #d1fae5; font-size: 1.1rem;'>This transaction maps well within standard legitimate behaviors. Fraud probability is <strong>{prob * 100:.2f}%</strong>.</p>
                        <p style='margin-bottom: 0; color: #d1fae5; font-weight: bold;'>Status: SECURE. No action required.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Fraud Risk Probability", f"{prob * 100:.2f}%", delta="SECURE", delta_color="inverse")
                
                # -------------------------------------------------------------
                # SHAP Explainable AI (XAI) Risk Factors Plot
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 🔍 Explainable AI (XAI) Risk Factor Analysis")
                st.markdown(
                    "This chart displays the top features driving the model's decision. "
                    "Features in <span style='color: #ef4444; font-weight: bold;'>red (positive values)</span> push the score higher "
                    "towards **Fraud**, while features in <span style='color: #3b82f6; font-weight: bold;'>blue (negative values)</span> pull it lower "
                    "towards **Legitimate**.",
                    unsafe_allow_html=True
                )
                
                with st.spinner("Calculating SHAP feature contributions..."):
                    try:
                        # Convert input to DataFrame to match preprocessing schema
                        df_single = pd.DataFrame([transaction_payload])
                        X_single_processed = preprocessor.transform(df_single)
                        
                        # Specify columns in ColumnTransformer execution order
                        processed_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
                        X_single_df = pd.DataFrame(X_single_processed, columns=processed_cols)
                        
                        # Compute SHAP explanation values
                        shap_values_single = explainer(X_single_df)
                        
                        # Build custom styled Matplotlib Figure
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        fig.patch.set_facecolor('#111827')
                        ax.set_facecolor('#111827')
                        
                        # Force font colors for coordinate axes
                        ax.tick_params(colors='#e2e8f0', which='both', labelsize=9)
                        ax.xaxis.label.set_color('#e2e8f0')
                        
                        # Generate the horizontal waterfall chart
                        shap.plots.waterfall(shap_values_single[0], max_display=8, show=False)
                        
                        # Re-apply text color configuration to custom text blocks
                        for text in fig.texts:
                            text.set_color('#f1f5f9')
                        
                        # Render Matplotlib figure within Streamlit
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"Could not load SHAP explanations: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: Batch Processing Pipeline
    # =========================================================================
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("Batch Processing Pipeline")
        st.markdown("""
        Upload a CSV file containing transactions to scan all rows at once.
        The uploaded CSV must contain the schema columns: `Time`, `Amount`, and `V1` to `V28`.
        """)
        
        uploaded_file = st.file_uploader("Drag and drop your transaction CSV here", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Load the CSV
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded file containing {len(df_batch):,} rows.")
                
                # Check for column compliance
                required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
                missing_cols = [col for col in required_cols if col not in df_batch.columns]
                
                if missing_cols:
                    st.error(f"Invalid Schema! Missing columns: {missing_cols}")
                else:
                    # Run predictions using batch transformation
                    with st.spinner("Processing batch predictions..."):
                        # Extract features in correct order
                        X_batch = df_batch[required_cols]
                        
                        # Apply preprocessor scaling
                        X_batch_processed = preprocessor.transform(X_batch)
                        
                        # Apply model predictions
                        predictions = model.predict(X_batch_processed)
                        probabilities = model.predict_proba(X_batch_processed)[:, 1]
                        
                        # Append predictions back to the DataFrame
                        df_results = df_batch.copy()
                        df_results["is_fraud"] = predictions
                        df_results["fraud_probability"] = probabilities
                        
                    # Calculate summary metrics
                    total_scanned = len(df_results)
                    flagged_fraud = int(np.sum(predictions == 1))
                    fraud_ratio = (flagged_fraud / total_scanned) * 100
                    
                    st.markdown("### Batch Results Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Scanned Transactions", f"{total_scanned:,}")
                    c2.metric("Flagged Fraud Transactions", f"{flagged_fraud:,}", delta="Alerts" if flagged_fraud > 0 else "Normal")
                    c3.metric("Fraud Ratio", f"{fraud_ratio:.4f}%")
                    
                    # Display flagged items table
                    st.markdown("### Flagged Fraud Transactions (Preview)")
                    flagged_df = df_results[df_results["is_fraud"] == 1].sort_values(by="fraud_probability", ascending=False)
                    
                    if len(flagged_df) > 0:
                        st.dataframe(
                            flagged_df[["Time", "Amount", "fraud_probability"] + [f"V{i}" for i in [17, 14, 12]]].head(50)
                        )
                    else:
                        st.info("No fraudulent transactions identified in this batch.")
                        
                    # Provide download link for the full output
                    st.markdown("### Download Full Pipeline Outputs")
                    # Convert dataframe to CSV byte array
                    csv_bytes = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Prediction Results (CSV)",
                        data=csv_bytes,
                        file_name="sentinel_fraud_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: Exploratory Data Insights
    # =========================================================================
    with tab3:
        st.header("Exploratory Data Insights")
        st.markdown("""
        These static charts represent the statistical signatures identified in the raw data during exploratory 
        analysis. Understanding these distributions is critical for building model intuition.
        """)
        
        # Layout plots in columns
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            # 1. Class Imbalance Plot
            imbalance_img_path = os.path.join(STATIC_DIR, 'class_imbalance.png')
            if os.path.exists(imbalance_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(imbalance_img_path, caption="Figure 1: Target Class Imbalance distribution (Log scale)")
                st.markdown("""
                **Insight**: Only 0.17% of all recorded transactions represent fraudulent charges. Because 
                this minority class is so rare, models trained without class weight balancing (`stratify=y` or 
                `scale_pos_weight`) would automatically predict 100% legitimate classifications, completely 
                failing to capture fraud patterns.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # 2. Correlation Vector Plot
            corr_img_path = os.path.join(STATIC_DIR, 'correlation_heatmap.png')
            if os.path.exists(corr_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(corr_img_path, caption="Figure 2: Linear correlation coefficients with target Class")
                st.markdown("""
                **Insight**: While most PCA components are uncorrelated with each other, they have varied linear 
                relationships with the target. Features like `V17`, `V14`, and `V12` are strongly negatively 
                correlated with Class, acting as primary signals for fraud.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col_right:
            # 3. Amount Distribution Plot
            amount_img_path = os.path.join(STATIC_DIR, 'amount_distribution.png')
            if os.path.exists(amount_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(amount_img_path, caption="Figure 3: Transaction Amount KDE distribution density (Under $250)")
                st.markdown("""
                **Insight**: Looking at the probability density, fraudulent transaction amounts peak in different 
                valleys compared to legitimate ones, suggesting attackers often utilize specific pricing floors.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # 4. Hidden Story 1 (Diurnal time distribution)
            time_img_path = os.path.join(STATIC_DIR, 'time_diurnal_distribution.png')
            if os.path.exists(time_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(time_img_path, caption="Figure 4: Transaction density by Hour-of-Day diurnal cycle")
                st.markdown("""
                **Insight**: Legitimate transactions show a diurnal human pattern, bottoming out between 1 AM and 
                5 AM. Fraudulent transactions, however, remain active overnight, suggesting automated scripts 
                or operations spanning global timezones.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # 5. Hidden Story 2 (PCA Separability Scatter Plot)
            separability_img_path = os.path.join(STATIC_DIR, 'pca_separability_scatter.png')
            if os.path.exists(separability_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(separability_img_path, caption="Figure 5: PCA Feature Space Separability (V14 vs V17)")
                st.markdown("""
                **Insight**: Downsampling legitimate transactions reveals a tight cluster of fraud points in 
                the bottom-left region of the V14/V17 space, showing the separability of classes.
                """)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
