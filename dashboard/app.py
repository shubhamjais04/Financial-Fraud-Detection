import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Financial Fraud Detection",
                   page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('../exports/fraud_predictions.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('../models/best_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    return model, scaler

df = load_data()
model, scaler = load_model()

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "📈 EDA & Patterns",
    "🤖 Model Performance",
    "🔎 Live Prediction",
    "📋 Transaction Table"
])
threshold = st.sidebar.slider("Fraud Decision Threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Internship Project**")
st.sidebar.markdown("Amdox Technologies")

# ── PAGE 1: OVERVIEW ──────────────────────────────────────────
if page == "📊 Overview":
    st.title("🔍 Financial Fraud Detection Dashboard")
    st.markdown("**End-to-end ML pipeline to detect fraudulent transactions — Amdox Technologies**")
    st.divider()

    total = len(df)
    fraud = df['Class'].sum()
    legit = total - fraud
    fraud_rate = fraud / total * 100
    avg_fraud_amt = df[df['Class']==1]['Amount_log'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraudulent", f"{fraud:,}", f"{fraud_rate:.4f}%", delta_color='inverse')
    c3.metric("Legitimate", f"{legit:,}")
    c4.metric("Avg Fraud Log-Amount", f"{avg_fraud_amt:.2f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df, names=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                     title='Transaction Distribution',
                     color_discrete_map={'Legitimate':'#2196F3','Fraud':'#F44336'},
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk = df['Risk_Level'].value_counts().reset_index()
        risk.columns = ['Risk Level', 'Count']
        fig2 = px.bar(risk, x='Risk Level', y='Count',
                      title='Transactions by Risk Level',
                      color='Risk Level',
                      color_discrete_map={'Low':'#4CAF50',
                                         'Medium':'#FF9800',
                                         'High':'#F44336'})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(df, x='Amount_log',
                            color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                            title='Log-Amount Distribution by Class',
                            barmode='overlay', opacity=0.7, nbins=60,
                            color_discrete_map={'Legitimate':'#2196F3',
                                               'Fraud':'#F44336'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        hour_fraud = df.groupby('Hour')['Class'].mean()*100
        hour_fraud = hour_fraud.reset_index()
        hour_fraud.columns = ['Hour', 'Fraud Rate %']
        fig4 = px.bar(hour_fraud, x='Hour', y='Fraud Rate %',
                      title='Fraud Rate by Hour of Day',
                      color='Fraud Rate %',
                      color_continuous_scale='Reds')
        st.plotly_chart(fig4, use_container_width=True)

# ── PAGE 2: EDA & PATTERNS ────────────────────────────────────
elif page == "📈 EDA & Patterns":
    st.title("📈 EDA & Fraud Patterns")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                     y='Amount_log',
                     title='Log-Amount Spread by Fraud Label',
                     color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                     color_discrete_map={'Legitimate':'#2196F3','Fraud':'#F44336'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='Hour',
                            color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                            title='Transaction Hour Distribution',
                            barmode='overlay', opacity=0.7,
                            color_discrete_map={'Legitimate':'#2196F3',
                                               'Fraud':'#F44336'})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top PCA Feature Distributions")
    top_v_cols = ['V14', 'V12', 'V10', 'V4', 'V11']
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(df, x=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      y='V14', title='V14 by Fraud Label',
                      color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      color_discrete_map={'Legitimate':'#2196F3','Fraud':'#F44336'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(df, x=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      y='V12', title='V12 by Fraud Label',
                      color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      color_discrete_map={'Legitimate':'#2196F3','Fraud':'#F44336'})
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig5 = px.box(df, x=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      y='V10', title='V10 by Fraud Label',
                      color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      color_discrete_map={'Legitimate':'#2196F3','Fraud':'#F44336'})
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.box(df, x=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      y='V4', title='V4 by Fraud Label',
                      color=df['Class'].map({0:'Legitimate',1:'Fraud'}),
                      color_discrete_map={'Legitimate':'#2196F3','Fraud':'#F44336'})
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("High Amount vs Fraud")
    high_amt = df.groupby('High_Amount')['Class'].mean()*100
    high_amt = high_amt.reset_index()
    high_amt.columns = ['High Amount', 'Fraud Rate %']
    high_amt['High Amount'] = high_amt['High Amount'].map({0:'Normal Amount',
                                                            1:'High Amount'})
    fig7 = px.bar(high_amt, x='High Amount', y='Fraud Rate %',
                  title='Fraud Rate: Normal vs High Amount Transactions',
                  color='Fraud Rate %', color_continuous_scale='Reds')
    st.plotly_chart(fig7, use_container_width=True)

# ── PAGE 3: MODEL PERFORMANCE ─────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.divider()

    results = {
        'Logistic Regression': {'Accuracy': 0.9860, 'ROC-AUC': 0.9720},
        'Decision Tree':       {'Accuracy': 0.9933, 'ROC-AUC': 0.8921},
        'Random Forest':       {'Accuracy': 0.9990, 'ROC-AUC': 0.9829},
        'XGBoost':             {'Accuracy': 0.9891, 'ROC-AUC': 0.9825},
        'Isolation Forest':    {'Accuracy': 0.9983, 'ROC-AUC': 0.9079},
    }

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('ROC-AUC', ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", "Random Forest")
    c2.metric("Best Accuracy", "99.90%")
    c3.metric("Best ROC-AUC", "0.9829")

    st.divider()
    st.subheader("Model Comparison Table")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#C8E6C9'),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(results_df.reset_index(), x='index', y='Accuracy',
                     title='Accuracy Comparison',
                     color='Accuracy', color_continuous_scale='Blues',
                     labels={'index':'Model'})
        fig.update_layout(yaxis_range=[0.9, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(results_df.reset_index(), x='index', y='ROC-AUC',
                      title='ROC-AUC Comparison',
                      color='ROC-AUC', color_continuous_scale='Greens',
                      labels={'index':'Model'})
        fig2.update_layout(yaxis_range=[0.85, 1.0])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Saved Charts from Notebook")
    col3, col4 = st.columns(2)
    with col3:
        st.image('../exports/notebook_charts/img_roc_comparison.png', caption='ROC Curve Comparison')
    with col4:
        st.image('../exports/notebook_charts/img_feature_importance.png', caption='Feature Importance')
    st.image('../exports/notebook_charts/img_model_comparison.png', caption='Model Accuracy & AUC Comparison')

# ── PAGE 4: LIVE PREDICTION ───────────────────────────────────
elif page == "🔎 Live Prediction":
    st.title("🔎 Live Fraud Prediction")
    st.markdown("Adjust the PCA feature values and amount to get an instant fraud risk score.")
    st.divider()

    st.info("💡 V1-V28 are PCA-transformed features from the original transaction data. Adjust sliders based on transaction profile.")

    col1, col2 = st.columns(2)

    # Get min/max from dataset for realistic slider ranges
    with col1:
        amount = st.number_input("Transaction Amount ($)", 
                                  0.0, 30000.0, 100.0, 10.0)
        hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        v1 = st.slider("V1", float(df['V1'].min()), float(df['V1'].max()), 0.0)
        v2 = st.slider("V2", float(df['V2'].min()), float(df['V2'].max()), 0.0)
        v3 = st.slider("V3", float(df['V3'].min()), float(df['V3'].max()), 0.0)
        v4 = st.slider("V4", float(df['V4'].min()), float(df['V4'].max()), 0.0)
        v5 = st.slider("V5", float(df['V5'].min()), float(df['V5'].max()), 0.0)
        v6 = st.slider("V6", float(df['V6'].min()), float(df['V6'].max()), 0.0)
        v7 = st.slider("V7", float(df['V7'].min()), float(df['V7'].max()), 0.0)
        v8 = st.slider("V8", float(df['V8'].min()), float(df['V8'].max()), 0.0)
        v9 = st.slider("V9", float(df['V9'].min()), float(df['V9'].max()), 0.0)
        v10 = st.slider("V10", float(df['V10'].min()), float(df['V10'].max()), 0.0)
        v11 = st.slider("V11", float(df['V11'].min()), float(df['V11'].max()), 0.0)
        v12 = st.slider("V12", float(df['V12'].min()), float(df['V12'].max()), 0.0)
        v13 = st.slider("V13", float(df['V13'].min()), float(df['V13'].max()), 0.0)
        v14 = st.slider("V14", float(df['V14'].min()), float(df['V14'].max()), 0.0)

    with col2:
        v15 = st.slider("V15", float(df['V15'].min()), float(df['V15'].max()), 0.0)
        v16 = st.slider("V16", float(df['V16'].min()), float(df['V16'].max()), 0.0)
        v17 = st.slider("V17", float(df['V17'].min()), float(df['V17'].max()), 0.0)
        v18 = st.slider("V18", float(df['V18'].min()), float(df['V18'].max()), 0.0)
        v19 = st.slider("V19", float(df['V19'].min()), float(df['V19'].max()), 0.0)
        v20 = st.slider("V20", float(df['V20'].min()), float(df['V20'].max()), 0.0)
        v21 = st.slider("V21", float(df['V21'].min()), float(df['V21'].max()), 0.0)
        v22 = st.slider("V22", float(df['V22'].min()), float(df['V22'].max()), 0.0)
        v23 = st.slider("V23", float(df['V23'].min()), float(df['V23'].max()), 0.0)
        v24 = st.slider("V24", float(df['V24'].min()), float(df['V24'].max()), 0.0)
        v25 = st.slider("V25", float(df['V25'].min()), float(df['V25'].max()), 0.0)
        v26 = st.slider("V26", float(df['V26'].min()), float(df['V26'].max()), 0.0)
        v27 = st.slider("V27", float(df['V27'].min()), float(df['V27'].max()), 0.0)
        v28 = st.slider("V28", float(df['V28'].min()), float(df['V28'].max()), 0.0)

    if st.button("🔍 Analyze Transaction", type='primary'):
        amount_log = np.log1p(amount)
        high_amount = 1 if amount > df['Amount_log'].quantile(0.90) else 0

        input_data = pd.DataFrame([{
            'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4,
            'V5': v5, 'V6': v6, 'V7': v7, 'V8': v8,
            'V9': v9, 'V10': v10, 'V11': v11, 'V12': v12,
            'V13': v13, 'V14': v14, 'V15': v15, 'V16': v16,
            'V17': v17, 'V18': v18, 'V19': v19, 'V20': v20,
            'V21': v21, 'V22': v22, 'V23': v23, 'V24': v24,
            'V25': v25, 'V26': v26, 'V27': v27, 'V28': v28,
            'Amount_log': amount_log,
            'Hour': hour,
            'High_Amount': high_amount
        }])

        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        is_fraud = prob >= threshold
        risk = "🔴 HIGH RISK" if prob > 0.7 else (
               "🟡 MEDIUM RISK" if prob > 0.4 else "🟢 LOW RISK")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Level", risk)
        c3.metric("Decision", "⛔ FRAUD" if is_fraud else "✅ LEGITIMATE")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if is_fraud else "green"},
                'steps': [
                    {'range': [0, 40], 'color': '#C8E6C9'},
                    {'range': [40, 70], 'color': '#FFF9C4'},
                    {'range': [70, 100], 'color': '#FFCDD2'}
                ],
                'threshold': {'line': {'color': 'black', 'width': 4},
                              'thickness': 0.75, 'value': threshold*100}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if is_fraud:
            st.error("⚠️ Transaction flagged. Recommended: Block and notify cardholder.")
        else:
            st.success("✅ Transaction appears legitimate. Safe to process.")

# ── PAGE 5: TRANSACTION TABLE ─────────────────────────────────
elif page == "📋 Transaction Table":
    st.title("📋 Transaction Records")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_type = st.selectbox("Filter by", 
                                    ["All", "Fraud Only", "Legitimate Only"])
    with col2:
        risk_filter = st.selectbox("Risk Level", 
                                    ["All", "Low", "Medium", "High"])
    with col3:
        min_amt, max_amt = st.slider("Log-Amount Range",
                                      float(df['Amount_log'].min()),
                                      float(df['Amount_log'].max()),
                                      (float(df['Amount_log'].min()),
                                       float(df['Amount_log'].max())))

    filtered = df.copy()
    if filter_type == "Fraud Only":
        filtered = filtered[filtered['Class']==1]
    elif filter_type == "Legitimate Only":
        filtered = filtered[filtered['Class']==0]
    if risk_filter != "All":
        filtered = filtered[filtered['Risk_Level']==risk_filter]
    filtered = filtered[(filtered['Amount_log'] >= min_amt) &
                        (filtered['Amount_log'] <= max_amt)]

    display_cols = ['V1', 'V2', 'V3', 'V4', 'V14', 'Amount_log',
                    'Hour', 'High_Amount', 'Class',
                    'Fraud_Probability', 'Risk_Level']

    st.dataframe(filtered[display_cols].head(500),
                 use_container_width=True)
    st.caption(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} records")