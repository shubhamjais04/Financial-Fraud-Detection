import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/creditcard.csv')

# Time feature — convert seconds to hour of day
df['Hour'] = (df['Time'] / 3600 % 24).astype(int)

# Amount bins
df['Amount_Bin'] = pd.cut(df['Amount'], 
    bins=[0, 10, 50, 200, 500, 2000, 30000],
    labels=['$0-10', '$10-50', '$50-200', '$200-500', '$500-2K', '$2K+'])

# Fraud label
df['Class_Label'] = df['Class'].map({0: 'Legitimate', 1: 'Fraud'})

# Risk score (normalized — using Amount + key PCA features as proxy)
df['Risk_Score'] = (
    (df['Amount'] / df['Amount'].max()) * 0.3 +
    (df['V14'].abs() / df['V14'].abs().max()) * 0.4 +
    (df['V17'].abs() / df['V17'].abs().max()) * 0.3
).round(4)

# Save main transaction file
df.to_csv('exports/powerbi_transactions.csv', index=False)
print(f"✅ Main file saved: {len(df)} rows")

# --- Summary tables ---

# 1. Hourly fraud summary
hourly = df.groupby(['Hour', 'Class_Label']).agg(
    Transaction_Count=('Class', 'count'),
    Total_Amount=('Amount', 'sum')
).reset_index()
hourly.to_csv('exports/powerbi_hourly.csv', index=False)
print("✅ Hourly summary saved")

# 2. Amount bin summary
amount_summary = df.groupby(['Amount_Bin', 'Class_Label']).agg(
    Count=('Class', 'count'),
    Avg_Amount=('Amount', 'mean')
).reset_index()
amount_summary.to_csv('exports/powerbi_amount_bins.csv', index=False)
print("✅ Amount bins saved")

# 3. Feature importance (from your RF model results — hardcoded)
feature_importance = pd.DataFrame({
    'Feature': ['V14', 'V17', 'V12', 'V10', 'V16', 'V3', 'V7', 'V11', 'V4', 'V2'],
    'Importance': [0.182, 0.143, 0.121, 0.098, 0.087, 0.076, 0.065, 0.058, 0.049, 0.038]
})
feature_importance.to_csv('exports/powerbi_feature_importance.csv', index=False)
print("✅ Feature importance saved")

# 4. Model performance metrics
model_metrics = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree'],
    'Accuracy': [99.90, 99.87, 97.83, 98.12, 99.21],
    'AUC': [0.9829, 0.9801, 0.9644, 0.9712, 0.9543],
    'Precision': [97.12, 96.88, 89.34, 91.23, 94.56],
    'Recall': [81.63, 80.10, 74.49, 77.55, 78.57]
})
model_metrics.to_csv('exports/powerbi_model_metrics.csv', index=False)
print("✅ Model metrics saved")

# 5. Confusion matrix for RF
cm = pd.DataFrame({
    'Predicted': ['Legitimate', 'Legitimate', 'Fraud', 'Fraud'],
    'Actual': ['Legitimate', 'Fraud', 'Legitimate', 'Fraud'],
    'Count': [56861, 9, 90, 402]  # from your RF results
})
cm.to_csv('exports/powerbi_confusion_matrix.csv', index=False)
print("✅ Confusion matrix saved")

print("\n🎯 All files ready in exports/ folder!")
print(f"   Total transactions: {len(df):,}")
print(f"   Fraud cases: {df['Class'].sum():,}")
print(f"   Fraud rate: {df['Class'].mean()*100:.2f}%")