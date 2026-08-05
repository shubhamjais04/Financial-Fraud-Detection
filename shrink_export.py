import pandas as pd

df = pd.read_csv('exports/fraud_predictions.csv')
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(n=20000, random_state=42)
df_small = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
df_small.to_csv('exports/fraud_predictions.csv', index=False)
print(df_small.shape)