
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  
import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
data_path = os.path.join(script_dir, 'data', 'train.csv')

print("--- Starting House Price Prediction Model ---")

try:
    df = pd.read_csv(data_path)
    print(f" Dataset loaded successfully.")
except FileNotFoundError:
    print(f" Error: Could not find 'train.csv' at the expected path: {data_path}")
    exit()

df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath']
target = 'SalePrice'
df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Data split for training and testing.")

model = LinearRegression()
model.fit(X_train, y_train)
print(" Linear Regression model trained.")

print("\n--- Evaluating Model Performance ---")
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
print(f" R-squared Score: {r2:.2f}")
print(f"(This means our model explains roughly {r2:.0%} of the variance in house prices)")

print("\n--- Creating Visualizations ---")

plt.figure(figsize=(10, 6))
sns.histplot(df['GrLivArea'], kde=True, bins=50)
plt.title('Distribution of Living Area (GrLivArea)', fontsize=16)
plt.xlabel('Square Feet', fontsize=12)
plt.ylabel('Number of Houses', fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(script_dir, 'living_area_distribution.png'))
print(f" Plot 2 saved as 'living_area_distribution.png'")

residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_test, y=residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.title('Residual Plot', fontsize=16)
plt.xlabel('Predicted Prices ($)', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(script_dir, 'residual_plot.png'))
print(f" Plot 3 saved as 'residual_plot.png'")

print("\n--- All tasks complete. ---") 
