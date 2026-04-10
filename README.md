# Students-performance-by-study-hours
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Score': [35, 40, 50, 55, 65, 70, 75, 85]
}

df = pd.DataFrame(data)

# Split data
X = df[['Hours']]
y = df['Score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
new_hours_data = {'Hours': [9, 10]}
new_hours_df = pd.DataFrame(new_hours_data)
predicted_scores = model.predict(new_hours_df)

print("Predicted Scores for 9 and 10 hours:", predicted_scores)

# --- Visualizations ---

# 1. Scatter Plot (Original)
plt.figure(figsize=(10, 6))
plt.scatter(df['Hours'], df['Score'], color='blue', label='Actual Data')
plt.plot(df['Hours'], model.predict(X), color='red', label='Prediction Line')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Score Prediction (Scatter Plot)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart of Actual Scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Hours', y='Score', data=df, palette='viridis', hue='Hours', legend=False)
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Actual Scores by Study Hours (Bar Chart)")
plt.grid(axis='y', linestyle='--')
plt.show()

# 3. Heatmap of Correlation Matrix
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap (Hours vs. Score)")
plt.show()
