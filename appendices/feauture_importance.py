import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create synthetic feature importance data
features = [
    'Exercise Frequency', 'Distance to Facility', 'Age', 'BMI',
    'Muscular Strength', 'Income Level', 'Gender', 'District',
    'Exercise Type', 'Previous Attendance', 'Age × BMI Interaction',
    'Time Available per Week', 'Exercise Intensity Preference',
    'Transportation Method', 'Weather Sensitivity'
]

# Generate importance values with some variance
np.random.seed(42)
importance = np.sort(np.random.random(len(features)))[::-1]
importance = importance / importance.sum()  # Normalize

# Create a dataframe
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sort by importance
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 10))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')

# Add percentage labels
for i, v in enumerate(feature_importance['Importance']):
    ax.text(v + 0.01, i, f"{v:.1%}", va='center')

# Customize
plt.title('Feature Importance for Fitness Adherence Prediction', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()

# Add vertical lines for importance thresholds
plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7)

# Add annotations for feature groups
plt.annotate('Primary\nPredictors', xy=(0.15, 1), xytext=(0.18, 2), 
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
plt.annotate('Secondary\nPredictors', xy=(0.07, 5), xytext=(0.1, 6), 
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
plt.annotate('Supporting\nVariables', xy=(0.03, 12), xytext=(0.06, 13), 
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()