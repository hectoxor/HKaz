import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data based on provided facts (using placeholder for Group Activities)
# Note: Market sizes are in different units (USD millions vs billions), scaled for visualization.
# Digital Fitness: $235.34M by 2025 (6.75% CAGR)
# Health Services: $25.79B by 2025 (5.34% CAGR) -> $25790M
# Fitness Industry Revenue: >$500M annually

data = {
    'Category': ['Fitness Services', 'Health-Related Services', 'Group Activities'],
    'Projected Market Size (USD M, Scaled)': [500, 2579, 150], # Scaled Health Services (25790/10), Placeholder for Group
    'Digital Growth Potential (CAGR %)': [6.75, 5.34, 5.0] # Placeholder for Group
}
df = pd.DataFrame(data)

# --- Visualization ---
x = np.arange(len(df['Category']))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 7))

# Bar chart for Market Size (Left Y-axis)
color = 'tab:blue'
rects1 = ax1.bar(x - width/2, df['Projected Market Size (USD M, Scaled)'], width, label='Market Size (USD M, Scaled)', color=color, alpha=0.8)
ax1.set_xlabel('Product Category')
ax1.set_ylabel('Market Size (USD M, Scaled)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Category'])
ax1.legend(loc='upper left')

# Line chart for Growth Rate (Right Y-axis)
ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
color = 'tab:red'
line1 = ax2.plot(x + width/2, df['Digital Growth Potential (CAGR %)'], label='Digital Growth (CAGR %)', color=color, marker='o', linestyle='--')
ax2.set_ylabel('Digital Growth (CAGR %)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.suptitle('Hong Kong Market Opportunities: Category Deep Dive')
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Add labels to bars
for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height:.0f}M',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3), # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.show()
# To save the plot:
# plt.savefig('category_deep_dive_chart.png', dpi=300)