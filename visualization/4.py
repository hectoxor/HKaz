import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define the data for the table with Unicode stars instead of emoji
data = {
    'Platform': ['1Fit', 'ClassPass', 'Pure Fitness', 'Paz Fitness', 'Fitness First'],
    'Pricing': ['★★★★', '★★', '★', '★★★', '★★'],
    'Location Coverage': ['★★', '★★★', '★★★★', '★★', '★★★★★'],
    'Digital Features': ['★★★★★', '★★★★', '★★★', '★★', '★★★'],
    'Variety of Services': ['★★★★', '★★★★★', '★★★', '★★★', '★★★★'],
    'Community Engagement': ['★★★', '★★', '★★★★', '★★★★★', '★★★'],
}

# Create a numeric representation for the star ratings
numeric_data = data.copy()
for col in data.keys():
    if col != 'Platform':
        numeric_data[col] = [len(x) for x in data[col]]

df = pd.DataFrame(data)
df_numeric = pd.DataFrame(numeric_data)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Create color map for highlighting cells
cmap = LinearSegmentedColormap.from_list('rating_colors', ['#f7f7f7', '#4472C4'])

# Create the table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# Highlight the cells based on numeric values
for i in range(len(df)):
    for j in range(1, len(df.columns)):
        # Get numeric value for color intensity
        val = df_numeric.iloc[i, j]
        max_val = df_numeric.iloc[:, j].max()
        # Color intensity based on relative rating
        color_intensity = val / max_val
        
        # Highlight 1Fit's competitive advantages
        if df.iloc[i, 0] == '1Fit' and val >= 4:
            cell_color = '#70AD47'  # Green for 1Fit strengths
            text_color = 'white'
        elif val == max_val:
            cell_color = cmap(0.8)  # Highlight category leaders
            text_color = 'white'
        else:
            cell_color = cmap(color_intensity * 0.6)
            text_color = 'black' if color_intensity < 0.7 else 'white'
            
        table[(i+1, j)].set_facecolor(cell_color)
        table[(i+1, j)].set_text_props(color=text_color)

# Style header row
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
    
# Style platform column
for i in range(len(df)):
    table[(i+1, 0)].set_facecolor('#D9E1F2')
    table[(i+1, 0)].set_text_props(fontweight='bold')

# Add title and notes
plt.figtext(0.5, 0.95, 'Hong Kong Fitness Platform Competitive Analysis', fontsize=20, ha='center', fontweight='bold')
plt.figtext(0.5, 0.02, 'Rating scale: ★ (Basic) to ★★★★★ (Excellent)', 
           fontsize=10, ha='center', style='italic')
plt.figtext(0.1, 0.02, 'Source: Market Analysis & Competitive Benchmarking 2025', 
           fontsize=8, ha='left', style='italic')

# Add a legend for highlighted cells
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#70AD47', markersize=15, label='1Fit Advantage'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(0.8), markersize=15, label='Category Leader')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 0.9])
plt.savefig('fitness_platform_comparison.png', dpi=300, bbox_inches='tight')
plt.show()