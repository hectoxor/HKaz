import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Create data for product categories
categories = {
    "Category": ["Fitness Services", "Health-Related Services", "Group Activities"],
    "Market Size (USD M)": [500, 25790, 150],  # Health services in billions, converted to M for scale
    "Growth Rate (CAGR %)": [6.75, 5.34, 4.5],
    "Scale": ["$500M annually", "$25.79B by 2025", "Growing market"],
    "Key Trend": ["Digital fitness solutions", "Aging population needs", "Community engagement"],
    "HK Specifics": ["Tech-savvy audience", "High purchasing power", "Diverse demographics"]
}

# Create DataFrame
df = pd.DataFrame(categories)

# Scale market size for visualization (health services is much larger)
df["Normalized Size"] = [500, 2500, 150]  # Scale down health services for better visualization

# Create a figure with grid layout
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])

# --- Bar and line chart for market size and growth ---
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(df['Category']))
width = 0.5

# Custom colormap for bars
colors = LinearSegmentedColormap.from_list("categories", ["#4575b4", "#91bfdb", "#e0f3f8"])
bar_colors = [colors(i/len(df)) for i in range(len(df))]

# Market size bars
bars = ax1.bar(x, df['Normalized Size'], width, label='Market Size (USD)', color=bar_colors)

# Create second y-axis for growth rate
ax2 = ax1.twinx()
line = ax2.plot(x, df['Growth Rate (CAGR %)'], 'o-', linewidth=2.5, 
                color='#d73027', markersize=10, label='Growth Rate (CAGR %)')

# Labels and formatting
ax1.set_xlabel('Product Category', fontsize=14, fontweight='bold')
ax1.set_ylabel('Market Size (Normalized for Visualization)', fontsize=12)
ax2.set_ylabel('Growth Rate (CAGR %)', fontsize=12, color='#d73027')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Category'], fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax2.tick_params(axis='y', labelcolor='#d73027')

# Add market size annotations to bars
for i, bar in enumerate(bars):
    original_value = df['Market Size (USD M)'][i]
    formatted_value = f"${original_value/1000:.2f}B" if original_value > 1000 else f"${original_value}M"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            formatted_value, ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add growth rate annotations
for i, v in enumerate(df['Growth Rate (CAGR %)']):
    ax2.text(i, v + 0.2, f"{v}%", ha='center', va='bottom', fontsize=10, 
            fontweight='bold', color='#d73027')

# Title
ax1.set_title('Hong Kong Market Opportunities: Product Category Analysis', fontsize=16, fontweight='bold')

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# --- Table with detailed information ---
ax_table = fig.add_subplot(gs[0, 1])
ax_table.axis('off')

# Create detailed data for table
table_data = [
    [df['Category'][i], df['Scale'][i], df['Key Trend'][i], df['HK Specifics'][i]] 
    for i in range(len(df))
]

ax_table = fig.add_subplot(gs[0, 1])
ax_table.axis('off')

# Create detailed data for table
table_data = [
    [df['Category'][i], df['Scale'][i], df['Key Trend'][i], df['HK Specifics'][i]] 
    for i in range(len(df))
]

# Add column headers to data
table_data.insert(0, ['Category', 'Market Scale', 'Key Trend', 'HK Specifics'])

# Create table with auto-wrapping text
table = ax_table.table(
    cellText=table_data,
    loc='center',
    cellLoc='center',
    colWidths=[0.25, 0.25, 0.25, 0.25]
)

# Style the table - reduced font size and increased scale
table.auto_set_font_size(False)
table.set_fontsize(8)  # Reduced from 10 to 8
table.scale(1.3, 1.8)  # Increased from 1.2, 1.5 to give more room

# Add text wrapping function
def wrap_text(cell):
    text = cell.get_text().get_text()
    wrapped_text = '\n'.join(textwrap.wrap(text, width=12))  # Adjust width as needed
    cell.get_text().set_text(wrapped_text)

# Apply text wrapping to all cells
import textwrap
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        wrap_text(cell)
        # Make cell heights taller to accommodate wrapped text
        cell.set_height(0.15)

# Style header row
for j in range(len(table_data[0])):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='w', fontweight='bold')
    
# Alternate row colors for readability
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#D9E1F2')
        else:
            table[(i, j)].set_facecolor('#E9EDF4')

# --- Opportunity gaps analysis ---
ax_gaps = fig.add_subplot(gs[1, :])
gap_categories = ['Booking System', 'Mobile Integration', 'Premium Offerings', 'Personalization']
gap_values = [85, 70, 60, 90]
gap_colors = ['#FFC000', '#70AD47', '#4472C4', '#ED7D31']

bars = ax_gaps.bar(gap_categories, gap_values, color=gap_colors, width=0.6)
ax_gaps.set_ylim(0, 100)
ax_gaps.set_ylabel('Opportunity Score (0-100)', fontsize=12)
ax_gaps.set_title('Market Gap Analysis: Opportunity Score by Category', fontsize=14, fontweight='bold')
ax_gaps.grid(axis='y', linestyle='--', alpha=0.3)

# Add value annotations
for bar in bars:
    height = bar.get_height()
    ax_gaps.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add explanation text boxes
gap_explanations = [
    'Gap: Complex booking processes\nOpportunity: Simplified all-in-one system',
    'Gap: Limited mobile functionality\nOpportunity: Advanced mobile features',
    'Gap: Standard offerings\nOpportunity: Premium experiences',
    'Gap: Generic services\nOpportunity: AI-driven personalization'
]

for i, (bar, explanation) in enumerate(zip(bars, gap_explanations)):
    ax_gaps.text(bar.get_x() + bar.get_width()/2., 30,
                explanation, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                fontsize=8)

plt.tight_layout()
plt.savefig('product_category_analysis.png', dpi=300, bbox_inches='tight')
plt.show()