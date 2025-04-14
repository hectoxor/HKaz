import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set up the figure with a grid layout
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])

# Create custom color palette
colors = {
    "primary": "#4472C4",
    "secondary": "#ED7D31",
    "highlight": "#FF5050",
    "success": "#70AD47",
    "neutral": "#9B9B9B"
}

# ----- 1. PAIN POINT SEVERITY HEATMAP -----
ax1 = fig.add_subplot(gs[0, 0])

# Define categories and specific pain points
categories = ["Booking System", "Customer Support", "Pricing & Membership", "Service Quality"]
pain_points = {
    "Booking System": ["Complex Process", "Lack of Integration"],
    "Customer Support": ["Slow Response Times", "Limited Availability"],
    "Pricing & Membership": ["High Costs", "Rigid Plans"],
    "Service Quality": ["Inconsistent Quality", "Limited Class Availability"]
}

# Flatten pain points for heatmap
all_pain_points = []
category_indices = []
for cat, points in pain_points.items():
    all_pain_points.extend(points)
    category_indices.extend([cat] * len(points))

# Create severity data (0-10 scale)
severity_data = [
    8.7, 7.9,  # Booking System
    7.2, 6.8,  # Customer Support
    8.3, 7.6,  # Pricing & Membership
    8.1, 7.5   # Service Quality
]

# Create impact data (user satisfaction impact, 0-10 scale)
impact_data = [
    9.0, 7.5,  # Booking System
    6.8, 6.2,  # Customer Support
    8.5, 7.8,  # Pricing & Membership
    8.7, 7.9   # Service Quality
]

# Create DataFrame for the heatmap
df_heatmap = pd.DataFrame({
    'Category': category_indices,
    'Pain Point': all_pain_points,
    'Severity': severity_data,
    'Impact': impact_data
})

# Reshape data for heatmap
heatmap_pivot = df_heatmap.pivot_table(
    index='Pain Point', 
    columns='Category', 
    values='Severity',
    aggfunc='first'
)

# Sort pain points by severity
sorted_points = df_heatmap.sort_values('Severity', ascending=False)['Pain Point'].tolist()
heatmap_pivot = heatmap_pivot.reindex(sorted_points)

# Create heatmap
sns.heatmap(heatmap_pivot, cmap='Reds', annot=True, fmt='.1f', linewidths=.5, ax=ax1,
           cbar_kws={'label': 'Severity Score (0-10)'})
ax1.set_title('Consumer Pain Points: Severity Analysis', fontsize=16, fontweight='bold')
ax1.set_ylabel('')
ax1.set_xlabel('')

# ----- 2. CUSTOMER JOURNEY PAIN POINTS -----
ax2 = fig.add_subplot(gs[0, 1])

# Define journey stages and corresponding pain point locations
stages = ['Discovery', 'Sign Up', 'Initial Use', 'Regular Usage', 'Support', 'Renewal']
pain_indices = [0.5, 1.2, 2.3, 3.1, 4.2, 5.1]  # Position of pain points on the journey

# Create journey visualization
y_pos = 0.5
ax2.plot([0, 5], [y_pos, y_pos], '-', color='#808080', linewidth=3, alpha=0.7)

# Add stage points
for i, stage in enumerate(stages):
    if i < len(stages) - 1:
        ax2.scatter(i, y_pos, s=100, color=colors['primary'], zorder=5)
    else:
        # Make the last stage special
        ax2.scatter(i, y_pos, s=150, color=colors['success'], marker='*', zorder=5)
    
    # Add stage labels with rotated text
    ax2.text(i, y_pos - 0.15, stage, ha='center', va='top', fontsize=10, 
            rotation=45, rotation_mode='anchor')

# Define pain points and their positions
journey_pain_points = [
    "Rigid Plan Options",
    "Complex Booking System", 
    "Limited Class Availability",
    "Inconsistent Quality",
    "Slow Support Response",
    "Renewal Flexibility"
]

# Add pain point markers and descriptions
for i, (pos, pain) in enumerate(zip(pain_indices, journey_pain_points)):
    # Alternate between top and bottom of the line
    if i % 2 == 0:
        y_offset = 0.3
        va = 'bottom'
        connection = [(pos, y_pos), (pos, y_pos + 0.1), (pos, y_pos + y_offset)]
    else:
        y_offset = -0.3
        va = 'top'
        connection = [(pos, y_pos), (pos, y_pos - 0.1), (pos, y_pos + y_offset)]
    
    # Draw connecting line
    ax2.plot(*zip(*connection), 'r-', linewidth=1.5)
    
    # Add pain point marker
    ax2.scatter(pos, y_pos + y_offset, s=200, color=colors['highlight'], marker='X', zorder=5, 
               edgecolors='black', linewidth=1)
    
    # Add text description in a box
    ax2.text(pos, y_pos + y_offset + (0.1 if i % 2 == 0 else -0.1), 
            pain, ha='center', va=va, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, 
                    edgecolor=colors['highlight']))

# Configure the axis
ax2.set_xlim(-0.5, len(stages) - 0.5)
ax2.set_ylim(-0.5, 1.5)
ax2.axis('off')
ax2.set_title('Customer Journey Pain Points', fontsize=16, fontweight='bold')

# ----- 3. PAIN POINT RESOLUTION PRIORITY MATRIX -----
ax3 = fig.add_subplot(gs[1, 0])

# Define pain points for the matrix
matrix_pain_points = [
    "Complex Booking", "Integration Issues", "Slow Support",
    "Limited Availability", "High Cost", "Rigid Plans", 
    "Inconsistent Quality", "Limited Classes"
]

# Impact and effort scores (1-10 scale)
impact_scores = [9.0, 7.5, 6.8, 7.9, 8.5, 7.8, 8.7, 7.9]
effort_scores = [7.0, 6.5, 3.0, 5.0, 8.0, 4.0, 8.5, 5.5]

# Create scatter plot with labeled points
scatter = ax3.scatter(
    effort_scores, 
    impact_scores, 
    s=550, 
    c=impact_scores,
    cmap='viridis', 
    alpha=0.7, 
    edgecolors='black', 
    linewidth=1
)

# Add labels for each point
for i, txt in enumerate(matrix_pain_points):
    ax3.annotate(
        txt, 
        (effort_scores[i], impact_scores[i]),
        xytext=(0, 0), 
        textcoords='offset points',
        fontsize=9, 
        fontweight='bold',
        ha='center', 
        va='center',
        color='white' if impact_scores[i] > 7.5 else 'black'
    )

# Add quadrant lines
ax3.axhline(y=7.5, color='gray', linestyle='--', alpha=0.7)
ax3.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7)

# Add quadrant labels
ax3.text(2.75, 8.75, "Quick Wins", fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#90EE90', alpha=0.3))
ax3.text(8, 8.75, "Major Projects", fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#ADD8E6', alpha=0.3))
ax3.text(2.75, 6.25, "Fill-in Tasks", fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFE0', alpha=0.3))
ax3.text(8, 6.25, "Thankless Tasks", fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFB6C1', alpha=0.3))

# Set axis labels and title
ax3.set_xlabel('Implementation Effort', fontsize=12)
ax3.set_ylabel('Customer Impact', fontsize=12)
ax3.set_title('Pain Point Resolution Priority Matrix', fontsize=16, fontweight='bold')

# Set axis limits and grid
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.grid(True, linestyle='--', alpha=0.6)

# ----- 4. PAIN POINT SATISFACTION METRICS -----
ax4 = fig.add_subplot(gs[1, 1])

# Define categories
categories = ['Booking System', 'Customer Support', 'Pricing', 'Service Quality']

# Define different metrics
current_satisfaction = [3.2, 4.1, 3.5, 3.8]  # Current satisfaction (1-10)
industry_benchmark = [7.5, 7.0, 6.5, 8.0]    # Industry benchmark
satisfaction_gap = [b - c for c, b in zip(current_satisfaction, industry_benchmark)]

# Create DataFrame for plotting
df_satisfaction = pd.DataFrame({
    'Category': categories,
    'Current Satisfaction': current_satisfaction,
    'Industry Benchmark': industry_benchmark,
    'Gap': satisfaction_gap
})

# Set position of bars
x = np.arange(len(categories))
width = 0.35

# Create bars
bars1 = ax4.bar(x - width/2, current_satisfaction, width, label='Current Satisfaction', color=colors['primary'])
bars2 = ax4.bar(x + width/2, industry_benchmark, width, label='Industry Benchmark', color=colors['secondary'])

# Add gap arrows
for i, (curr, bench) in enumerate(zip(current_satisfaction, industry_benchmark)):
    ax4.annotate('', 
                xy=(i - width/2, bench), 
                xytext=(i - width/2, curr),
                arrowprops=dict(arrowstyle='<->', color=colors['highlight'], lw=2))
    
    # Add gap text
    gap = bench - curr
    ax4.text(i, curr + gap/2, f"{gap:.1f}", ha='center', va='center', 
             color=colors['highlight'], fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

# Set chart properties
ax4.set_ylabel('Satisfaction Score (1-10)', fontsize=12)
ax4.set_title('Satisfaction Gaps vs. Industry Benchmark', fontsize=16, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(axis='y', linestyle='--', alpha=0.6)

# Set y-axis to start from 0
ax4.set_ylim(0, 10)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

# ----- OVERALL FIGURE LAYOUT -----
# Add a main title for the entire figure
fig.suptitle('Consumer Pain Points & Satisfaction Analysis', fontsize=22, fontweight='bold', y=0.98)

# Add explanatory subtitle
plt.figtext(0.5, 0.94, 
           'Addressing key pain points to enhance the fitness service experience', 
           ha='center', fontsize=14, style='italic')

# Add source note
plt.figtext(0.98, 0.01, 
           'Source: Consumer Surveys & Market Analysis', 
           ha='right', fontsize=8, style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.92])
plt.savefig('consumer_pain_points_analysis.png', dpi=300, bbox_inches='tight')
plt.show()