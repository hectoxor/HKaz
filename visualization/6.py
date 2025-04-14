import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, Arrow, Polygon, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Set the plotting style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Create a large figure with subplots
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1], wspace=0.25, hspace=0.35)

# Define common colors
colors = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "tertiary": "#3498DB",
    "highlight": "#F1C40F",
    "success": "#27AE60",
    "light": "#ECF0F1",
    "dark": "#2C3E50",
    "gray": "#95A5A6"
}

# ----- 1. MOBILITY CHALLENGE VISUALIZATION -----
ax1 = fig.add_subplot(gs[0, 0])

# Create data for visualization
distances = ["0-5 mins", "5-10 mins", "10-15 mins", "15-20 mins", "20+ mins"]
gym_attendance = [95, 85, 65, 40, 15]  # Percentage willing to attend
width = 0.65

# Create bars with gradient color (darker = less willing to travel)
cmap = LinearSegmentedColormap.from_list("travel_willingness", 
                                        [colors["success"], colors["tertiary"], 
                                         colors["highlight"], colors["secondary"], 
                                         colors["secondary"]])
bar_colors = cmap(np.linspace(0, 1, len(distances)))

# Plot the bars
bars = ax1.bar(distances, gym_attendance, width, color=bar_colors, edgecolor='black', linewidth=0.5)

# Add a horizontal line at 70% with annotation
ax1.axhline(y=70, color=colors["dark"], linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(4.0, 72, "70% threshold", color=colors["dark"], fontweight='bold', ha='right')

# Add 15-minute vertical line with annotation
ax1.axvline(x=2.5, color=colors["dark"], linestyle='--', alpha=0.7, linewidth=1.5)
ax1.annotate('15-minute threshold', 
            xy=(2.5, 50),
            xytext=(3.0, 50),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontweight='bold')

# Add shaded area for "Low Attendance Zone"
ax1.add_patch(Rectangle((2.5, 0), 2.5, 70, alpha=0.2, color=colors["secondary"], 
                      hatch='///', edgecolor=colors["secondary"]))
ax1.text(3.75, 30, "LOW ATTENDANCE\nZONE", ha='center', va='center', 
        color=colors["secondary"], fontweight='bold', fontsize=12)

# Add style to the chart
ax1.set_ylabel('Willingness to Attend (%)', fontsize=12, fontweight='bold')
ax1.set_title('Distance Barrier to Fitness Engagement', fontsize=16, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Add annotation explaining the data
ax1.text(-0.2, -15, "Source: HKU Faculty of Architecture Urban Mobility Study, 2023",
         fontsize=9, style='italic')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold')

# ----- 2. BUNDLED SERVICES DEMAND VS AFFORDABILITY GAP -----
ax2 = fig.add_subplot(gs[0, 1])

# Create data for grouped bar chart
categories = ['Want Bundled\nWellness Services', 'Can Afford\nSeparate Memberships']
values = [51, 18]

# Create horizontal bar chart showing the gap
bars = ax2.barh(categories, values, height=0.5, color=[colors["tertiary"], colors["success"]])

# Add the gap visualization
ax2.add_patch(Rectangle((18, 0.25), 33, 0.5, color=colors["secondary"], alpha=0.3))
ax2.annotate('33% AFFORDABILITY GAP', 
            xy=(34, 0.5),
            xytext=(34, 0.5),
            ha='center', va='center',
            color=colors["secondary"],
            fontweight='bold',
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc=colors["light"], ec=colors["secondary"], alpha=0.8))

# Add details note
ax2.text(50, 2.1, "51% of Hong Kong consumers want bundled wellness services\nbut only 18% can currently afford separate memberships",
         ha='center', style='italic', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.5", fc=colors["light"], alpha=0.8,
                 ec=colors["tertiary"], linewidth=1))
# Style the chart
ax2.set_xlim(0, 100)
ax2.set_xlabel('Percentage of Hong Kong Consumers (%)', fontsize=12, fontweight='bold')
ax2.set_title('Wellness Services: Demand vs. Affordability', fontsize=16, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.3)

# Add value labels to the end of bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
             f'{width}%', va='center', fontweight='bold')

# Add source note
ax2.text(0, -0.5, "Source: NielsenIQ Hong Kong Consumer Survey, 2023",
         fontsize=9, style='italic')

# ----- 3. TRAINER INCOME OPTIMIZATION VISUALIZATION -----
ax3 = fig.add_subplot(gs[1, 0])

# Create data for visualization
current_data = {'Category': ['Utilized Time Slots', 'Lost Potential Income'],
               'Value': [70, 30]}

# Create exploded pie chart
explode = (0, 0.1)  # only "explode" the 2nd slice (Lost Income)
wedges, texts, autotexts = ax3.pie(current_data['Value'], 
                                   explode=explode,
                                   labels=None,  # We'll add custom labels
                                   autopct='%1.0f%%',
                                   startangle=90,
                                   colors=[colors["success"], colors["secondary"]])

# Style the percentage texts
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

# Add a white circle at the center to create a donut chart
centre_circle = plt.Circle((0,0), 0.5, fc='white')
ax3.add_artist(centre_circle)

# Add trainer icon and text in the middle
ax3.text(0, 0, "TRAINER\nINCOME", ha='center', va='center', fontsize=14, 
         fontweight='bold', color=colors["dark"])

# Add annotations with arrows
ax3.annotate('Current Income\n(70% of potential)',
            xy=(0.7, 0.7),
            xytext=(1.3, 0.7),
            arrowprops=dict(facecolor=colors["success"], shrink=0.05, width=2),
            fontsize=12, fontweight='bold', ha='center')

ax3.annotate('Lost Income due to\nunderutilized time slots\nand lack of digital tools',
            xy=(-0.7, -0.7),
            xytext=(-1.3, -1.0),
            arrowprops=dict(facecolor=colors["secondary"], shrink=0.05, width=2),
            fontsize=12, fontweight='bold', ha='center')

# Add title and source information
ax3.set_title('Trainer Income Potential Loss', fontsize=16, fontweight='bold')
ax3.text(-1.5, -1.5, "Source: Hong Kong Sports Commission Report, 2021",
         fontsize=9, style='italic')

# ----- 4. SOLUTION OVERVIEW VISUALIZATION -----
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')  # Turn off axis for this panel as we'll use it for a custom solution graphic

# Create a custom solution visual using shapes and annotations

# Background rectangle
solution_bg = Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=colors["light"], 
                      edgecolor=colors["dark"], linewidth=2, alpha=0.5)
ax4.add_patch(solution_bg)

# Add title
title = ax4.text(0.5, 0.88, "1FIT SOLUTION", ha='center', va='center',
               fontsize=18, fontweight='bold', color=colors["dark"])
title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Add central platform icon
central_platform = Circle((0.5, 0.55), 0.15, facecolor=colors["tertiary"], 
                       edgecolor=colors["dark"], linewidth=2)
ax4.add_patch(central_platform)
ax4.text(0.5, 0.55, "1FIT\nPLATFORM", ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')

# Add surrounding elements with connecting lines
# ... [existing code for surrounding elements] ...

# Add key solution points - adjusted y positions to prevent overlap


# Add surrounding elements with connecting lines
# 1. Gyms - Fixed arrowstyle from '-|>' to '->' which is valid
gym = Circle((0.25, 0.7), 0.08, facecolor=colors["success"], alpha=0.8)
ax4.add_patch(gym)
ax4.text(0.25, 0.7, "GYM", ha='center', va='center', fontsize=10, fontweight='bold')
ax4.add_patch(FancyArrowPatch((0.33, 0.7), (0.35, 0.6), 
                            arrowstyle='->', mutation_scale=15, 
                            color=colors["dark"], linewidth=1.5))

# 2. Spas - Fixed arrowstyle
spa = Circle((0.75, 0.7), 0.08, facecolor=colors["highlight"], alpha=0.8)
ax4.add_patch(spa)
ax4.text(0.75, 0.7, "SPA", ha='center', va='center', fontsize=10, fontweight='bold')
ax4.add_patch(FancyArrowPatch((0.67, 0.7), (0.65, 0.6), 
                            arrowstyle='->', mutation_scale=15, 
                            color=colors["dark"], linewidth=1.5))

# 3. Trainers - Fixed arrowstyle
trainer = Circle((0.25, 0.35), 0.08, facecolor=colors["primary"], alpha=0.8)
ax4.add_patch(trainer)
ax4.text(0.25, 0.35, "TRAINER", ha='center', va='center', fontsize=8, fontweight='bold', color='white')
ax4.add_patch(FancyArrowPatch((0.33, 0.35), (0.36, 0.48), 
                            arrowstyle='->', mutation_scale=15, 
                            color=colors["dark"], linewidth=1.5))

# 4. Wellness Centers - Fixed arrowstyle
wellness = Circle((0.75, 0.35), 0.08, facecolor=colors["secondary"], alpha=0.8)
ax4.add_patch(wellness)
ax4.text(0.75, 0.35, "WELLNESS", ha='center', va='center', fontsize=8, fontweight='bold', color='white')
ax4.add_patch(FancyArrowPatch((0.67, 0.35), (0.64, 0.48), 
                            arrowstyle='->', mutation_scale=15, 
                            color=colors["dark"], linewidth=1.5))

# Add users/consumers - Fixed bidirectional arrowstyle from '<-|>' to '<->'
user = Circle((0.5, 0.25), 0.08, facecolor=colors["light"], edgecolor=colors["dark"], linewidth=2)
ax4.add_patch(user)
ax4.text(0.5, 0.25, "USERS", ha='center', va='center', fontsize=10, fontweight='bold')
ax4.add_patch(FancyArrowPatch((0.5, 0.33), (0.5, 0.4), 
                            arrowstyle='<->', mutation_scale=15, 
                            color=colors["dark"], linewidth=1.5))

# Add key solution points
# ax4.text(0.08, 0.8, "✓ Flexible city-wide access", fontsize=12, fontweight='bold', color=colors["success"])
# ax4.text(0.08, 0.75, "✓ Bundled wellness services", fontsize=12, fontweight='bold', color=colors["highlight"])
# ax4.text(0.08, 0.7, "✓ Centralized trainer system", fontsize=12, fontweight='bold', color=colors["primary"])

# Add explanatory notes
benefit_box = Rectangle((0.1, 0.02), 0.8, 0.15, facecolor='white', alpha=0.8,
                      edgecolor=colors["dark"], linewidth=1, linestyle='--')
ax4.add_patch(benefit_box)

benefits = [
    "• Removes 15-minute distance barrier",
    "• Bridges 33% affordability gap",
    "• Optimizes trainer schedules & income"
]

for i, benefit in enumerate(benefits):
    ax4.text(0.15, 0.14 - i*0.04, benefit, fontsize=11, va='center')

# ----- MAIN TITLE AND FOOTER -----
fig.suptitle('Hong Kong Fitness Market: Problems & Solutions', fontsize=22, fontweight='bold', y=0.99)

plt.figtext(0.5, 0.01, "Proposed by 1Fit - Revolutionizing Hong Kong's Fitness & Wellness Industry", 
           fontsize=12, ha='center', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('hk_fitness_market_solution.png', dpi=300, bbox_inches='tight')
plt.show()