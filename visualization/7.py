import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from matplotlib.textpath import TextPath
import matplotlib.transforms as transforms
import matplotlib.patheffects as PathEffects

# Set up the figure
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Define colors
colors = {
    'background': '#F5F7FA',
    'stage': '#3498DB',  # Blue
    'pain': '#E74C3C',   # Red
    'solution': '#27AE60',  # Green
    'arrow': '#34495E',  # Dark blue
    'text': '#2C3E50',   # Dark gray
    'highlight': '#F39C12',  # Orange
    'light_blue': '#D6EAF8'
}

# Set background color
fig.patch.set_facecolor(colors['background'])
ax.set_facecolor(colors['background'])

# Define the stages of the customer journey
stages = [
    "Discovery", 
    "Sign Up", 
    "Initial Use", 
    "Regular Usage", 
    "Support", 
    "Renewal"
]

# Define the pain points for each stage
pain_points = [
    "Rigid plan options\nlimit exploration",
    "Complex booking\nsystems",
    "Limited class\navailability",
    "Inconsistent\nquality",
    "Slow response\ntimes",
    "Inflexible renewal\noptions"
]

# Define the 1FIT solutions
solutions = [
    "Flexible subscription\noptions",
    "Streamlined booking\nprocess",
    "Multi-location\naccess",
    "Quality standards\nacross partners",
    "24/7 in-app\nsupport",
    "Customizable plan\nadjustments"
]

# Calculate positions
stage_x_positions = np.linspace(10, 90, len(stages))
stage_y_position = 45
pain_y_position = 25
solution_y_position = 10
box_width = 12
stage_height = 6
pain_solution_height = 8

# Draw the main flow path connecting stages
# Create points for the path
path_points = []
for i, x in enumerate(stage_x_positions):
    if i == 0:
        path_points.extend([(x, stage_y_position)])
    else:
        path_points.extend([
            (x - box_width/2 - 2, stage_y_position),
            (x, stage_y_position)
        ])

# Draw the path with a nice arrow
for i in range(len(stages)-1):
    x1 = stage_x_positions[i] + box_width/2
    x2 = stage_x_positions[i+1] - box_width/2
    
    # Draw arrow connecting boxes
    arrow = patches.FancyArrowPatch(
        (x1, stage_y_position), 
        (x2, stage_y_position),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2,
        color=colors['arrow'],
        zorder=2
    )
    ax.add_patch(arrow)

# Draw the boxes for each stage and add the text
for i, (stage, x) in enumerate(zip(stages, stage_x_positions)):
    # Stage boxes with shadow effect
    shadow = patches.Rectangle(
        (x - box_width/2 + 0.5, stage_y_position - stage_height/2 - 0.5),
        box_width, stage_height, 
        facecolor='#B3B6B7',
        edgecolor='none',
        alpha=0.6,
        zorder=1
    )
    ax.add_patch(shadow)
    
    stage_box = patches.Rectangle(
        (x - box_width/2, stage_y_position - stage_height/2),
        box_width, stage_height, 
        facecolor=colors['stage'],
        edgecolor='white',
        linewidth=2,
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(stage_box)
    
    # Stage number
    stage_num = ax.text(
        x - box_width/2 + 1.5, stage_y_position, 
        f"{i+1}", 
        color='white', 
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=3
    )
    
    # Add circular background for the number
    circle = plt.Circle(
        (x - box_width/2 + 1.5, stage_y_position),
        1.2, 
        color='white', 
        alpha=0.3, 
        zorder=2.5
    )
    ax.add_patch(circle)
    
    # Stage text
    stage_text = ax.text(
        x + 0.5, stage_y_position, 
        stage, 
        color='white', 
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=3
    )
    
    # Pain point boxes with connecting lines
    pain_box = patches.FancyBboxPatch(
        (x - box_width/2, pain_y_position - pain_solution_height/2),
        box_width, pain_solution_height, 
        boxstyle=patches.BoxStyle.Round(pad=0.3),
        facecolor=colors['pain'],
        edgecolor='none',
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(pain_box)
    
    # Pain point text
    pain_text = ax.text(
        x, pain_y_position, 
        pain_points[i], 
        color='white', 
        fontsize=10,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=3
    )
    
    # Solution boxes
    solution_box = patches.Rectangle(
        (x - box_width/2, solution_y_position - pain_solution_height/2),
        box_width, pain_solution_height, 
        facecolor=colors['solution'],
        edgecolor='none',
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(solution_box)
    
    # Solution text
    solution_text = ax.text(
        x, solution_y_position, 
        solutions[i], 
        color='white', 
        fontsize=10,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=3
    )
    
    # Connect pain points to stages
    connector1 = patches.FancyArrowPatch(
        (x, stage_y_position - stage_height/2),
        (x, pain_y_position + pain_solution_height/2),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-",
        linestyle='--',
        linewidth=1.5,
        color=colors['pain'],
        alpha=0.7,
        zorder=1.5
    )
    ax.add_patch(connector1)
    
    # Connect solutions to pain points
    connector2 = patches.FancyArrowPatch(
        (x, pain_y_position - pain_solution_height/2),
        (x, solution_y_position + pain_solution_height/2),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="->",
        linewidth=1.5,
        color=colors['solution'],
        alpha=0.7,
        zorder=1.5
    )
    ax.add_patch(connector2)

# Add title and subtitle
title = ax.text(
    50, 57, 
    "Customer Journey: Pain Points & 1FIT Solutions",
    color=colors['text'],
    fontsize=20, 
    fontweight='bold',
    ha='center'
)
title.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

subtitle = ax.text(
    50, 53, 
    "Addressing Key Customer Pain Points Across the Fitness Service Experience",
    color=colors['text'],
    fontsize=14,
    fontstyle='italic',
    ha='center'
)

# Add legend
legend_y = 4
legend_items = [
    (colors['stage'], "Journey Stage"),
    (colors['pain'], "Customer Pain Point"),
    (colors['solution'], "1FIT Solution")
]

for i, (color, label) in enumerate(legend_items):
    x_pos = 15 + i * 25
    legend_box = patches.Rectangle(
        (x_pos - 2, legend_y - 1),
        4, 2, 
        facecolor=color,
        edgecolor='none',
        alpha=0.9
    )
    ax.add_patch(legend_box)
    
    ax.text(
        x_pos + 4, legend_y, 
        label, 
        color=colors['text'], 
        fontsize=12,
        va='center'
    )

# Add 1FIT logo at bottom right
ax.text(
    95, 2, 
    "1FIT",
    color=colors['stage'],
    fontsize=18, 
    fontweight='bold',
    ha='right',
    va='center'
)

plt.tight_layout()
plt.savefig('customer_journey_flowchart.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()