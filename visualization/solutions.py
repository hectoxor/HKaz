import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.patheffects as path_effects

# Set up figure with compact design
fig, ax = plt.subplots(figsize=(12, 3.5), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis('off')

# Define colors with better contrast
colors = {
    "primary": "#2C3E50",    # Dark blue
    "secondary": "#3498DB",   # Blue
    "tertiary": "#2ECC71",    # Green
    "highlight": "#E74C3C",   # Red
    "background": "#F8F9FA",  # Light gray
    "text": "#34495E"         # Dark text
}

# Add title with better styling
title = ax.text(50, 27, "1FIT Solution Strategy", 
             fontsize=16, fontweight='bold', color=colors['primary'], ha='center')
title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Define the solutions
solutions = [
    {
        "title": "PILOTING",
        "subtitle": "High-Adherence Districts",
        "points": ["Central & Western (73%)", "Wan Chai (61%)", "Yau Tsim Mong (75%)"],
        "icon": "🎯",
        "color": colors["secondary"]
    },
    {
        "title": "TOGETHER",
        "subtitle": "Ecosystem Integration",
        "points": ["Unified platform", "Multiple providers", "Quality standards"],
        "icon": "🤝",
        "color": colors["highlight"]
    },
    {
        "title": "ACCELERATION",
        "subtitle": "Technology-Enabled Growth",
        "points": ["Simplified booking", "Personalized recs", "Capacity optimization"],
        "icon": "🚀",
        "color": colors["tertiary"]
    }
]

# Create a horizontal flow with better positioning
x_positions = [20, 50, 80]
y_pos = 15

# Draw arrows between solutions
for i in range(len(solutions)-1):
    # Main arrow with better styling
    arrow = patches.FancyArrowPatch(
        (x_positions[i] + 11, y_pos), 
        (x_positions[i+1] - 11, y_pos),
        connectionstyle="arc3,rad=0.1",
        arrowstyle="fancy",
        mutation_scale=15,
        linewidth=2,
        color=colors["primary"],
        alpha=0.7,
        zorder=2
    )
    ax.add_patch(arrow)

# Draw each solution pillar with improved spacing
for i, solution in enumerate(solutions):
    # Calculate box height based on content
    box_height = 18  # Reduced height for compactness
    
    # Shadow effect
    shadow_box = patches.FancyBboxPatch(
        (x_positions[i] - 11.5, y_pos - 8.5), 
        23, box_height,
        boxstyle=patches.BoxStyle.Round(pad=0.3),
        facecolor='gray',
        alpha=0.2,
        zorder=1
    )
    ax.add_patch(shadow_box)
    
    # Main solution box - slightly smaller for better containment
    box = patches.FancyBboxPatch(
        (x_positions[i] - 11, y_pos - 8), 
        22, box_height,
        boxstyle=patches.BoxStyle.Round(pad=0.3),
        facecolor='white',
        edgecolor=solution["color"],
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)

    # Header bar - aligned perfectly with box
    header_bar = patches.FancyBboxPatch(
        (x_positions[i] - 11, y_pos + 4), 
        22, 6,
        boxstyle=patches.BoxStyle.Round(pad=0.1),
        facecolor=solution["color"],
        alpha=0.2,
        edgecolor='none',
        zorder=3
    )
    ax.add_patch(header_bar)
    
    # Icon circle - positioned within header
    icon_circle = plt.Circle((x_positions[i] - 7, y_pos + 7), 2.5, 
                          facecolor=solution["color"], alpha=0.3, zorder=3)
    ax.add_artist(icon_circle)
    
    # Icon - aligned with circle
    ax.text(
        x_positions[i] - 7, y_pos + 7,
        solution["icon"],
        fontsize=11,
        ha='center',
        va='center',
        zorder=4
    )
    
    # Title - positioned properly in header
    title_x_position = x_positions[i] + 1
    
    # Move ACCELERATION title more to the left
    if i == 2:  # The ACCELERATION title is the third one (index 2)
        title_x_position = x_positions[i] - 0.5  # Move 1.5 units more to the left
    
    ax.text(
        title_x_position, y_pos + 7,
        solution["title"],
        fontsize=11, 
        fontweight='bold',
        color=solution["color"],
        ha='left',
        va='center',
        zorder=4
    )
    
    # Subtitle - positioned closer to header
    ax.text(
        x_positions[i], y_pos + 1,
        solution["subtitle"],
        fontsize=9,
        fontstyle='italic',
        color=colors["text"],
        ha='center',
        va='center',
        zorder=3
    )
    
    # Bullet points - closer together and properly contained
    for j, point in enumerate(solution["points"]):
        # Reduced spacing between points
        point_y = y_pos - 2.5 - (j * 2.5)  # Reduced from 3.5 to 2.5
        
        # Bullet
        bullet = plt.Circle((x_positions[i] - 8, point_y), 0.6, 
                         facecolor=solution["color"], zorder=3)
        ax.add_artist(bullet)
        
        # Point text - positioned to stay within box
        ax.text(
            x_positions[i] - 6.5, point_y,
            point,
            fontsize=9,  # Reduced size
            color=colors["text"],
            ha='left',
            va='center',
            zorder=3
        )

# Add bottom connector
connector_box = patches.FancyBboxPatch(
    (25, 3), 
    50, 2.5,
    boxstyle=patches.BoxStyle.Round(pad=0.2),
    facecolor=colors["primary"],
    alpha=0.1,
    zorder=1
)
ax.add_patch(connector_box)

# Bottom text
bottom_text = ax.text(50, 4.3, "Creating a Unified Wellness Ecosystem", 
                   fontsize=10,
                   color=colors["primary"], ha='center',
                   va='center')

plt.tight_layout()
plt.savefig('1fit_solution_strategy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()