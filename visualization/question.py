import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.patheffects as path_effects

# Set up figure with more space
fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Define colors - using a clearer color scheme
colors = {
    "primary": "#2C3E50",    # Dark blue
    "secondary": "#3498DB",   # Light blue
    "highlight": "#E74C3C",   # Red
    "accent": "#F39C12",      # Orange
    "background": "#F8F9FA"   # Light gray
}

# Add title with more space below
title = ax.text(50, 55, "Key Questions Facing Hong Kong's Fitness Market", 
             fontsize=18, fontweight='bold', color=colors['primary'], ha='center')
title.set_path_effects([path_effects.withStroke(linewidth=4, foreground='white')])

# Redefine questions with cleaner text
questions = [
    {
        "metric": "70%", 
        "challenge": "Mobility Challenge", 
        "question_line1": "How do we overcome",
        "question_line2": "the 15-minute barrier?",
        "icon": "🚶",
        "color": colors["secondary"]
    },
    {
        "metric": "33%", 
        "challenge": "Affordability Gap", 
        "question_line1": "How do we bridge",
        "question_line2": "the wellness service gap?",
        "icon": "💰",
        "color": colors["highlight"]
    },
    {
        "metric": "30%", 
        "challenge": "Capacity Loss", 
        "question_line1": "How can we optimize",
        "question_line2": "underutilized revenue?",
        "icon": "📊",
        "color": colors["accent"]
    }
]

# Position boxes with more space between them
x_positions = [20, 50, 80]
y_pos = 30

# Draw each question box with clear spacing
for i, q in enumerate(questions):
    # Create question box with more distinct appearance
    box = patches.FancyBboxPatch(
        (x_positions[i] - 15, y_pos - 18), 
        30, 36,  # Taller box for better spacing
        boxstyle=patches.BoxStyle.Round(pad=0.6),
        facecolor=colors["background"],
        edgecolor=q["color"],
        linewidth=3,
        alpha=0.95,
        zorder=1
    )
    ax.add_patch(box)
    
    # Clear vertical spacing between elements
    
    # 1. Challenge name at top with colored background for emphasis
    challenge_bg = patches.FancyBboxPatch(
        (x_positions[i] - 13, y_pos + 10), 
        26, 6,
        boxstyle=patches.BoxStyle.Round(pad=0.3),
        facecolor=q["color"],
        alpha=0.2,
        zorder=2
    )
    ax.add_patch(challenge_bg)
    
    challenge_text = ax.text(
        x_positions[i], y_pos + 13,
        q["challenge"],
        fontsize=12, 
        fontweight='bold',
        color=q["color"],
        ha='center',
        zorder=3
    )
    
    # 2. Large metric with clear space around it
    metric_text = ax.text(
        x_positions[i], y_pos + 1,
        q["metric"],
        fontsize=24, 
        fontweight='bold',
        color=q["color"],
        ha='center',
        va='center',
        zorder=3
    )
    
    # 3. Question text with clear spacing and consistent formatting
    ax.text(
        x_positions[i], y_pos - 8,
        q["question_line1"],
        fontsize=11,
        color=colors["primary"],
        ha='center',
        va='center',
        zorder=3
    )
    
    # Bold emphasis on second line
    ax.text(
        x_positions[i], y_pos - 13,
        q["question_line2"],
        fontsize=11, 
        fontweight='bold',
        color=colors["primary"],
        ha='center',
        va='center',
        zorder=3
    )
    
    icon_x_position = x_positions[i] - 10
    
    # Move icons more to the left for the first two challenges
    if i < 2:  # For Mobility Challenge and Affordability Gap
        icon_x_position = x_positions[i] - 12  # Move 2 units more to the left
    
    icon_circle = plt.Circle((icon_x_position, y_pos + 13), 2.5, 
                           facecolor=q["color"], 
                           alpha=0.9, zorder=3)
    ax.add_artist(icon_circle)
    
    ax.text(
        icon_x_position, y_pos + 13,
        q["icon"],
        fontsize=10,
        ha='center',
        va='center',
        color='white',
        zorder=4
    )

# Add connecting element at bottom with clearer formatting
# Use a box instead of just a line for better visibility
connector_box = patches.FancyBboxPatch(
    (25, 5), 
    50, 6,
    boxstyle=patches.BoxStyle.Round(pad=0.3),
    facecolor=colors["primary"],
    alpha=0.1,
    zorder=1
)
ax.add_patch(connector_box)

bottom_text = ax.text(50, 8, "How can 1FIT address these challenges?", 
                    fontsize=13, fontweight='bold', color=colors["primary"], 
                    ha='center', va='center')

plt.tight_layout()
plt.savefig('hk_fitness_market_questions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()