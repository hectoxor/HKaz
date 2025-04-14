import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Create data for trends with impact ratings (1-5 scale)
trends = {
    "Emerging Tech (AI, VR, AR)": 4.5,
    "Mobile Commerce": 5.0,
    "Social Commerce": 4.0,
    "Sustainability & Ethical Consumption": 3.5,
    "Live Commerce": 3.0
}

# Sort trends by importance
sorted_trends = dict(sorted(trends.items(), key=lambda item: item[1], reverse=True))
names = list(sorted_trends.keys())
values = list(sorted_trends.values())

# Create colormap for gradient effect
colors = LinearSegmentedColormap.from_list("importance", ["#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8"])
bar_colors = [colors(v/5.0) for v in values]

# Create plot with more professional styling
plt.figure(figsize=(12, 8))
bars = plt.barh(names, values, color=bar_colors, edgecolor='black', linewidth=0.5)
plt.xlabel('Impact Rating (1-5 Scale)', fontsize=12, fontweight='bold')
plt.title('Key E-Commerce & Fitness Technology Trends (2025)', fontsize=16, fontweight='bold')
plt.xlim(0, 5.5)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add annotations explaining each trend
annotations = {
    "Emerging Tech (AI, VR, AR)": "Personalized recommendations, virtual try-ons, immersive fitness",
    "Mobile Commerce": "Drives 63% of HK e-commerce; essential for fitness booking",
    "Social Commerce": "Key platform for marketing and direct sales conversion",
    "Sustainability & Ethical Consumption": "Growing customer preference in purchasing decisions",
    "Live Commerce": "Interactive product demos and real-time fitness classes"
}

# Add value labels and annotations
for i, (bar, name) in enumerate(zip(bars, names)):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}", va='center', fontweight='bold')
    
    # Add annotation text with smaller font below each bar
    plt.text(0.1, bar.get_y() + bar.get_height()/4,
            annotations[name], va='center', color='#444444', 
            fontsize=8, style='italic')

plt.tight_layout()
plt.savefig('ecommerce_fitness_trends.png', dpi=300, bbox_inches='tight')
plt.show()