import matplotlib.pyplot as plt
import numpy as np

# Trends and assigned importance scores (adjust scores as needed)
trends = {
    "Emerging Tech (AI, VR)": 4.5,
    "Mobile Commerce": 5.0,
    "Social Commerce": 4.0,
    "Sustainability": 3.5,
    "Live Commerce": 3.0
}

# Sort trends by importance
sorted_trends = dict(sorted(trends.items(), key=lambda item: item[1]))
names = list(sorted_trends.keys())
values = list(sorted_trends.values())

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(names, values, color='skyblue')
ax.set_xlabel('Perceived Importance (1-5 Scale)')
ax.set_title('Key E-commerce & Fitness Tech Trends Impact')
ax.set_xlim(0, 5.5)

# Add value labels
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
            f'{bar.get_width():.1f}',
            va='center')

plt.tight_layout()
plt.show()
# To save the plot:
plt.savefig('trends_impact_chart.png', dpi=300)