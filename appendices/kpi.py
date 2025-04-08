import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

# Create phases data
phases = ['Initial', 'Growth', 'Maturity']

# KPI data
cac = [8, 50, 35]
retention = [0.4, 0.6, 0.75]
partners = [50, 800, 2000]
users = [5000, 60000, 150000]

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig)

# Custom styling
plt.style.use('fivethirtyeight')
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

# 1. CAC Trend
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(phases, cac, 'o-', linewidth=3, color=colors[0], markersize=10)
ax1.set_title('Customer Acquisition Cost (CAC)', fontsize=14)
ax1.set_ylabel('CAC ($)', fontsize=12)
for i, value in enumerate(cac):
    ax1.text(i, value + 2, f"${value}", ha='center', fontsize=12)

# 2. Retention Rate
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(phases, retention, 'o-', linewidth=3, color=colors[1], markersize=10)
ax2.set_title('Retention Rate', fontsize=14)
ax2.set_ylabel('Rate (%)', fontsize=12)
ax2.set_ylim(0, 1)
for i, value in enumerate(retention):
    ax2.text(i, value + 0.05, f"{value:.0%}", ha='center', fontsize=12)

# 3. Partner Growth
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(phases, partners, color=colors[2], alpha=0.7)
ax3.set_title('Partner Locations', fontsize=14)
ax3.set_ylabel('Number of Partners', fontsize=12)
for i, value in enumerate(partners):
    ax3.text(i, value + 100, f"{value:,}", ha='center', fontsize=12)

# 4. User Growth (Log scale)
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(phases, users, color=colors[3], alpha=0.7)
ax4.set_title('Paying Users', fontsize=14)
ax4.set_ylabel('Number of Users', fontsize=12)
ax4.set_yscale('log')
for i, value in enumerate(users):
    ax4.text(i, value * 1.1, f"{value:,}", ha='center', fontsize=12)

# Add growth indicators
for i in range(1, len(phases)):
    user_growth = users[i] / users[i-1] - 1
    ax4.annotate(f"+{user_growth:.0%}", 
                xy=(i-0.5, (users[i] + users[i-1])/2),
                xytext=(i-0.5, (users[i] + users[i-1])/2 * 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, ha='center')

# Title and subtitle
plt.suptitle('1fit Growth Benchmarks by Business Phase', fontsize=18, y=0.98)
plt.figtext(0.5, 0.92, 'Key performance indicators at different business stages', 
            ha='center', fontsize=12, fontstyle='italic')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('kpi_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()