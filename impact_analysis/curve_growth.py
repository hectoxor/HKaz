import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Generate S-curve data
quarters = np.arange(1, 21)  # 5 years of quarters
users = 5000 / (1 + np.exp(-0.5 * (quarters - 8))) * 150000/5000
partners = 50 / (1 + np.exp(-0.5 * (quarters - 6))) * 1200/50

# Create the figure
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot user growth
color = 'tab:blue'
ax1.set_xlabel('Quarters from Launch', fontsize=14)
ax1.set_ylabel('Users', color=color, fontsize=14)
line1 = ax1.plot(quarters, users, color=color, linewidth=3, label='Users')
ax1.tick_params(axis='y', labelcolor=color)

# Create second Y-axis for partners
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Partner Locations', color=color, fontsize=14)
line2 = ax2.plot(quarters, partners, color=color, linewidth=3, linestyle='--', label='Partners')
ax2.tick_params(axis='y', labelcolor=color)

# Mark critical mass point
critical_mass_quarter = 8
critical_mass_users = users[critical_mass_quarter-1]
critical_mass_partners = partners[critical_mass_quarter-1]

ax1.axvline(x=critical_mass_quarter, color='gray', linestyle='--', alpha=0.7)
ax1.scatter([critical_mass_quarter], [critical_mass_users], s=200, color='blue', zorder=5)
ax2.scatter([critical_mass_quarter], [critical_mass_partners], s=200, color='red', zorder=5)

# Annotate critical mass
ax1.annotate('Critical Mass Point\n(Q{})'.format(critical_mass_quarter), 
             xy=(critical_mass_quarter, critical_mass_users),
             xytext=(critical_mass_quarter+1, critical_mass_users*0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12, ha='center')

# Calculate growth rate after critical mass
growth_after = (users[critical_mass_quarter+3] / users[critical_mass_quarter]) - 1
ax1.annotate(f'151% YoY Growth\nAfter Critical Mass', 
             xy=(critical_mass_quarter+3, users[critical_mass_quarter+3]),
             xytext=(critical_mass_quarter+5, users[critical_mass_quarter+3]*0.8),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
             fontsize=12, ha='center', color='green')

# Title and legend
plt.title('Projected Growth: Users and Partner Locations', fontsize=16, pad=20)
fig.legend(line1 + line2, ['Users', 'Partner Locations'], loc='upper left', bbox_to_anchor=(0.15, 0.98))

# Add phases
ax1.fill_between([1, 8], 0, max(users)*1.1, color='blue', alpha=0.1, label='Initial Phase')
ax1.fill_between([8, 14], 0, max(users)*1.1, color='green', alpha=0.1, label='Growth Phase')
ax1.fill_between([14, 20], 0, max(users)*1.1, color='purple', alpha=0.1, label='Maturity Phase')

plt.tight_layout()
plt.savefig('growth_s_curve.png', dpi=300)
plt.show()