import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Create data
markets = ['Central Asia', 'Hong Kong', 'Southeast Asia', 'Western Markets']
multiples = [1.7, 3.5, 2.8, 5.5]
revenue = [100, 30, 50, 20]  # Hypothetical revenue in $100k

# Create a figure
plt.figure(figsize=(12, 8))

# Create bubble chart
sns.scatterplot(x=np.arange(len(markets)), y=multiples, size=revenue, sizes=(100, 1500), 
                alpha=0.6, palette="viridis", legend=False)

# Connect bubbles with arrow to show progression
for i in range(len(markets)-1):
    plt.annotate('', xy=(i+1, multiples[i+1]), xytext=(i, multiples[i]), 
                 arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Add value labels
for i, (market, multiple, rev) in enumerate(zip(markets, multiples, revenue)):
    plt.text(i, multiple + 0.3, f"{multiple}x", ha='center', fontsize=12)
    plt.text(i, multiple - 0.3, f"${rev*100}k", ha='center', fontsize=10, alpha=0.7)

# Highlight the 3x valuation increase
plt.annotate('3x Valuation\nIncrease Potential', 
             xy=(3, multiples[3]), 
             xytext=(2.5, multiples[3] + 1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Customize axes
plt.xticks(np.arange(len(markets)), markets, fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.ylim(0, multiples[-1] + 2)

# Titles and labels
plt.title('Valuation Multiple Comparison by Geographic Market', fontsize=16)
plt.ylabel('Revenue Multiple (x)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('valuation_multiples.png', dpi=300)
plt.show()