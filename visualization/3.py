import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Remove axes
ax.axis('off')

# Create SWOT quadrants
quadrant_colors = {'Strengths': '#C5E0B4', 'Weaknesses': '#FFBF86', 
                  'Opportunities': '#BDD7EE', 'Threats': '#F8CECC'}
quadrant_titles = {'Strengths': 'STRENGTHS', 'Weaknesses': 'WEAKNESSES', 
                  'Opportunities': 'OPPORTUNITIES', 'Threats': 'THREATS'}

# SWOT content
swot_data = {
    'Strengths': [
        'Diverse Service Offerings',
        'Flexible Membership Plans',
        'Social Media Integration',
        'Partner Analytics & CRM Features'
    ],
    'Weaknesses': [
        'Limited Local Brand Recognition',
        'No Booking System Support',
        'Over-reliance on Social Media Marketing'
    ],
    'Opportunities': [
        'Growing Health/Fitness Awareness in HK',
        'Digital Fitness Market Growth ($235M by 2025)',
        'Partnerships with Local Wellness Businesses',
        'Corporate Wellness Programs'
    ],
    'Threats': [
        'Intense Competition (ClassPass, Pure, F45)',
        'Economic Uncertainty',
        'Potential Regulatory Changes',
        'Market Saturation Concerns'
    ]
}

# Details for each SWOT item
swot_details = {
    'Diverse Service Offerings': 'Gyms, studios, sports activities',
    'Flexible Membership Plans': 'Multiple pricing tiers (3-24 months)',
    'Social Media Integration': 'TikTok, Instagram engagement',
    'Partner Analytics & CRM Features': 'Business intelligence tools',
    
    'Limited Local Brand Recognition': 'vs. established local competitors',
    'No Booking System Support': 'Missing management solutions',
    'Over-reliance on Social Media Marketing': 'Limited reach to offline segments',
    
    'Growing Health/Fitness Awareness in HK': 'Post-pandemic health focus',
    'Digital Fitness Market Growth ($235M by 2025)': '6.75% CAGR growth trajectory',
    'Partnerships with Local Wellness Businesses': 'Cross-promotion opportunities',
    'Corporate Wellness Programs': 'B2B expansion potential',
    
    'Intense Competition (ClassPass, Pure, F45)': 'Established loyalty programs',
    'Economic Uncertainty': 'Shifting consumer spending',
    'Potential Regulatory Changes': 'Health service regulations',
    'Market Saturation Concerns': 'Increasing fitness options'
}

# Create quadrants
positions = {
    'Strengths': [0.02, 0.52, 0.46, 0.46],
    'Weaknesses': [0.52, 0.52, 0.46, 0.46],
    'Opportunities': [0.02, 0.02, 0.46, 0.46],
    'Threats': [0.52, 0.02, 0.46, 0.46]
}

for quadrant, pos in positions.items():
    rect = patches.Rectangle((pos[0], pos[1]), pos[2], pos[3], 
                           linewidth=2, edgecolor='black', 
                           facecolor=quadrant_colors[quadrant], alpha=0.6)
    ax.add_patch(rect)
    
    # Add quadrant title
    ax.text(pos[0] + 0.01, pos[1] + pos[3] - 0.05, quadrant_titles[quadrant], 
           fontsize=14, fontweight='bold')
    
    # Add content
    y_pos = pos[1] + pos[3] - 0.11
    for i, item in enumerate(swot_data[quadrant]):
        ax.text(pos[0] + 0.03, y_pos - i*0.08, f"• {item}", fontsize=11, fontweight='bold')
        ax.text(pos[0] + 0.05, y_pos - i*0.08 - 0.03, f"   {swot_details[item]}", 
               fontsize=9, fontweight='normal', style='italic')

# Add title and source
plt.figtext(0.5, 0.95, "SWOT Analysis: 1Fit in Hong Kong", fontsize=20, 
            ha='center', fontweight='bold')
plt.figtext(0.5, 0.01, "Sources: 1Fit Internal Analysis, LCSD Surveys, Statista Digital Fitness HK, Market Analysis, Economic Forecasts", 
            fontsize=8, ha='center', style='italic')

# Add "Internal Factors" and "External Factors" labels
plt.figtext(0.5, 0.76, "INTERNAL FACTORS", fontsize=12, ha='center', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', edgecolor='gray'))
plt.figtext(0.5, 0.24, "EXTERNAL FACTORS", fontsize=12, ha='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', edgecolor='gray'))

# Add "Helpful" and "Harmful" labels
plt.figtext(0.25, 0.5, "HELPFUL", fontsize=12, rotation=90, va='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', edgecolor='gray'))
plt.figtext(0.75, 0.5, "HARMFUL", fontsize=12, rotation=90, va='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', edgecolor='gray'))

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
plt.savefig('1fit_swot_analysis.png', dpi=300, bbox_inches='tight')
plt.show()