import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import json
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create synthetic data for districts
districts = [
    'Central & Western', 'Wan Chai', 'Eastern', 'Southern', 
    'Yau Tsim Mong', 'Sham Shui Po', 'Kowloon City', 'Wong Tai Sin', 
    'Kwun Tong', 'Kwai Tsing', 'Tsuen Wan', 'Tuen Mun', 
    'Yuen Long', 'North', 'Tai Po', 'Sha Tin', 
    'Sai Kung', 'Islands'
]

# Create more varied adherence rates to ensure districts fall into different phases
adherence_rates = []
for district in districts:
    # Assign different adherence ranges based on district characteristics
    if district in ['Central & Western', 'Wan Chai', 'Yau Tsim Mong']:
        # High-income, central districts with good facilities
        rate = random.uniform(0.6, 0.8)
    elif district in ['Wong Tai Sin', 'Kwun Tong', 'Sham Shui Po']:
        # Lower-income districts
        rate = random.uniform(0.3, 0.45)
    else:
        # Medium adherence for other districts
        rate = random.uniform(0.45, 0.6)
    adherence_rates.append(rate)

# Create district data
district_data = pd.DataFrame({
    'district': districts,
    'adherence_rate': adherence_rates,
    'phase': pd.cut(adherence_rates, bins=[0, 0.4, 0.6, 1], 
                   labels=['Phase 3', 'Phase 2', 'Phase 1'])
})

# Approximate district centers
district_centers = {
    'Central & Western': [22.2826, 114.1452],
    'Wan Chai': [22.2808, 114.1826],
    'Eastern': [22.2845, 114.2256],
    'Southern': [22.2458, 114.1600],
    'Yau Tsim Mong': [22.3203, 114.1694],
    'Sham Shui Po': [22.3303, 114.1622],
    'Kowloon City': [22.3287, 114.1839],
    'Wong Tai Sin': [22.3419, 114.1953],
    'Kwun Tong': [22.3100, 114.2260],
    'Kwai Tsing': [22.3561, 114.1324],
    'Tsuen Wan': [22.3725, 114.1170],
    'Tuen Mun': [22.3908, 113.9725],
    'Yuen Long': [22.4445, 114.0225],
    'North': [22.4940, 114.1386],
    'Tai Po': [22.4513, 114.1644],
    'Sha Tin': [22.3864, 114.1928],
    'Sai Kung': [22.3809, 114.2707],
    'Islands': [22.2627, 113.9456]
}

# Add coordinates to dataframe
district_data['lat'] = district_data['district'].apply(lambda x: district_centers[x][0])
district_data['lon'] = district_data['district'].apply(lambda x: district_centers[x][1])

# Create a map centered on Hong Kong
hk_map = folium.Map(location=[22.3, 114.1], zoom_start=11, tiles="CartoDB positron")

# Define phase colors with better contrast
phase_colors = {
    'Phase 1': '#1a9641',  # Green
    'Phase 2': '#fdae61',  # Orange
    'Phase 3': '#d7191c'   # Red
}

# Add circular markers with reasonable radius
for _, row in district_data.iterrows():
    color = phase_colors[row['phase']]
    # Use a much smaller radius (5-10 instead of 15000*)
    radius = 5 + (row['adherence_rate'] * 10)  # Scale between 5-13
    
    popup_text = f"""
    <strong>{row['district']}</strong><br>
    Adherence Rate: {row['adherence_rate']:.2f}<br>
    Expansion: {row['phase']}<br>
    Potential Users: {int(row['adherence_rate'] * 50000)}
    """
    
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.7,
        weight=2,
        popup=folium.Popup(popup_text, max_width=200)
    ).add_to(hk_map)
    
    # Add district labels
    folium.Marker(
        location=[row['lat'], row['lon']],
        icon=folium.DivIcon(
            icon_size=(150,36),
            icon_anchor=(75,18),
            html=f'<div style="font-size: 10pt; color: black; text-align: center;">{row["district"]}</div>'
        )
    ).add_to(hk_map)

# FIX: Remove the problematic heatmap approach
# Instead, let's add a choropleth-like visualization that's more reliable

# Create separate feature groups for each phase for toggling
phase1_group = folium.FeatureGroup(name="Phase 1 Districts")
phase2_group = folium.FeatureGroup(name="Phase 2 Districts")
phase3_group = folium.FeatureGroup(name="Phase 3 Districts")

# Add districts to their respective phase groups
for _, row in district_data.iterrows():
    color = phase_colors[row['phase']]
    
    # Create a larger semi-transparent circle to represent district area
    area_marker = folium.Circle(
        location=[row['lat'], row['lon']],
        radius=1000,  # 1km radius
        color=color,
        fill=True,
        fill_opacity=0.3,
        weight=1
    )
    
    # Add to appropriate group
    if row['phase'] == 'Phase 1':
        area_marker.add_to(phase1_group)
    elif row['phase'] == 'Phase 2':
        area_marker.add_to(phase2_group)
    else:
        area_marker.add_to(phase3_group)

# Add all phase groups to the map
phase1_group.add_to(hk_map)
phase2_group.add_to(hk_map)
phase3_group.add_to(hk_map)

# Add a legend
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; background-color: white; 
    padding: 10px; border: 2px solid grey; border-radius: 5px">
<p><strong>Expansion Strategy</strong></p>
<p><i class="fa fa-circle" style="color:#1a9641"></i> Phase 1: High Adherence (>60%)</p>
<p><i class="fa fa-circle" style="color:#fdae61"></i> Phase 2: Medium Adherence (40-60%)</p>
<p><i class="fa fa-circle" style="color:#d7191c"></i> Phase 3: Low Adherence (<40%)</p>
</div>
"""
hk_map.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl().add_to(hk_map)

# Save the map
hk_map.save('hk_district_strategy.html')

print("Map saved to hk_district_strategy.html")
print("District breakdown by phase:")
print(district_data['phase'].value_counts())