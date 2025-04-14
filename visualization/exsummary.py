import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Set up the figure with high resolution and good dimensions
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#FFFFFF')

# Define color scheme
colors = {
    "primary": "#1F77B4",    # Blue for main elements
    "secondary": "#FF7F0E",  # Orange for secondary elements
    "tertiary": "#2CA02C",   # Green for positive metrics
    "highlight": "#D62728",  # Red for emphasis
    "accent1": "#9467BD",    # Purple
    "accent2": "#8C564B",    # Brown
    "accent3": "#E377C2",    # Pink
    "background": "#FFFFFF", # White background
    "text": "#333333",       # Dark text
    "muted": "#7F7F7F",      # Gray for less important text
}

# Create a grid layout
gs = gridspec.GridSpec(5, 6, height_ratios=[0.8, 0.2, 1.5, 1.5, 1.8])

# ----- HEADER/LOGO SECTION -----
# ----- HEADER/LOGO SECTION -----
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis('off')

# Add company logo/name - Replace Rectangle with FancyBboxPatch
logo_bg = patches.FancyBboxPatch((0.01, 0.1), 0.15, 0.8, 
                                boxstyle=patches.BoxStyle.Round(pad=0.1),
                                facecolor=colors["primary"], alpha=0.9,
                                transform=ax_header.transAxes, zorder=2)
ax_header.add_patch(logo_bg)
logo = ax_header.text(0.085, 0.5, "1FIT", fontsize=32, fontweight='bold', 
                    color='white', ha='center', va='center', transform=ax_header.transAxes, zorder=3)

# Add executive summary title
title = ax_header.text(0.55, 0.5, "EXECUTIVE SUMMARY", fontsize=36, fontweight='bold',
                     color=colors["primary"], ha='center', va='center', transform=ax_header.transAxes)
title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#E0E0E0')])

# Add date
ax_header.text(0.95, 0.3, "April 2025", fontsize=12, ha='right',
              color=colors["muted"], transform=ax_header.transAxes)

# ----- VISION SECTION -----
ax_vision = fig.add_subplot(gs[1, 1:5])
ax_vision.axis('off')

# Add vision text
vision_text = "With our flexible access model, 1FIT's vision of creating value through customer-centric fitness\nsolutions is no longer a dream but a reality - transforming how Hong Kong residents access wellness services citywide."
vision = ax_vision.text(0.5, 0.5, vision_text, fontsize=14, ha='center', va='center', 
                      style='italic', color=colors["text"], transform=ax_vision.transAxes)
vision.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# ----- CHALLENGES SECTION -----
ax_challenges_header = fig.add_subplot(gs[2, :])
ax_challenges_header.axis('off')

# Add challenges header
ax_challenges_header.text(0.03, 0.9, "Key Challenges", fontsize=24, fontweight='bold',
                        color=colors["primary"])

# Draw horizontal line
ax_challenges_header.axhline(y=0.8, xmin=0.03, xmax=0.97, color=colors["primary"], linewidth=2, alpha=0.5)

# Create three challenge visuals
# 1. Low Mobility Challenge with mini bar chart
ax_mobility = fig.add_subplot(gs[2, 0:2])
distances = ["0-5 mins", "5-10 mins", "10-15 mins", ">15 mins"]
attendance = [95, 85, 70, 30]
mobility_bars = ax_mobility.bar(distances, attendance, color=cm.YlOrRd(np.linspace(0.3, 0.8, 4)))
ax_mobility.set_ylim(0, 100)
ax_mobility.set_title("Low Mobility Barrier", fontsize=16, fontweight='bold', pad=10, color=colors["primary"])
ax_mobility.set_ylabel("Willingness to Attend (%)")
ax_mobility.axhline(y=70, color='red', linestyle='--', alpha=0.7)
ax_mobility.text(3, 72, "70% threshold", color='red', fontsize=10)
mobility_note = "HKU Faculty of Architecture, 2023"

for bar in mobility_bars:
    height = bar.get_height()
    ax_mobility.text(bar.get_x() + bar.get_width()/2., height + 3,
                   f'{height}%', ha='center', fontsize=9, fontweight='bold')

ax_mobility.text(0.5, 0.01, mobility_note, transform=ax_mobility.transAxes, ha='center',
               fontsize=9, style='italic', color=colors["muted"])

# 2. Affordability Gap with bullet chart
ax_afford = fig.add_subplot(gs[2, 2:4])
ax_afford.axis('on')
ax_afford.set_xlim(0, 100)
ax_afford.set_ylim(0, 2)
ax_afford.set_yticks([])
ax_afford.set_xlabel('Percentage of Hong Kong Consumers (%)')

# Add bars for wants vs. can afford
ax_afford.barh(0.7, 51, height=0.5, color=colors["primary"], alpha=0.7)
ax_afford.barh(1.3, 18, height=0.5, color=colors["tertiary"], alpha=0.7)

# Add gap visualization
ax_afford.add_patch(patches.Rectangle((18, 0.7), 33, 0.5, color=colors["highlight"], alpha=0.2))
ax_afford.annotate('33% GAP', xy=(34, 0.95), xytext=(34, 0.95), fontsize=12,
                 fontweight='bold', color=colors["highlight"], ha='center')

# Add labels
ax_afford.text(52, 0.7, "51% want bundled services", va='center', fontsize=11, fontweight='bold')
ax_afford.text(19, 1.3, "18% can afford separate memberships", va='center', fontsize=11, fontweight='bold')
ax_afford.set_title("Affordability Gap", fontsize=16, fontweight='bold', pad=10, color=colors["primary"])
ax_afford.text(0.5, 0.01, "NielsenIQ Hong Kong, 2023", transform=ax_afford.transAxes, ha='center',
            fontsize=9, style='italic', color=colors["muted"])

# 3. Trainer Income Loss with pie chart
ax_trainer = fig.add_subplot(gs[2, 4:6])
trainer_sizes = [70, 30]  # 70% utilized, 30% lost
trainer_colors = [colors["tertiary"], colors["highlight"]]
trainer_explode = (0, 0.1)

wedges, texts = ax_trainer.pie(trainer_sizes, explode=trainer_explode, colors=trainer_colors,
                             shadow=False, startangle=90, wedgeprops=dict(width=0.5))

# Add center circle to create donut chart
centre_circle = plt.Circle((0,0), 0.3, fc='white')
ax_trainer.add_artist(centre_circle)

# Add text in center
ax_trainer.text(0, 0, "Trainer\nIncome", ha='center', va='center', fontsize=12, fontweight='bold')

# Add annotations
ax_trainer.annotate('70% Utilized', xy=(0.7, 0.3), xytext=(1.0, 0.3),
                 arrowprops=dict(arrowstyle='->'), fontsize=12)
ax_trainer.annotate('30% Lost Potential', xy=(-0.6, -0.5), xytext=(-1.2, -0.7),
                  arrowprops=dict(arrowstyle='->'), fontsize=12)

ax_trainer.set_title("Trainer Income Loss", fontsize=16, fontweight='bold', pad=10, color=colors["primary"])
ax_trainer.text(0, -1.1, "Hong Kong Sports Commission, 2021", ha='center',
              fontsize=9, style='italic', color=colors["muted"])

# ----- SOLUTIONS SECTION -----
ax_solutions_header = fig.add_subplot(gs[3, :])
ax_solutions_header.axis('off')

# Add solutions header
ax_solutions_header.text(0.03, 0.9, "Solutions", fontsize=24, fontweight='bold',
                       color=colors["primary"])

# Draw horizontal line
ax_solutions_header.axhline(y=0.8, xmin=0.03, xmax=0.97, color=colors["primary"], linewidth=2, alpha=0.5)

# Create solution panels with visual maps
solution_titles = ["PILOTING", "TOGETHER", "Acceleration"]
solution_subtitles = [
    "in High-Adherence Districts",
    "We Connect Ecosystem Partners",
    "to Comprehensive Wellness"
]

# 1. Piloting
# Replace the district_bg image loading code with this:

# 1. Piloting
ax_sol1 = fig.add_subplot(gs[3, 0:2])
ax_sol1.axis('off')

# Use PIL and urllib to properly load image from URL
try:
    import urllib.request
    from PIL import Image
    import io
    
    # Attempt to load an image from URL
    url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAk1BMVEU0R/b///8nPfapsPovQ/YjO/bV2f319v9YZvju8P7Fyv0yRfYdNvUrQPYuQvYlPPbLz/w1Sfawtvvk5/7R1f3q7P76+v/BxvxOXvdicPiFj/lsefidpfpAU/fy8/9+ifk7TvZyfvh3g/mVnfq7wPyRmvpJWvfb3v1ebPeiqvu0uvsSMPWLlPnf4v6HkfmmrftpdfhPbI+4AAAHPUlEQVR4nO2ce1PyOhDGoaQUCCmXIoIKgiLgBTnf/9Md9XhkO/Pm6RaSSXxnf/9axiwkz17TRkMQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEH4u9Am4aLpx9ifwij/BiZPLS6d03L07Mj+GGI8922iTp+bbIiF5ob/Mcht7tnCwRN/MYU+bdPB2I2B2VaD1Tkgua+xmmtDvpmdGwuL3K+FyW1WYzV3yc8H9aLrxsLrgVcDzb6OgfTI5HM3BtJvzQP5rKizmCU5MkmN4wt5MGCBFxu4qHeW2qOThem1GwOzjkdnoUaTeqt5SX8+q/XBjYXFyJ/QaNOruZq304ZSs1oH2E7Pn5TqpLZD2xN/v3ZjYPPZm9DUCmX+42pB/H3tT1u49yY0aX0tnJw8lzZDNwZOvQlNWieU+eZIhGZVy83YufIlNMk5/npNhGY/dWPh0JPQmHOUcDrzkFi0/MRs+Wx5xmIKRSIaR4mFJ6FR5x2iHjmGqmasYGXmQ2hU47zlPZLEYnXOJvgDy4aHY6iVNaK8hr8tTSz+cWNgs23cW6jNi+3f9TpIILs0sXh0ZCGJdJ0ZaA9ldg3oI3dEaNINeDDjM71xLjR68GZbWHfWbyELx2QxfbCdp7cdPivnm9Relcn2JoeRGEks9AJs5yLPFRv3Bq6tnv4h0asrYOB0zkwsNn7rLhizt4r8e1oRiXXJhkruwINvPqsSFeRb64/U6ldFYjSA7KPtvPdfpbehFlZ92KS6KhIjAaTWYDt3PZc/AVq3bauafJ34xPr3T0hJTCG/2XPv4bgGvlq3VrFSX98AjMRIfm+Q3/Rb/kQG9q1Oern9CsfUHhlIE4sUbedQx1APrN78wxF+PWKsscAndPOh7TztBzqG9gbT9OF7W6XWePWTR2bHYtIPY2D6YF3S2/9L78P69/yUWMDtfAzj75Nb64qe0+9dhVO+bMHsWMx9tzv/iJlbF//yk5/hlG+XMzsWOsQxVB2rgx6eankJFBoSbOoEdCzaIXyFbox37T/TW5ykHaZ8zXeSWDTAc+Mg/l4P+hbS05aCv0zJy8HtvA4Ydlegt6hpXUosQMci24YLu6vAQtMmPiAFPblDuLC7Ety0PhIB6QOv4qGw5AzctL6nFQzwnPvCkjO0QRFNqWMBEgt/rbLL0StYwSCim4CCXOGhvusKPB1DE4sUJBabiI8hLmLTUagREJr3RLMIYSEWGhJO5yix6OQsAz3Ol9hBRexmkxxDmCd3WWSPAYJXnSIDi9fTkwnMk3l4nfOygGs0RECwV+ERxKdgoaEdi+3lrdEr9+2YarDQdJiJBZNhCK8JhaY7ctsabQU4hjiimRDpQ4kFlxDBa26vVH1+53QG4/JRqGmIYhXeew/MxIJJN4TQDJDQUHHHPzaPifvGbyXaoL1XkJGXwfFyC8chIpotmnIbnur0TmYu3wJYiPce7ViMHNyxCNGcMlBoyCiU6lxuYBad0NCOhXm/3MJdCKGBFwsOpF1mHMxchhhG0VsU0dA7Fgp2+nk8hRAaGE3TjoWLy1whhAZHNHu3/n4aJHVC0fSSCAPWXB4h6v5aIaEZEmEwsAHHI8S4jV6hFbWIv0dfxaTHYujviowdfLrueZe5ltuUdzU9RJsfDhqWOhZr+3OHMGVeHgZFNAdawQBfxSbeplNFREMvb6OZyxBFXi4KRjSljgWYufT+1oALMPZxqSY7saDheXRAoaGXt1FrdBex0GgoNG3mzOUm4mOoR6hGQxOLHCQWIRIGLqqDajQ3zMQi4PR6JVhomB2LaZCeJxMoNFfMjoWPG2jOSFF9cEJHn0FiEfOEgtZIaJ7JzGUDtEbfI47Z1AxFNPSOBaoaRzwohO9oZTSxAIoUpNfCBb78gb4VCinS7xUamlgkoJgTotfCBUc0j8TfG+DvI55IxHe0yokFePDXRjTcxCJqoUlRx7NNhCYFM5ftkNdiK8Adz9LlbeDvWzELDXxBBh2FQleGfq/Q0MQCDdhGHdGgy6ClUSgwc5klEQuN/dplszx+hq4MTWJOLKDQHOl7r0BN1dNbgpwAK6Alf48SixDzsFyw0BA/nq95z0UHfIkCfXHqACQW2Sv4D6FBgUr5jgU4r8NAV5tZ9NH7ougoFCp1tGKW0gSVSunl7Rl4LuI7lXiGa8l9S4TvNzlfAlz4zvA6FoWP15G5AgrNC71TCToWw4iPIVx4qQT6Cp67izmiWaEZLlKZUFvwXNRCg14L1SXHC13myry899ARcFiUJgwosTgEmBZlA9/WUrq8DRKLXswRzQAJDb1jgS7UBHtnEgP8/jni7+FU2DriKRN45bBL71SixCLmKRMoNDSxGICORRHzMYRCQ0ehcpBYxNz8bSTouuuc2bGIeZxN66HltTyfkHq+mtve3/NBzD2Zj+wwtUP1Q4HnYjZQEARBEARBEARBEARBEARBEARBEARBEARBEARBEARBEARB8MW//wCGjpAz8E8AAAAASUVORK5CYII='
    with urllib.request.urlopen(url) as response:
        district_bg = np.array(Image.open(io.BytesIO(response.read())))
    ax_sol1.imshow(district_bg, aspect='auto', alpha=0.3, extent=[0, 10, 0, 10])
except:
    # If image loading fails, create a simple Hong Kong outline
    print("Could not load image from URL, using placeholder")
    
    # Create a simple background representing Hong Kong
    x = np.linspace(0, 10, 100)
    y1 = 4 + 0.5*np.sin(x*1.5) + np.random.normal(0, 0.1, 100)  # Top coastline
    y2 = 3 + 0.3*np.sin(x*2) + np.random.normal(0, 0.1, 100)    # Bottom coastline
    
    ax_sol1.fill_between(x, y1, y2, color=colors["primary"], alpha=0.1)
    ax_sol1.plot(x, y1, '-', color=colors["primary"], alpha=0.5, linewidth=1)
    ax_sol1.plot(x, y2, '-', color=colors["primary"], alpha=0.5, linewidth=1)
if district_bg is not None:
    ax_sol1.imshow(district_bg, aspect='auto', alpha=0.3, extent=[0, 10, 0, 10])

# Add district markers
districts = [
    {"name": "Central & Western", "x": 3, "y": 5, "adherence": 73},
    {"name": "Wan Chai", "x": 5, "y": 5, "adherence": 61},
    {"name": "Yau Tsim Mong", "x": 7, "y": 5, "adherence": 75}
]

for district in districts:
    circle_size = district["adherence"] / 10
    ax_sol1.add_patch(plt.Circle((district["x"], district["y"]), circle_size, 
                              color=colors["primary"], alpha=0.7))
    ax_sol1.text(district["x"], district["y"], f"{district['name']}\n({district['adherence']}%)", 
                ha='center', va='center', fontsize=9, fontweight='bold')

ax_sol1.set_xlim(0, 10)
ax_sol1.set_ylim(0, 10)
ax_sol1.set_title(f"{solution_titles[0]}\n{solution_subtitles[0]}", fontsize=16, 
                fontweight='bold', pad=10, color=colors["primary"])

# 2. Together
ax_sol2 = fig.add_subplot(gs[3, 2:4])
ax_sol2.axis('off')

# Create partner ecosystem visualization
center = (5, 5)
ax_sol2.add_patch(plt.Circle(center, 1.5, color=colors["primary"], alpha=0.2))
ax_sol2.text(center[0], center[1], "1FIT\nPLATFORM", ha='center', va='center', 
           fontsize=14, fontweight='bold', color=colors["primary"])

partners = [
    {"name": "Fitness Centers", "angle": 45, "distance": 3},
    {"name": "Wellness Services", "angle": 135, "distance": 3},
    {"name": "Trainers", "angle": 225, "distance": 3},
    {"name": "Users", "angle": 315, "distance": 3}
]

for partner in partners:
    angle_rad = np.deg2rad(partner["angle"])
    x = center[0] + partner["distance"] * np.cos(angle_rad)
    y = center[1] + partner["distance"] * np.sin(angle_rad)
    
    ax_sol2.add_patch(plt.Circle((x, y), 1, color=colors["secondary"], alpha=0.2))
    ax_sol2.text(x, y, partner["name"], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add connecting line
    ax_sol2.plot([center[0], x], [center[1], y], '--', color=colors["secondary"], alpha=0.5)

ax_sol2.set_xlim(0, 10)
ax_sol2.set_ylim(0, 10)
ax_sol2.set_title(f"{solution_titles[1]}\n{solution_subtitles[1]}", fontsize=16, 
                fontweight='bold', pad=10, color=colors["primary"])

# 3. Acceleration
ax_sol3 = fig.add_subplot(gs[3, 4:6])
ax_sol3.axis('off')

# Create mini roadmap
road_y = 5
road_color = '#DDDDDD'
ax_sol3.plot([1, 9], [road_y, road_y], '-', color=road_color, linewidth=10, solid_capstyle='round')

milestones = [
    {"name": "Simplified\nBooking", "x": 2, "y": road_y},
    {"name": "Personalized\nRecommendations", "x": 5, "y": road_y},
    {"name": "Quality\nStandardization", "x": 8, "y": road_y}
]

for i, milestone in enumerate(milestones):
    # Add milestone marker
    marker_color = colors["primary"] if i == 0 else colors["secondary"] if i == 1 else colors["tertiary"]
    ax_sol3.add_patch(plt.Circle((milestone["x"], milestone["y"]), 0.8, color=marker_color, alpha=0.2))
    ax_sol3.add_patch(plt.Circle((milestone["x"], milestone["y"]), 0.8, fill=False, edgecolor=marker_color))
    
    # Add text
    ax_sol3.text(milestone["x"], milestone["y"], milestone["name"], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Add connecting arrow to next milestone if not last
    if i < len(milestones) - 1:
        next_x = milestones[i+1]["x"]
        ax_sol3.annotate('', xy=(next_x - 0.8, road_y), xytext=(milestone["x"] + 0.8, road_y),
                     arrowprops=dict(arrowstyle='->', color=marker_color, lw=2))

ax_sol3.set_xlim(0, 10)
ax_sol3.set_ylim(0, 10)
ax_sol3.set_title(f"{solution_titles[2]}\n{solution_subtitles[2]}", fontsize=16, 
                fontweight='bold', pad=10, color=colors["primary"])

# ----- IMPACT METRICS SECTION -----
ax_metrics_header = fig.add_subplot(gs[4, :])
ax_metrics_header.axis('off')

# Add impact metrics header
ax_metrics_header.text(0.03, 0.95, "Impact Metrics", fontsize=24, fontweight='bold',
                     color=colors["primary"])

# Draw horizontal line
ax_metrics_header.axhline(y=0.9, xmin=0.03, xmax=0.97, color=colors["primary"], linewidth=2, alpha=0.5)

# Create metrics visualization
metrics = [
    {
        "value": "25%",
        "title": "Increase",
        "subtitle": "Trainer Income Optimization",
        "timeline": "by 2027",
        "icon": "📈",
        "color": colors["tertiary"]
    },
    {
        "value": "33%",
        "title": "Reduction",
        "subtitle": "in Wellness Affordability Gap",
        "timeline": "by 2026",
        "icon": "💰",
        "color": colors["highlight"]
    },
    {
        "value": "150%",
        "title": "Growth",
        "subtitle": "in Annual Revenue",
        "timeline": "by 2027",
        "icon": "🚀",
        "color": colors["primary"]
    }
]

for i, metric in enumerate(metrics):
    col_start = i*2
    ax_metric = fig.add_subplot(gs[4, col_start:col_start+2])
    ax_metric.axis('off')
    
    # Create background shape
    bg_height = 0.8
    bg_y = 0.1
    background = patches.FancyBboxPatch((0.1, bg_y), 0.8, bg_height, 
                                      boxstyle=patches.BoxStyle.Round(pad=0.02), 
                                      facecolor=metric["color"], alpha=0.1,
                                      transform=ax_metric.transAxes)
    ax_metric.add_patch(background)
    
    # Add large percentage value
    value_text = ax_metric.text(0.5, 0.7, metric["value"], fontsize=42, fontweight='bold',
                              color=metric["color"], ha='center', transform=ax_metric.transAxes)
    value_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Add icon
    ax_metric.text(0.85, 0.75, metric["icon"], fontsize=28, ha='center', va='center', 
                 transform=ax_metric.transAxes)
    
    # Add title
    ax_metric.text(0.5, 0.45, metric["title"], fontsize=18, fontweight='bold',
                 color=colors["text"], ha='center', transform=ax_metric.transAxes)
    
    # Add subtitle
    ax_metric.text(0.5, 0.3, metric["subtitle"], fontsize=14,
                 color=colors["text"], ha='center', transform=ax_metric.transAxes)
    
    # Add timeline
    ax_metric.text(0.5, 0.15, metric["timeline"], fontsize=12,
                 color=colors["muted"], ha='center', transform=ax_metric.transAxes)

# Add footer text
plt.figtext(0.5, 0.02, 
          "This strategy directly addresses consumer pain points while creating a sustainable marketplace that\n" +
          "benefits users, wellness providers, and trainers across Hong Kong's diverse districts.",
          ha="center", fontsize=12, style='italic', color=colors["text"])

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
plt.savefig('1fit_executive_summary_visual.png', dpi=300, bbox_inches='tight', facecolor=colors["background"])
plt.show()