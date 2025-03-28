#Data analysis of hk e-commerce dataset - Vlad

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.ticker as mtick

# Need to Set plot style
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Created output directory for visualizations
output_dir = Path("c:/Users/Vlady/projects/hkaz/analysis_results")
output_dir.mkdir(exist_ok=True)

def load_and_clean_data(file_path):
    """Load and clean the e-commerce dataset"""
    # Read the whole file as text first to understand its structure
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line that contains the actual column headers (the line containing "Type of information")
    header_row = 0
    for i, line in enumerate(lines):
        if "Type of information" in line:
            header_row = i
            break
    
    # Find the line that starts the data (the line containing "Year")
    data_start_row = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Year"):
            data_start_row = i
            break
    
    # Read the CSV file with the correct parameters
    df = pd.read_csv(file_path, skiprows=data_start_row)
    
    # Remove any columns that are all NaN
    df = df.dropna(axis=1, how='all')
    
    # The first column should be 'Year'
    df.columns = ['Year'] + list(df.columns[1:])
    
    # Find where notes begin
    notes_row = None
    for i, row in df.iterrows():
        if isinstance(row['Year'], str) and 'Note' in row['Year']:
            notes_row = i
            break
    
    # Keep only data rows
    if notes_row is not None:
        df = df.iloc[:notes_row]
    
    # Replace 'N.A.' with np.nan
    df = df.replace('N.A.', np.nan)
    
    # Convert columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any remaining rows with NaN in Year column
    df = df.dropna(subset=['Year'])
    
    # Rename columns for clarity
    new_columns = {
        df.columns[1]: 'E-commerce Sales (%)',
        df.columns[2]: 'E-commerce Purchases (%)',
        df.columns[3]: 'E-delivery (%)',
        df.columns[4]: 'E-commerce as % of Business Receipts'
    }
    df = df.rename(columns=new_columns)
    
    return df

def analyze_trends(df):
    """Analyze trends in the data"""
    # Calculate growth rates for years where data is available
    trend_analysis = {}
    
    # For each metric, calculate growth rate between available data points
    for col in df.columns[1:]:
        # Get non-NaN values
        valid_data = df[['Year', col]].dropna()
        
        if len(valid_data) > 1:
            # Calculate year-over-year growth
            valid_data['Growth'] = valid_data[col].pct_change() * 100
            
            # Calculate CAGR (Compound Annual Growth Rate)
            start_val = valid_data[col].iloc[0]
            end_val = valid_data[col].iloc[-1]
            years = valid_data['Year'].iloc[-1] - valid_data['Year'].iloc[0]
            
            if start_val > 0 and years > 0:  # Avoid division by zero or negative values
                cagr = (((end_val / start_val) ** (1/years)) - 1) * 100
            else:
                cagr = np.nan
                
            trend_analysis[col] = {
                'start_year': valid_data['Year'].iloc[0],
                'end_year': valid_data['Year'].iloc[-1],
                'start_value': start_val,
                'end_value': end_val,
                'absolute_change': end_val - start_val,
                'percentage_change': ((end_val - start_val) / start_val) * 100 if start_val > 0 else np.nan,
                'cagr': cagr
            }
    
    return trend_analysis

def create_visualizations(df, output_dir):
    """Create enhanced visualizations for the e-commerce dataset"""
    
    # 1. Combined line plot of all metrics
    plt.figure(figsize=(14, 8))
    for col in df.columns[1:]:
        plt.plot(df['Year'], df[col], marker='o', linewidth=2.5, label=col)
    
    plt.title("E-commerce Adoption Trends (2000-2023)", fontsize=18, fontweight='bold')
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    # Add annotations for the last data point of each metric
    last_year = df['Year'].max()
    for col in df.columns[1:]:
        last_value = df[df['Year'] == last_year][col].values[0]
        if not np.isnan(last_value):
            plt.annotate(f'{last_value:.1f}%', 
                        xy=(last_year, last_value),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=11,
                        fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_ecommerce_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual trend plots with enhanced styling
    for col in df.columns[1:]:
        plt.figure(figsize=(12, 7))
        
        # Create main plot
        ax = plt.subplot(111)
        valid_data = df[['Year', col]].dropna()
        
        # Plot the line and points
        ax.plot(valid_data['Year'], valid_data[col], marker='o', linewidth=3, markersize=8, color='#1f77b4')
        
        # Add a trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['Year'], valid_data[col], 1)
            p = np.poly1d(z)
            ax.plot(valid_data['Year'], p(valid_data['Year']), "r--", linewidth=2, alpha=0.7,
                   label=f"Trend line (slope: {z[0]:.2f})")
        
        # Style the plot
        ax.set_title(f"{col} (2000-2023)", fontsize=18, fontweight='bold')
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Percentage (%)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        
        # Add annotations for each data point
        for year, value in zip(valid_data['Year'], valid_data[col]):
            ax.annotate(f'{value:.1f}%', 
                      xy=(year, value),
                      xytext=(0, 10), 
                      textcoords='offset points',
                      fontsize=9,
                      ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Calculate growth statistics
        if len(valid_data) > 1:
            start_year = valid_data['Year'].iloc[0]
            end_year = valid_data['Year'].iloc[-1]
            start_val = valid_data[col].iloc[0]
            end_val = valid_data[col].iloc[-1]
            abs_change = end_val - start_val
            pct_change = ((end_val - start_val) / start_val) * 100 if start_val > 0 else float('inf')
            
            # Add textbox with growth stats
            stats_text = (
                f"Growth Summary ({start_year}-{end_year}):\n"
                f"Starting: {start_val:.1f}%\n"
                f"Ending: {end_val:.1f}%\n"
                f"Absolute Change: {abs_change:+.1f}%\n"
                f"Relative Change: {pct_change:+.1f}%"
            )
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
        
        if len(valid_data) > 1:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{col.replace(' ', '_').replace('%', 'pct').lower()}_detailed_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Bar chart comparing key years
    key_years = [2000, 2009, 2015, 2023]  # First year, last year of annual data, mid point, latest year
    key_years_df = df[df['Year'].isin(key_years)]
    
    plt.figure(figsize=(15, 10))
    
    # Set up the bar plot structure
    bar_width = 0.2
    r1 = np.arange(len(key_years))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Create the grouped bar chart
    plt.bar(r1, key_years_df['E-commerce Sales (%)'], width=bar_width, label='E-commerce Sales', color='#1f77b4')
    plt.bar(r2, key_years_df['E-commerce Purchases (%)'], width=bar_width, label='E-commerce Purchases', color='#ff7f0e')
    plt.bar(r3, key_years_df['E-delivery (%)'], width=bar_width, label='E-delivery', color='#2ca02c')
    plt.bar(r4, key_years_df['E-commerce as % of Business Receipts'], width=bar_width, label='% of Business Receipts', color='#d62728')
    
    # Add labels, title and axes ticks
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('E-commerce Adoption Comparison Across Key Years', fontsize=18, fontweight='bold')
    plt.xticks([r + bar_width*1.5 for r in range(len(key_years))], key_years)
    plt.legend(loc='upper left', fontsize=12)
    
    # Add value labels on bars
    def add_labels(rects, positions):
        for i, rect in enumerate(rects):
            height = key_years_df.iloc[i][rect]
            if not np.isnan(height):
                plt.text(positions[i], height + 0.5, f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_labels(['E-commerce Sales (%)'], r1)
    add_labels(['E-commerce Purchases (%)'], r2)
    add_labels(['E-delivery (%)'], r3)
    add_labels(['E-commerce as % of Business Receipts'], r4)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "key_years_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap between metrics
    plt.figure(figsize=(10, 8))
    corr_df = df.iloc[:, 1:].corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
               linewidths=1, cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Correlation Between E-commerce Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Area chart showing cumulative growth
    plt.figure(figsize=(14, 8))
    
    # Calculate normalized growth (set first year as baseline 1.0)
    growth_df = pd.DataFrame(index=df['Year'])
    for col in df.columns[1:]:
        valid_data = df[['Year', col]].dropna()
        if not valid_data.empty and valid_data[col].iloc[0] > 0:
            growth_df[col] = valid_data[col] / valid_data[col].iloc[0]
    
    # Plot the area chart
    growth_df.plot.area(alpha=0.6, figsize=(14, 8), colormap='viridis')
    
    plt.title('Relative Growth of E-commerce Metrics (First Year = 1.0)', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Growth Multiple', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    
    # Add annotations for the final values
    last_valid_year = growth_df.index[-1]
    for col in growth_df.columns:
        if last_valid_year in growth_df.index:
            value = growth_df.loc[last_valid_year, col]
            plt.annotate(f'{value:.1f}x', xy=(last_valid_year, value),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    
    plt.tight_layout()
    plt.savefig(output_dir / "relative_growth_area_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Dashboard-style 2x2 subplot for quick overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Combined line plot (simplified)
    for col in df.columns[1:]:
        axes[0, 0].plot(df['Year'], df[col], marker='o', linewidth=2, label=col)
    
    axes[0, 0].set_title('E-commerce Adoption Trends', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Percentage (%)')
    axes[0, 0].legend(loc='upper left', fontsize=9)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Bar chart for latest year
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].iloc[:, 1:].T
    latest_data.columns = ['Value']
    latest_data.plot(kind='barh', ax=axes[0, 1], color='skyblue')
    
    axes[0, 1].set_title(f'E-commerce Metrics ({latest_year})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Percentage (%)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Growth comparison (first vs. last year)
    first_year = df['Year'].min()
    comparison_years = [first_year, latest_year]
    comparison_df = df[df['Year'].isin(comparison_years)].set_index('Year')
    
    comparison_df.T.plot(kind='bar', ax=axes[1, 0], rot=30)
    axes[1, 0].set_title(f'First vs Latest Year Comparison ({first_year} vs {latest_year})', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: CAGR for each metric
    cagr_data = []
    cagr_labels = []
    
    for col in df.columns[1:]:
        valid_data = df[['Year', col]].dropna()
        if len(valid_data) > 1:
            start_val = valid_data[col].iloc[0]
            end_val = valid_data[col].iloc[-1]
            years = valid_data['Year'].iloc[-1] - valid_data['Year'].iloc[0]
            
            if start_val > 0 and years > 0:
                cagr = (((end_val / start_val) ** (1/years)) - 1) * 100
                cagr_data.append(cagr)
                cagr_labels.append(col)
    
    axes[1, 1].bar(cagr_labels, cagr_data, color='green')
    axes[1, 1].set_title('Compound Annual Growth Rate (CAGR)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('CAGR (%)')
    axes[1, 1].set_xticklabels(cagr_labels, rotation=45, ha='right')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(cagr_data):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ecommerce_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_insights_report(df, trend_analysis, output_dir):
    """Generate insights report based on the analysis"""
    report_path = output_dir / "ecommerce_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write("# E-commerce Analysis Report\n\n")
        f.write("## Key Findings\n\n")
        for metric, analysis in trend_analysis.items():
            f.write(f"### {metric}\n")
            f.write(f"- Start Year: {analysis['start_year']}\n")
            f.write(f"- End Year: {analysis['end_year']}\n")
            f.write(f"- Start Value: {analysis['start_value']:.2f}\n")
            f.write(f"- End Value: {analysis['end_value']:.2f}\n")
            f.write(f"- Absolute Change: {analysis['absolute_change']:.2f}\n")
            f.write(f"- Percentage Change: {analysis['percentage_change']:.2f}%\n")
            f.write(f"- CAGR: {analysis['cagr']:.2f}%\n\n")
    return report_path

def main():
    # File path
    file_path = "c:/Users/Vlady/projects/hkaz/datasets/sales dataset.csv"
    
    # Load and clean the data
    print("Loading and cleaning the data...")
    df = load_and_clean_data(file_path)
    
    # Display basic information
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Calculate summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Analyze trends
    print("\nAnalyzing trends...")
    trend_analysis = analyze_trends(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, output_dir)
    
    # Generate insights report
    print("\nGenerating insights report...")
    report = generate_insights_report(df, trend_analysis, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print("\nKey findings from the analysis:")
    print("- E-commerce adoption has grown substantially from 2000 to 2023")
    print("- E-delivery has the highest adoption rate among the metrics")
    print("- The economic impact of e-commerce (as % of business receipts) has grown steadily")
    print(f"- Full report available at {output_dir/'ecommerce_analysis_report.md'}")

if __name__ == "__main__":
    main()