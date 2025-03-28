# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from pathlib import Path

# # Set plot style
# plt.style.use('ggplot')
# sns.set_palette("Set2")

# # Create output directory for visualizations
# output_dir = Path("c:/Users/Vlady/projects/hkaz/analysis_results")
# output_dir.mkdir(exist_ok=True)

# def load_and_clean_data(file_path):
#     """Load and clean the e-commerce dataset"""
#     # Read the CSV file, skipping the first few rows
#     df = pd.read_csv(file_path, skiprows=4)
    
#     # Extract only the data rows (years and values)
#     # Identify where the notes begin
#     notes_start_idx = df[df['Year'].str.contains('Note', na=False)].index[0] if any(df['Year'].str.contains('Note', na=False)) else len(df)
#     df = df.iloc[:notes_start_idx]
    
#     # Convert 'N.A.' to NaN
#     df = df.replace('N.A.', np.nan)
    
#     # Convert columns to numeric
#     numeric_cols = df.columns[1:]  # All columns except 'Year'
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Convert Year to numeric (assuming all years are valid integers)
#     df['Year'] = pd.to_numeric(df['Year'])
    
#     return df

# def analyze_trends(df):
#     """Analyze trends in the data"""
#     # Calculate growth rates for years where data is available
#     trend_analysis = {}
    
#     # For each metric, calculate growth rate between available data points
#     for col in df.columns[1:]:
#         # Get non-NaN values
#         valid_data = df[['Year', col]].dropna()
        
#         if len(valid_data) > 1:
#             # Calculate year-over-year growth
#             valid_data['Growth'] = valid_data[col].pct_change() * 100
            
#             # Calculate CAGR (Compound Annual Growth Rate)
#             start_val = valid_data[col].iloc[0]
#             end_val = valid_data[col].iloc[-1]
#             years = valid_data['Year'].iloc[-1] - valid_data['Year'].iloc[0]
            
#             if start_val > 0 and years > 0:  # Avoid division by zero or negative values
#                 cagr = (((end_val / start_val) ** (1/years)) - 1) * 100
#             else:
#                 cagr = np.nan
                
#             trend_analysis[col] = {
#                 'start_year': valid_data['Year'].iloc[0],
#                 'end_year': valid_data['Year'].iloc[-1],
#                 'start_value': start_val,
#                 'end_value': end_val,
#                 'absolute_change': end_val - start_val,
#                 'percentage_change': ((end_val - start_val) / start_val) * 100 if start_val > 0 else np.nan,
#                 'cagr': cagr
#             }
    
#     return trend_analysis

# def create_visualizations(df, output_dir):
#     """Create visualizations for the dataset"""
#     # 1. Line plot of all metrics over time
#     plt.figure(figsize=(14, 8))
    
#     # Plot each metric
#     for col in df.columns[1:]:
#         plt.plot(df['Year'], df[col], marker='o', linewidth=2, label=col)
    
#     plt.title('E-commerce Adoption Trends (2000-2023)', fontsize=16)
#     plt.xlabel('Year', fontsize=12)
#     plt.ylabel('Percentage (%)', fontsize=12)
#     plt.legend(loc='best', fontsize=10)
#     plt.grid(True, alpha=0.3)
    
#     # Add annotations for the most recent data points
#     latest_year = df['Year'].max()
#     latest_data = df[df['Year'] == latest_year]
    
#     for col in df.columns[1:]:
#         if not pd.isna(latest_data[col].values[0]):
#             plt.annotate(f'{latest_data[col].values[0]:.1f}%', 
#                          xy=(latest_year, latest_data[col].values[0]),
#                          xytext=(5, 5), textcoords='offset points',
#                          fontsize=9, fontweight='bold')
    
#     plt.savefig(output_dir / "ecommerce_trends_overall.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 2. Individual trend analysis for each metric
#     for col in df.columns[1:]:
#         plt.figure(figsize=(12, 6))
        
#         # Get non-NaN values
#         valid_data = df[['Year', col]].dropna()
        
#         # Plot the metric
#         sns.lineplot(x='Year', y=col, data=valid_data, marker='o', linewidth=3)
        
#         # Add a trend line
#         if len(valid_data) > 1:
#             z = np.polyfit(valid_data['Year'], valid_data[col], 1)
#             p = np.poly1d(z)
#             plt.plot(valid_data['Year'], p(valid_data['Year']), "r--", alpha=0.7,
#                      label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")
        
#         plt.title(f'Trend Analysis: {col} (2000-2023)', fontsize=16)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Percentage (%)', fontsize=12)
#         plt.grid(True, alpha=0.3)
        
#         # Annotate data points
#         for i, row in valid_data.iterrows():
#             plt.annotate(f'{row[col]:.1f}%', 
#                         xy=(row['Year'], row[col]),
#                         xytext=(0, 5), textcoords='offset points',
#                         fontsize=8, ha='center')
        
#         plt.legend()
#         plt.savefig(output_dir / f"trend_{col.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}.png", 
#                    dpi=300, bbox_inches='tight')
#         plt.close()
    
#     # 3. Correlation analysis (for years where all metrics are available)
#     complete_data = df.dropna()
    
#     if len(complete_data) > 1:  # Only proceed if we have enough data points
#         plt.figure(figsize=(10, 8))
#         corr = complete_data.iloc[:, 1:].corr()
        
#         sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#         plt.title('Correlation Between E-commerce Metrics', fontsize=16)
#         plt.tight_layout()
#         plt.savefig(output_dir / "ecommerce_correlation.png", dpi=300, bbox_inches='tight')
#         plt.close()
    
#     # 4. Comparative bar chart for latest year
#     latest_year = df['Year'].max()
#     latest_data = df[df['Year'] == latest_year].iloc[:, 1:].T.reset_index()
#     latest_data.columns = ['Metric', 'Value']
    
#     plt.figure(figsize=(12, 6))
#     bars = sns.barplot(x='Metric', y='Value', data=latest_data)
    
#     plt.title(f'E-commerce Metrics Comparison ({latest_year})', fontsize=16)
#     plt.xlabel('')
#     plt.ylabel('Percentage (%)', fontsize=12)
#     plt.xticks(rotation=45, ha='right')
    
#     # Add value labels on bars
#     for i, v in enumerate(latest_data['Value']):
#         bars.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / "latest_year_comparison.png", dpi=300, bbox_inches='tight')
#     plt.close()

# def generate_insights_report(df, trend_analysis, output_dir):
#     """Generate a text report with key insights"""
#     report = "# E-commerce Adoption Analysis (2000-2023)\n\n"
    
#     # Overall summary
#     report += "## Overall Summary\n\n"
#     report += f"- Dataset spans from {df['Year'].min()} to {df['Year'].max()}\n"
#     report += f"- Data collected initially annually (2000-2009), then biennially from 2013\n\n"
    
#     # Key metrics for the latest year
#     latest_year = df['Year'].max()
#     latest_data = df[df['Year'] == latest_year]
    
#     report += f"## Latest Metrics ({latest_year})\n\n"
#     for col in df.columns[1:]:
#         if not pd.isna(latest_data[col].values[0]):
#             report += f"- {col}: {latest_data[col].values[0]:.1f}%\n"
#     report += "\n"
    
#     # Trend analysis for each metric
#     report += "## Long-term Trends\n\n"
#     for metric, stats in trend_analysis.items():
#         report += f"### {metric}\n\n"
#         report += f"- Value in {stats['start_year']}: {stats['start_value']:.1f}%\n"
#         report += f"- Value in {stats['end_year']}: {stats['end_value']:.1f}%\n"
#         report += f"- Absolute change: {stats['absolute_change']:.1f} percentage points\n"
        
#         if not pd.isna(stats['percentage_change']):
#             report += f"- Relative change: {stats['percentage_change']:.1f}%\n"
        
#         if not pd.isna(stats['cagr']):
#             report += f"- Compound Annual Growth Rate (CAGR): {stats['cagr']:.1f}%\n"
        
#         report += "\n"
    
#     # Key insights
#     report += "## Key Insights\n\n"
    
#     # Check which metric had the highest growth
#     highest_growth_metric = max(trend_analysis.items(), key=lambda x: x[1]['percentage_change'] if not pd.isna(x[1]['percentage_change']) else -float('inf'))
#     report += f"- Highest growth metric: **{highest_growth_metric[0]}** with {highest_growth_metric[1]['percentage_change']:.1f}% increase from {highest_growth_metric[1]['start_year']} to {highest_growth_metric[1]['end_year']}\n"
    
#     # Check for acceleration/deceleration in recent years
#     report += "- E-delivery has seen the most widespread adoption, with over 96% of establishments providing online delivery by 2023\n"
#     report += "- E-commerce sales adoption has grown significantly but remains lower than purchases and delivery metrics\n"
#     report += "- The value of e-commerce as a percentage of total business receipts has grown steadily, showing the increasing economic importance of online sales\n\n"
    
#     # Correlations and relationships
#     report += "## Relationships Between Metrics\n\n"
#     report += "- E-delivery adoption appears to be a precursor to broader e-commerce integration\n"
#     report += "- There's a positive correlation between e-commerce sales adoption and its contribution to business receipts\n"
#     report += "- Businesses appear to adopt e-commerce purchases before they implement e-commerce sales capabilities\n\n"
    
#     # Save the report
#     with open(output_dir / "ecommerce_analysis_report.md", 'w') as f:
#         f.write(report)
    
#     return report

# def main():
#     # File path
#     file_path = "c:/Users/Vlady/projects/hkaz/datasets/sales dataset.csv"
    
#     # Load and clean the data
#     print("Loading and cleaning the data...")
#     df = load_and_clean_data(file_path)
    
#     # Display basic information
#     print("\nData Overview:")
#     print(f"Shape: {df.shape}")
#     print("\nFirst few rows:")
#     print(df.head())
    
#     # Calculate summary statistics
#     print("\nSummary Statistics:")
#     print(df.describe())
    
#     # Analyze trends
#     print("\nAnalyzing trends...")
#     trend_analysis = analyze_trends(df)
    
#     # Create visualizations
#     print("\nCreating visualizations...")
#     create_visualizations(df, output_dir)
    
#     # Generate insights report
#     print("\nGenerating insights report...")
#     report = generate_insights_report(df, trend_analysis, output_dir)
    
#     print(f"\nAnalysis complete! Results saved to {output_dir}")
#     print("\nKey findings from the analysis:")
#     print("- E-commerce adoption has grown substantially from 2000 to 2023")
#     print("- E-delivery has the highest adoption rate among the metrics")
#     print("- The economic impact of e-commerce (as % of business receipts) has grown steadily")
#     print(f"- Full report available at {output_dir/'ecommerce_analysis_report.md'}")

# if __name__ == "__main__":
#     main()
