#!/usr/bin/env python3
"""
GA Performance Analyzer - Visualize evolution of time and fitness across compressors

This tool analyzes GA results and creates plots showing:
1. Time per generation evolution for each compressor
2. Best fitness evolution for each compressor
3. Comparative analysis across compressors

Usage:
    python tools/ga_performance_analyzer.py --results_dir ga_results_*/
    python tools/ga_performance_analyzer.py --input_pattern "*_ga_results"
"""

import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for styling
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
# sns.set_palette("husl")  # Skip seaborn palette


class GAPerformanceAnalyzer:
    """Analyze and visualize GA performance across different compressors."""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results_data = {}
        
    def find_ga_results_directories(self, pattern: str = "*ga_results*") -> List[Path]:
        """Find all GA results directories matching the pattern."""
        directories = []
        
        # Look in current directory
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                directories.append(Path(path))
        
        # Also check common locations
        common_locations = [".", "ga_results_*", "../ga_results_*"]
        for location_pattern in common_locations:
            for path in glob.glob(location_pattern):
                if os.path.isdir(path) and "ga_results" in path:
                    directories.append(Path(path))
        
        # Remove duplicates and sort
        directories = sorted(list(set(directories)))
        logger.info(f"Found {len(directories)} GA results directories")
        
        return directories
    
    def extract_compressor_name(self, dir_name: str) -> str:
        """Extract compressor name from directory name."""
        # Common patterns: ga_results_zstd, zstd_results, ga_zstd_results, etc.
        compressor_patterns = [
            r'ga_results_(\w+)',
            r'(\w+)_ga_results',
            r'ga_(\w+)_results',
            r'results_(\w+)',
            r'(\w+)_results'
        ]
        
        for pattern in compressor_patterns:
            match = re.search(pattern, dir_name.lower())
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, try to find known compressor names
        known_compressors = ['zstd', 'lzma', 'brotli', 'paq8', 'ac2']
        for comp in known_compressors:
            if comp in dir_name.lower():
                return comp.upper()
        
        # Fallback to directory name
        return dir_name.replace('_', ' ').title()
    
    def parse_generation_csv(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Parse a generation CSV file and extract relevant data."""
        try:
            df = pd.read_csv(csv_file)
            
            # Check if this is generation-level data (fitness history format)
            if 'Generation' in df.columns and 'Best_Fitness' in df.columns:
                # Already in expected generation-level format
                logger.debug(f"  Using generation-level format from {csv_file.name}")
                return df
            elif 'Individual_Name' in df.columns and 'Fitness' in df.columns:
                # Individual-level data - need to convert to generation statistics
                logger.debug(f"  Converting individual-level data from {csv_file.name}")
                return self._convert_individual_to_generation_stats(df)
            else:
                logger.warning(f"Missing required columns in {csv_file}")
                logger.debug(f"  Available columns: {list(df.columns)}")
                return None
            
        except Exception as e:
            logger.warning(f"Error parsing {csv_file}: {e}")
            return None
    
    def _convert_individual_to_generation_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert individual-level data to generation-level statistics (legacy support)."""
        # Extract generation number from Individual_Name (e.g., "Gen1_Ind34" -> 1)
        if 'Generation' not in df.columns:
            df['Generation'] = df['Individual_Name'].str.extract(r'Gen(\d+)_').astype(int)
        
        return self._convert_to_generation_stats(df)
    
    def _convert_to_generation_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert individual-level data to generation-level statistics with timing support."""
        # Find fitness column (could be 'Fitness' or other names)
        fitness_col = None
        for col in df.columns:
            if col.lower() in ['fitness', 'best_fitness', 'score']:
                fitness_col = col
                break
        
        if fitness_col is None:
            raise ValueError(f"No fitness column found. Available columns: {list(df.columns)}")
        
        # Prepare aggregation dictionary
        agg_dict = {
            fitness_col: ['max', 'mean', 'min', 'std', 'count']
        }
        
        # Find compression time column
        time_col = None
        for col in df.columns:
            if col.lower() in ['compression_time', 'time', 'duration'] or 'time' in col.lower():
                time_col = col
                break
        
        
        # Add compression time aggregation if available
        if time_col is not None:
            agg_dict[time_col] = ['mean', 'max', 'min', 'std', 'sum']
        
        # Calculate generation-level statistics
        generation_stats = df.groupby('Generation').agg(agg_dict).reset_index()
        
        # Flatten column names
        if time_col is not None:
            generation_stats.columns = [
                'Generation', 'Best_Fitness', 'Avg_Fitness', 
                'Min_Fitness', 'Std_Fitness', 'Population_Size',
                'Avg_Compression_Time', 'Max_Compression_Time', 
                'Min_Compression_Time', 'Std_Compression_Time', 'Total_Compression_Time'
            ]
            # Use total compression time as generation time (sum of all individual compression times)
            generation_stats['Generation_Time'] = generation_stats['Total_Compression_Time']
        else:
            generation_stats.columns = [
                'Generation', 'Best_Fitness', 'Avg_Fitness', 
                'Min_Fitness', 'Std_Fitness', 'Population_Size'
            ]
            # Add placeholder generation time
            generation_stats['Generation_Time'] = 0.0
            generation_stats['Avg_Compression_Time'] = 0.0
            generation_stats['Max_Compression_Time'] = 0.0
            generation_stats['Min_Compression_Time'] = 0.0
        
        return generation_stats
    
    def _extract_timing_from_individual_files(self, directory: Path) -> Optional[Dict[int, Dict[str, float]]]:
        """Extract timing data from individual generation files."""
        generation_files = list(directory.glob("generation_*.csv"))
        if not generation_files:
            return None
        
        timing_data = {}
        
        for gen_file in generation_files:
            try:
                # Extract generation number
                gen_number = int(gen_file.stem.split('_')[-1])
                
                # Read CSV and look for timing data
                df = pd.read_csv(gen_file)
                if 'Compression_Time' in df.columns:
                    timing_stats = {
                        'avg_time': df['Compression_Time'].mean(),
                        'max_time': df['Compression_Time'].max(),
                        'min_time': df['Compression_Time'].min(),
                        'total_time': df['Compression_Time'].sum(),
                        'std_time': df['Compression_Time'].std()
                    }
                    timing_data[gen_number] = timing_stats
                    
            except Exception as e:
                logger.debug(f"Could not extract timing from {gen_file.name}: {e}")
                continue
        
        return timing_data if timing_data else None
    
    def _merge_timing_data(self, fitness_df: pd.DataFrame, timing_data: Dict[int, Dict[str, float]]) -> pd.DataFrame:
        """Merge timing data into fitness DataFrame."""
        df = fitness_df.copy()
        
        # Add timing columns
        df['Avg_Compression_Time'] = df['Generation'].map(lambda x: timing_data.get(x, {}).get('avg_time', 0.0))
        df['Max_Compression_Time'] = df['Generation'].map(lambda x: timing_data.get(x, {}).get('max_time', 0.0))
        df['Min_Compression_Time'] = df['Generation'].map(lambda x: timing_data.get(x, {}).get('min_time', 0.0))
        df['Total_Compression_Time'] = df['Generation'].map(lambda x: timing_data.get(x, {}).get('total_time', 0.0))
        df['Generation_Time'] = df['Total_Compression_Time']  # Use total compression time as generation time
        
        return df
    
    def load_ga_results(self, directories: List[Path]) -> Dict[str, pd.DataFrame]:
        """Load GA results from all directories."""
        results = {}
        
        for directory in directories:
            compressor = self.extract_compressor_name(directory.name)
            logger.info(f"Processing {directory} as {compressor}")
            
            # First, look for fitness history files (already in generation-level format)
            fitness_history_files = list(directory.glob("*fitness_history*.csv"))
            
            if fitness_history_files:
                logger.info(f"  Found fitness history file: {fitness_history_files[0].name}")
                # Use fitness history file (already in generation-level format)
                df = self.parse_generation_csv(fitness_history_files[0])
                if df is not None:
                    # Try to augment with timing data from individual generation files
                    timing_data = self._extract_timing_from_individual_files(directory)
                    if timing_data:
                        df = self._merge_timing_data(df, timing_data)
                        logger.info(f"  Augmented fitness history with timing data from individual files")
                    else:
                        # Add timing columns with zeros if no timing data available
                        if 'Generation_Time' not in df.columns:
                            df['Generation_Time'] = 0.0
                        if 'Avg_Compression_Time' not in df.columns:
                            df['Avg_Compression_Time'] = 0.0
                            df['Max_Compression_Time'] = 0.0
                            df['Min_Compression_Time'] = 0.0
                            df['Total_Compression_Time'] = 0.0
                    
                    df['Compressor'] = compressor
                    df['Directory'] = str(directory)
                    results[compressor] = df
                    logger.info(f"  Loaded {len(df)} generations for {compressor} from fitness history")
                    continue
            
            # Look for individual-level generation CSV files (generation_1.csv, generation_2.csv, etc.)
            generation_files = list(directory.glob("generation_*.csv"))
            
            if not generation_files:
                logger.warning(f"No generation CSV files found in {directory}")
                logger.debug(f"Available files in {directory}: {[f.name for f in directory.glob('*')]}")
                
                # Fallback to old pattern matching
                csv_files = list(directory.glob("*.csv"))
                logger.debug(f"CSV files found: {[f.name for f in csv_files]}")
                
                generation_files = [f for f in csv_files if any(keyword in f.name.lower() 
                                   for keyword in ['generation', 'fitness', 'results'])]
                
                if not generation_files:
                    logger.warning(f"No generation data files found in {directory}")
                    logger.debug(f"Tried keywords: generation, fitness, results")
                    continue
            
            # Sort generation files by generation number
            try:
                generation_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            except (ValueError, IndexError):
                # Fallback to name sorting if number extraction fails
                generation_files.sort()
            
            logger.info(f"  Found {len(generation_files)} generation files")
            
            # Load and combine all generation files
            all_generation_data = []
            
            for gen_file in generation_files:
                df = self.parse_generation_csv(gen_file)
                if df is not None:
                    # Extract generation number from filename
                    try:
                        gen_number = int(gen_file.stem.split('_')[-1])
                        df['Generation'] = gen_number
                    except (ValueError, IndexError):
                        # Fallback: try to extract from Individual_Name
                        if 'Individual_Name' in df.columns:
                            df['Generation'] = df['Individual_Name'].str.extract(r'Gen(\d+)_').astype(int)
                    
                    all_generation_data.append(df)
            
            if all_generation_data:
                # Combine all generations into one DataFrame
                combined_df = pd.concat(all_generation_data, ignore_index=True)
                
                # Convert to generation-level statistics
                generation_stats = self._convert_to_generation_stats(combined_df)
                
                generation_stats['Compressor'] = compressor
                generation_stats['Directory'] = str(directory)
                results[compressor] = generation_stats
                logger.info(f"  Loaded {len(generation_stats)} generations for {compressor}")
            else:
                logger.warning(f"No valid generation data found in {directory}")
        
        return results
    
    def create_fitness_evolution_plot(self, results: Dict[str, pd.DataFrame]) -> Path:
        """Create plot showing fitness evolution for each compressor."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, (compressor, df) in enumerate(results.items()):
            ax.plot(df['Generation'], df['Best_Fitness'], 
                   label=compressor, linewidth=2, color=colors[i],
                   marker='o', markersize=3, alpha=0.8)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Best Fitness (Compression Ratio)', fontsize=12)
        ax.set_title('Fitness Evolution Across Compressors', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "fitness_evolution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created fitness evolution plot: {output_file}")
        return output_file
    
    def create_time_evolution_plot(self, results: Dict[str, pd.DataFrame]) -> Path:
        """Create plot showing compression time evolution for each compressor."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        # Plot 1: Total compression time per generation
        for i, (compressor, df) in enumerate(results.items()):
            if 'Generation_Time' in df.columns and df['Generation_Time'].sum() > 0:
                ax1.plot(df['Generation'], df['Generation_Time'], 
                        label=f"{compressor} (Total)", linewidth=2, color=colors[i],
                        marker='s', markersize=3, alpha=0.8)
            elif 'Total_Compression_Time' in df.columns:
                ax1.plot(df['Generation'], df['Total_Compression_Time'], 
                        label=f"{compressor} (Total)", linewidth=2, color=colors[i],
                        marker='s', markersize=3, alpha=0.8)
            else:
                logger.warning(f"No total time data found for {compressor}")
        
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Total Compression Time (seconds)', fontsize=12)
        ax1.set_title('Total Compression Time per Generation', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average compression time per individual
        for i, (compressor, df) in enumerate(results.items()):
            if 'Avg_Compression_Time' in df.columns and df['Avg_Compression_Time'].sum() > 0:
                ax2.plot(df['Generation'], df['Avg_Compression_Time'], 
                        label=f"{compressor} (Avg per individual)", linewidth=2, color=colors[i],
                        marker='o', markersize=3, alpha=0.8, linestyle='--')
            else:
                logger.warning(f"No average time data found for {compressor}")
        
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Average Compression Time per Individual (seconds)', fontsize=12)
        ax2.set_title('Average Individual Compression Time Evolution', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        # Add some styling
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "time_evolution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created time evolution plot: {output_file}")
        return output_file
    
    def create_comparative_analysis_plot(self, results: Dict[str, pd.DataFrame]) -> Path:
        """Create combined plot showing both fitness and time trends."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        # Top plot: Fitness evolution
        for i, (compressor, df) in enumerate(results.items()):
            ax1.plot(df['Generation'], df['Best_Fitness'], 
                    label=compressor, linewidth=2.5, color=colors[i],
                    marker='o', markersize=4, alpha=0.9)
        
        ax1.set_ylabel('Best Fitness (Compression Ratio)', fontsize=12)
        ax1.set_title('Comparative GA Performance Analysis', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Bottom plot: Time evolution (if available)
        time_plotted = False
        for i, (compressor, df) in enumerate(results.items()):
            time_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'duration', 'elapsed']):
                    time_col = col
                    break
            
            if time_col is not None:
                time_data = pd.to_numeric(df[time_col], errors='coerce')
                if not time_data.isna().all():
                    ax2.plot(df['Generation'], time_data, 
                           label=compressor, linewidth=2.5, color=colors[i],
                           marker='s', markersize=4, alpha=0.9)
                    time_plotted = True
        
        if time_plotted:
            ax2.set_xlabel('Generation', fontsize=12)
            ax2.set_ylabel('Time per Generation (seconds)', fontsize=12)
            ax2.legend(loc='best', frameon=True, shadow=True)
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        else:
            ax2.text(0.5, 0.5, 'Time data not available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=14, style='italic', alpha=0.6)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        
        output_file = self.output_dir / "comparative_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created comparative analysis plot: {output_file}")
        return output_file
    
    def create_summary_statistics(self, results: Dict[str, pd.DataFrame]) -> Path:
        """Create summary statistics table and save as CSV."""
        summary_stats = []
        
        for compressor, df in results.items():
            stats = {
                'Compressor': compressor,
                'Total_Generations': len(df),
                'Final_Best_Fitness': df['Best_Fitness'].iloc[-1] if len(df) > 0 else None,
                'Max_Fitness_Achieved': df['Best_Fitness'].max(),
                'Fitness_Improvement': (df['Best_Fitness'].iloc[-1] - df['Best_Fitness'].iloc[0]) if len(df) > 1 else 0,
                'Fitness_Std': df['Best_Fitness'].std(),
                'Convergence_Generation': None  # Generation where fitness stopped improving significantly
            }
            
            # Find convergence point (where improvement becomes minimal)
            if len(df) > 10:
                fitness_diff = df['Best_Fitness'].diff().rolling(window=5).mean()
                convergence_threshold = 0.01  # 1% improvement threshold
                convergence_idx = fitness_diff[fitness_diff < convergence_threshold].first_valid_index()
                if convergence_idx is not None:
                    stats['Convergence_Generation'] = df.loc[convergence_idx, 'Generation']
            
            # Time statistics (if available)
            time_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'duration', 'elapsed']):
                    time_col = col
                    break
            
            if time_col is not None:
                time_data = pd.to_numeric(df[time_col], errors='coerce')
                stats.update({
                    'Avg_Time_Per_Generation': time_data.mean(),
                    'Total_Runtime': time_data.sum(),
                    'Min_Time_Per_Generation': time_data.min(),
                    'Max_Time_Per_Generation': time_data.max()
                })
            
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        output_file = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(output_file, index=False, float_format='%.4f')
        
        logger.info(f"Created summary statistics: {output_file}")
        return output_file
    
    def generate_report(self, results: Dict[str, pd.DataFrame]) -> Path:
        """Generate a comprehensive markdown report."""
        report_content = f"""# GA Performance Analysis Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report analyzes the performance evolution of genetic algorithm optimization across different compressors.

### Compressors Analyzed
{chr(10).join([f"- **{comp}**: {len(df)} generations" for comp, df in results.items()])}

## Key Findings

### Final Performance Comparison
"""
        
        # Add performance table
        for compressor, df in results.items():
            final_fitness = df['Best_Fitness'].iloc[-1] if len(df) > 0 else 'N/A'
            max_fitness = df['Best_Fitness'].max() if len(df) > 0 else 'N/A'
            report_content += f"""
#### {compressor}
- Final Best Fitness: {final_fitness:.4f}
- Maximum Fitness Achieved: {max_fitness:.4f}
- Total Generations: {len(df)}
"""
        
        report_content += """
## Visualizations Generated

1. **fitness_evolution.png**: Shows how the best fitness evolves over generations for each compressor
2. **time_evolution.png**: Shows how time per generation evolves for each compressor
3. **comparative_analysis.png**: Combined view of fitness and time evolution
4. **summary_statistics.csv**: Detailed numerical summary of all metrics

## Usage

These plots help identify:
- Which compressor achieves the best final compression ratio
- How quickly each compressor converges to optimal parameters
- Computational efficiency trade-offs between compressors
- Convergence patterns and potential early stopping points

## Recommendations

Based on the analysis, consider:
1. Using the compressor with the highest final fitness for maximum compression
2. Balancing fitness gains against computational time requirements
3. Implementing early stopping if convergence is detected early
4. Adjusting population size or generations based on convergence patterns
"""
        
        report_file = self.output_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated comprehensive report: {report_file}")
        return report_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze GA performance across compressors')
    parser.add_argument('--input_pattern', '-p', 
                       default='*ga_results*',
                       help='Pattern to match GA results directories')
    parser.add_argument('--output_dir', '-o',
                       default='analysis_results',
                       help='Output directory for plots and analysis')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip plot generation (only create statistics)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GAPerformanceAnalyzer(args.output_dir)
    
    # Find and load GA results
    logger.info("Searching for GA results directories...")
    directories = analyzer.find_ga_results_directories(args.input_pattern)
    
    if not directories:
        logger.error("No GA results directories found!")
        logger.error(f"Searched pattern: {args.input_pattern}")
        logger.error("Make sure GA results directories exist and contain CSV files")
        return 1
    
    logger.info(f"Loading data from {len(directories)} directories...")
    results = analyzer.load_ga_results(directories)
    
    if not results:
        logger.error("No valid GA results data found!")
        return 1
    
    logger.info(f"Successfully loaded data for {len(results)} compressors")
    
    # Generate analysis outputs
    if not args.no_plots:
        logger.info("Generating visualization plots...")
        analyzer.create_fitness_evolution_plot(results)
        analyzer.create_time_evolution_plot(results)
        analyzer.create_comparative_analysis_plot(results)
    
    logger.info("Creating summary statistics...")
    analyzer.create_summary_statistics(results)
    
    logger.info("Generating comprehensive report...")
    analyzer.generate_report(results)
    
    logger.info(f"Analysis complete! Results saved to: {analyzer.output_dir}")
    logger.info("Generated files:")
    for file in analyzer.output_dir.iterdir():
        logger.info(f"  - {file.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())