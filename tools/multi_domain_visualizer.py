"""
Multi-Domain Compression Analysis Visualizer

Advanced visualization tool for multi-domain compression analysis results.
Creates comprehensive plots and reports for cross-domain performance evaluation,
compressor comparisons, and domain-specific analysis.
"""

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import seaborn as sns
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ga_components.file_analyzer import DataType, DataDomain

# Set plotting style
plt.style.use('default')
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class MultiDomainVisualizer:
    """
    Comprehensive visualization system for multi-domain compression analysis.
    
    Creates various plots and reports to analyze compressor performance
    across different data domains, types, and characteristics.
    """
    
    def __init__(self, output_dir: str = "visualization_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Color schemes for consistency
        self.compressor_colors = {
            'zstd': '#1f77b4',    # Blue
            'lzma': '#ff7f0e',    # Orange  
            'brotli': '#2ca02c',  # Green
            'paq8': '#d62728',    # Red
            'ac2': '#9467bd'      # Purple
        }
        
        self.domain_colors = {
            'natural_language': '#1f77b4',
            'source_code': '#ff7f0e', 
            'chemical_data': '#2ca02c',
            'genomic_data': '#d62728',
            'numerical_data': '#9467bd',
            'binary': '#8c564b',
            'mixed': '#e377c2',
            'configuration': '#7f7f7f',
            'log_data': '#bcbd22',
            'image_data': '#17becf'
        }
    
    def create_comprehensive_analysis(self, results_file: str) -> Dict[str, str]:
        """
        Create comprehensive visual analysis from benchmark results.
        
        Args:
            results_file: Path to benchmark results JSON file
            
        Returns:
            Dictionary mapping plot types to output file paths
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        # Load results data
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        raw_results = data.get('raw_results', [])
        if not raw_results:
            raise ValueError("No raw results found in the results file")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(raw_results)
        
        # Extract file characteristics into separate columns
        char_df = pd.json_normalize(df['file_characteristics'])
        df = pd.concat([df.drop('file_characteristics', axis=1), char_df], axis=1)
        
        created_plots = {}
        
        # Create various visualizations
        created_plots['compression_performance'] = self._create_compression_performance_plot(df)
        created_plots['domain_comparison'] = self._create_domain_comparison_plot(df)
        created_plots['compressor_heatmap'] = self._create_compressor_performance_heatmap(df)
        created_plots['file_size_analysis'] = self._create_file_size_analysis(df)
        created_plots['entropy_correlation'] = self._create_entropy_correlation_plot(df)
        created_plots['prediction_accuracy'] = self._create_prediction_accuracy_plot(df)
        created_plots['performance_radar'] = self._create_performance_radar_chart(df)
        created_plots['runtime_analysis'] = self._create_runtime_analysis_plot(df)
        created_plots['cross_domain_summary'] = self._create_cross_domain_summary(df)
        
        # Create comprehensive report
        created_plots['analysis_report'] = self._create_analysis_report(df, data)
        
        return created_plots
    
    def _create_compression_performance_plot(self, df: pd.DataFrame) -> str:
        """Create comprehensive compression performance visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Box plot of compression ratios by compressor
        if SEABORN_AVAILABLE:
            sns.boxplot(data=df, x='compressor_name', y='best_compression_ratio', ax=ax1)
        else:
            compressors = df['compressor_name'].unique()
            ratios_by_comp = [df[df['compressor_name'] == comp]['best_compression_ratio'].values 
                             for comp in compressors]
            ax1.boxplot(ratios_by_comp, labels=compressors)
        
        ax1.set_title('Compression Ratio Distribution by Compressor')
        ax1.set_ylabel('Compression Ratio')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Scatter plot: compression ratio vs file size
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            ax2.scatter(comp_data['file_size'], comp_data['best_compression_ratio'], 
                       label=compressor, alpha=0.7, 
                       color=self.compressor_colors.get(compressor))
        
        ax2.set_xlabel('File Size (bytes)')
        ax2.set_ylabel('Compression Ratio')
        ax2.set_title('Compression Ratio vs File Size')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Average compression ratio by data type
        type_performance = df.groupby(['data_type', 'compressor_name'])['best_compression_ratio'].mean().unstack()
        type_performance.plot(kind='bar', ax=ax3, color=[self.compressor_colors.get(comp) for comp in type_performance.columns])
        ax3.set_title('Average Compression Ratio by Data Type')
        ax3.set_ylabel('Average Compression Ratio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Compression efficiency (ratio/time)
        df_with_time = df[df['best_compression_time'] > 0].copy()
        df_with_time['efficiency'] = df_with_time['best_compression_ratio'] / df_with_time['best_compression_time']
        
        efficiency_by_comp = df_with_time.groupby('compressor_name')['efficiency'].mean()
        ax4.bar(efficiency_by_comp.index, efficiency_by_comp.values,
                color=[self.compressor_colors.get(comp) for comp in efficiency_by_comp.index])
        ax4.set_title('Compression Efficiency (Ratio/Time)')
        ax4.set_ylabel('Efficiency (ratio/second)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / "compression_performance_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_domain_comparison_plot(self, df: pd.DataFrame) -> str:
        """Create domain-specific performance comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Heatmap of average compression ratios by domain and compressor
        domain_comp_pivot = df.groupby(['data_domain', 'compressor_name'])['best_compression_ratio'].mean().unstack()
        
        if SEABORN_AVAILABLE:
            sns.heatmap(domain_comp_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1)
        else:
            im = ax1.imshow(domain_comp_pivot.values, cmap='YlOrRd', aspect='auto')
            ax1.set_xticks(range(len(domain_comp_pivot.columns)))
            ax1.set_yticks(range(len(domain_comp_pivot.index)))
            ax1.set_xticklabels(domain_comp_pivot.columns)
            ax1.set_yticklabels(domain_comp_pivot.index)
            
            # Add text annotations
            for i in range(len(domain_comp_pivot.index)):
                for j in range(len(domain_comp_pivot.columns)):
                    ax1.text(j, i, f'{domain_comp_pivot.iloc[i, j]:.2f}',
                            ha='center', va='center', color='black')
            
            plt.colorbar(im, ax=ax1)
        
        ax1.set_title('Average Compression Ratio by Domain and Compressor')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # 2. Domain performance ranking
        domain_performance = df.groupby('data_domain')['best_compression_ratio'].agg(['mean', 'std'])
        domain_performance_sorted = domain_performance.sort_values('mean', ascending=True)
        
        y_pos = np.arange(len(domain_performance_sorted))
        ax2.barh(y_pos, domain_performance_sorted['mean'], 
                xerr=domain_performance_sorted['std'],
                color=[self.domain_colors.get(domain, '#1f77b4') 
                      for domain in domain_performance_sorted.index])
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([domain.replace('_', ' ').title() 
                           for domain in domain_performance_sorted.index])
        ax2.set_xlabel('Average Compression Ratio')
        ax2.set_title('Domain Performance Ranking')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "domain_comparison_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_compressor_performance_heatmap(self, df: pd.DataFrame) -> str:
        """Create detailed compressor performance heatmap."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate different metrics
        metrics = {
            'Compression Ratio': df.groupby(['data_type', 'compressor_name'])['best_compression_ratio'].mean(),
            'Compression Time': df[df['best_compression_time'] > 0].groupby(['data_type', 'compressor_name'])['best_compression_time'].mean(),
            'Runtime': df.groupby(['data_type', 'compressor_name'])['total_runtime'].mean(),
            'Success Rate': df.groupby(['data_type', 'compressor_name']).apply(
                lambda x: (x['best_fitness'] > 1.0).mean()
            )
        }
        
        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = axes[idx // 2, idx % 2]
            
            try:
                pivot_data = metric_data.unstack()
                
                if SEABORN_AVAILABLE:
                    sns.heatmap(pivot_data, annot=True, fmt='.2f', 
                              cmap='RdYlBu_r' if 'Time' in metric_name or 'Runtime' in metric_name else 'YlOrRd',
                              ax=ax)
                else:
                    im = ax.imshow(pivot_data.values, cmap='RdYlBu_r' if 'Time' in metric_name else 'YlOrRd', 
                                  aspect='auto')
                    ax.set_xticks(range(len(pivot_data.columns)))
                    ax.set_yticks(range(len(pivot_data.index)))
                    ax.set_xticklabels(pivot_data.columns)
                    ax.set_yticklabels(pivot_data.index)
                    
                    # Add annotations
                    for i in range(len(pivot_data.index)):
                        for j in range(len(pivot_data.columns)):
                            ax.text(j, i, f'{pivot_data.iloc[i, j]:.2f}',
                                   ha='center', va='center', color='black')
                
                ax.set_title(f'{metric_name} by Data Type and Compressor')
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Insufficient data\nfor {metric_name}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(metric_name)
        
        plt.tight_layout()
        output_path = self.output_dir / "compressor_performance_heatmaps.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_file_size_analysis(self, df: pd.DataFrame) -> str:
        """Analyze performance by file size ranges."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Define size categories
        df_copy = df.copy()
        df_copy['size_category'] = pd.cut(df_copy['file_size'], 
                                         bins=[0, 1024, 1024*1024, 10*1024*1024, float('inf')],
                                         labels=['< 1KB', '1KB-1MB', '1MB-10MB', '> 10MB'])
        
        # 1. Compression ratio by file size category
        size_performance = df_copy.groupby(['size_category', 'compressor_name'])['best_compression_ratio'].mean().unstack()
        size_performance.plot(kind='bar', ax=ax1, 
                             color=[self.compressor_colors.get(comp) for comp in size_performance.columns])
        ax1.set_title('Compression Ratio by File Size Category')
        ax1.set_ylabel('Average Compression Ratio')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. File size distribution by compressor performance
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            ax2.hist(comp_data['file_size'], bins=20, alpha=0.6, label=compressor,
                    color=self.compressor_colors.get(compressor))
        
        ax2.set_xlabel('File Size (bytes)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('File Size Distribution by Compressor')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "file_size_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_entropy_correlation_plot(self, df: pd.DataFrame) -> str:
        """Create entropy vs compression performance correlation plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Entropy vs Compression Ratio scatter plot
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            ax1.scatter(comp_data['entropy'], comp_data['best_compression_ratio'],
                       alpha=0.7, label=compressor, 
                       color=self.compressor_colors.get(compressor))
        
        ax1.set_xlabel('File Entropy')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('Compression Ratio vs File Entropy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Text ratio vs Compression Ratio
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            ax2.scatter(comp_data['text_ratio'], comp_data['best_compression_ratio'],
                       alpha=0.7, label=compressor,
                       color=self.compressor_colors.get(compressor))
        
        ax2.set_xlabel('Text Ratio')
        ax2.set_ylabel('Compression Ratio')  
        ax2.set_title('Compression Ratio vs Text Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Repetition factor vs Compression Ratio
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            ax3.scatter(comp_data['repetition_factor'], comp_data['best_compression_ratio'],
                       alpha=0.7, label=compressor,
                       color=self.compressor_colors.get(compressor))
        
        ax3.set_xlabel('Repetition Factor')
        ax3.set_ylabel('Compression Ratio')
        ax3.set_title('Compression Ratio vs Repetition Factor')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation heatmap
        numerical_cols = ['entropy', 'text_ratio', 'repetition_factor', 
                         'best_compression_ratio', 'file_size']
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) > 2:
            correlation_matrix = df[available_cols].corr()
            
            if SEABORN_AVAILABLE:
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                           center=0, ax=ax4)
            else:
                im = ax4.imshow(correlation_matrix.values, cmap='coolwarm', 
                               vmin=-1, vmax=1, aspect='auto')
                ax4.set_xticks(range(len(correlation_matrix.columns)))
                ax4.set_yticks(range(len(correlation_matrix.index)))
                ax4.set_xticklabels(correlation_matrix.columns)
                ax4.set_yticklabels(correlation_matrix.index)
                
                for i in range(len(correlation_matrix.index)):
                    for j in range(len(correlation_matrix.columns)):
                        ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                ha='center', va='center', color='black')
                
                plt.colorbar(im, ax=ax4)
            
            ax4.set_title('Feature Correlation Matrix')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Insufficient numerical\nfeatures for correlation',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        output_path = self.output_dir / "entropy_correlation_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_prediction_accuracy_plot(self, df: pd.DataFrame) -> str:
        """Create prediction accuracy visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate actual compressibility categories
        df_copy = df.copy()
        df_copy['actual_compressibility'] = pd.cut(df_copy['best_compression_ratio'],
                                                  bins=[0, 2.0, 3.0, float('inf')],
                                                  labels=['low', 'medium', 'high'])
        
        # 1. Prediction accuracy by compressor
        accuracy_by_comp = []
        compressors = []
        
        for compressor in df_copy['compressor_name'].unique():
            comp_data = df_copy[df_copy['compressor_name'] == compressor]
            accurate = (comp_data['predicted_compressibility'] == 
                       comp_data['actual_compressibility']).sum()
            total = len(comp_data)
            accuracy = accurate / total if total > 0 else 0
            accuracy_by_comp.append(accuracy)
            compressors.append(compressor)
        
        ax1.bar(compressors, accuracy_by_comp,
                color=[self.compressor_colors.get(comp) for comp in compressors])
        ax1.set_title('Compressibility Prediction Accuracy by Compressor')
        ax1.set_ylabel('Accuracy Rate')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, v in enumerate(accuracy_by_comp):
            ax1.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # 2. Confusion matrix for predictions
        if 'predicted_compressibility' in df_copy.columns:
            confusion_data = pd.crosstab(df_copy['predicted_compressibility'], 
                                       df_copy['actual_compressibility'])
            
            if SEABORN_AVAILABLE:
                sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=ax2)
            else:
                im = ax2.imshow(confusion_data.values, cmap='Blues', aspect='auto')
                ax2.set_xticks(range(len(confusion_data.columns)))
                ax2.set_yticks(range(len(confusion_data.index)))
                ax2.set_xticklabels(confusion_data.columns)
                ax2.set_yticklabels(confusion_data.index)
                
                for i in range(len(confusion_data.index)):
                    for j in range(len(confusion_data.columns)):
                        ax2.text(j, i, f'{confusion_data.iloc[i, j]}',
                                ha='center', va='center', color='black')
            
            ax2.set_title('Compressibility Prediction Confusion Matrix')
            ax2.set_xlabel('Actual Compressibility')
            ax2.set_ylabel('Predicted Compressibility')
        else:
            ax2.text(0.5, 0.5, 'No prediction data\navailable', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Prediction Accuracy Analysis')
        
        plt.tight_layout()
        output_path = self.output_dir / "prediction_accuracy_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_performance_radar_chart(self, df: pd.DataFrame) -> str:
        """Create radar chart for compressor performance comparison."""
        # Calculate normalized metrics for each compressor
        compressors = df['compressor_name'].unique()
        
        # Define metrics (normalized 0-1)
        metrics = {}
        for compressor in compressors:
            comp_data = df[df['compressor_name'] == compressor]
            
            # Calculate metrics
            avg_ratio = comp_data['best_compression_ratio'].mean()
            avg_time = comp_data[comp_data['best_compression_time'] > 0]['best_compression_time'].mean()
            success_rate = (comp_data['best_fitness'] > 1.0).mean()
            versatility = len(comp_data['data_domain'].unique()) / len(df['data_domain'].unique())
            
            metrics[compressor] = {
                'Compression Ratio': min(avg_ratio / df['best_compression_ratio'].max(), 1.0),
                'Speed': min(1.0 / (avg_time / df[df['best_compression_time'] > 0]['best_compression_time'].min()) if avg_time > 0 else 0.5, 1.0),
                'Success Rate': success_rate,
                'Versatility': versatility,
                'Consistency': 1.0 - (comp_data['best_compression_ratio'].std() / comp_data['best_compression_ratio'].mean())
            }
        
        # Create radar chart
        categories = list(metrics[compressors[0]].keys())
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for compressor in compressors:
            values = list(metrics[compressor].values())
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=compressor,
                   color=self.compressor_colors.get(compressor))
            ax.fill(angles, values, alpha=0.25, 
                   color=self.compressor_colors.get(compressor))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Compressor Performance Comparison\n(Radar Chart)', 
                    size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / "performance_radar_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_runtime_analysis_plot(self, df: pd.DataFrame) -> str:
        """Create runtime and efficiency analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Runtime distribution by compressor
        runtime_data = []
        labels = []
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]['total_runtime']
            runtime_data.append(comp_data.values)
            labels.append(compressor)
        
        ax1.boxplot(runtime_data, labels=labels)
        ax1.set_title('Runtime Distribution by Compressor')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Compression time vs file size
        df_with_time = df[df['best_compression_time'] > 0]
        for compressor in df_with_time['compressor_name'].unique():
            comp_data = df_with_time[df_with_time['compressor_name'] == compressor]
            ax2.scatter(comp_data['file_size'], comp_data['best_compression_time'],
                       alpha=0.7, label=compressor,
                       color=self.compressor_colors.get(compressor))
        
        ax2.set_xlabel('File Size (bytes)')
        ax2.set_ylabel('Compression Time (seconds)')
        ax2.set_title('Compression Time vs File Size')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency by data domain
        if len(df_with_time) > 0:
            df_with_time = df_with_time.copy()
            df_with_time['efficiency'] = df_with_time['best_compression_ratio'] / df_with_time['best_compression_time']
            
            domain_efficiency = df_with_time.groupby(['data_domain', 'compressor_name'])['efficiency'].mean().unstack()
            domain_efficiency.plot(kind='bar', ax=ax3,
                                  color=[self.compressor_colors.get(comp) for comp in domain_efficiency.columns])
            ax3.set_title('Compression Efficiency by Domain')
            ax3.set_ylabel('Efficiency (ratio/second)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No timing data\navailable', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Compression Efficiency by Domain')
        
        # 4. Performance vs runtime scatter
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            ax4.scatter(comp_data['total_runtime'], comp_data['best_compression_ratio'],
                       alpha=0.7, label=compressor,
                       color=self.compressor_colors.get(compressor))
        
        ax4.set_xlabel('Total Runtime (seconds)')
        ax4.set_ylabel('Compression Ratio')
        ax4.set_title('Performance vs Runtime Trade-off')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "runtime_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_cross_domain_summary(self, df: pd.DataFrame) -> str:
        """Create comprehensive cross-domain summary visualization."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall performance ranking (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        overall_performance = df.groupby('compressor_name')['best_compression_ratio'].mean().sort_values(ascending=False)
        bars = ax1.bar(overall_performance.index, overall_performance.values,
                      color=[self.compressor_colors.get(comp) for comp in overall_performance.index])
        ax1.set_title('Overall Performance Ranking', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Compression Ratio')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_performance.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Domain leadership chart (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        domain_leaders = {}
        for domain in df['data_domain'].unique():
            domain_data = df[df['data_domain'] == domain]
            leader = domain_data.groupby('compressor_name')['best_compression_ratio'].mean().idxmax()
            domain_leaders[domain] = leader
        
        leadership_counts = pd.Series(domain_leaders).value_counts()
        ax2.pie(leadership_counts.values, labels=leadership_counts.index, autopct='%1.1f%%',
               colors=[self.compressor_colors.get(comp) for comp in leadership_counts.index])
        ax2.set_title('Domain Leadership Distribution', fontsize=14, fontweight='bold')
        
        # 3. File count and size distribution (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        file_stats = df.groupby('compressor_name').agg({
            'file_path': 'count',
            'file_size': 'mean'
        }).rename(columns={'file_path': 'file_count', 'file_size': 'avg_file_size'})
        
        x = np.arange(len(file_stats.index))
        width = 0.35
        
        ax3_twin = ax3.twinx()
        bars1 = ax3.bar(x - width/2, file_stats['file_count'], width, 
                       label='File Count', alpha=0.8, 
                       color=[self.compressor_colors.get(comp) for comp in file_stats.index])
        bars2 = ax3_twin.bar(x + width/2, file_stats['avg_file_size'] / (1024*1024), width,
                            label='Avg Size (MB)', alpha=0.8, 
                            color=[self.compressor_colors.get(comp) for comp in file_stats.index])
        
        ax3.set_title('File Count and Average Size by Compressor', fontsize=14, fontweight='bold')
        ax3.set_ylabel('File Count')
        ax3_twin.set_ylabel('Average Size (MB)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(file_stats.index, rotation=45)
        
        # 4. Success rate by data type (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        success_by_type = df.groupby(['data_type', 'compressor_name']).apply(
            lambda x: (x['best_fitness'] > 1.0).mean()
        ).unstack()
        
        if SEABORN_AVAILABLE:
            sns.heatmap(success_by_type, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4)
        else:
            im = ax4.imshow(success_by_type.values, cmap='RdYlGn', aspect='auto')
            ax4.set_xticks(range(len(success_by_type.columns)))
            ax4.set_yticks(range(len(success_by_type.index)))
            ax4.set_xticklabels(success_by_type.columns)
            ax4.set_yticklabels(success_by_type.index)
        
        ax4.set_title('Success Rate by Data Type', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Performance consistency (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        consistency_data = df.groupby('compressor_name')['best_compression_ratio'].agg(['std', 'mean'])
        consistency_data['cv'] = consistency_data['std'] / consistency_data['mean']  # Coefficient of variation
        
        bars = ax5.bar(consistency_data.index, 1 - consistency_data['cv'],  # Higher is more consistent
                      color=[self.compressor_colors.get(comp) for comp in consistency_data.index])
        ax5.set_title('Performance Consistency Score', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Consistency Score (0-1)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.set_ylim(0, 1)
        
        # 6. Best use cases (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # Create recommendations text
        recommendations_text = "RECOMMENDED USE CASES:\n\n"
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            best_domains = comp_data.groupby('data_domain')['best_compression_ratio'].mean().nlargest(2)
            
            recommendations_text += f"{compressor.upper()}:\n"
            for domain, ratio in best_domains.items():
                recommendations_text += f"  • {domain.replace('_', ' ').title()} (avg: {ratio:.2f}x)\n"
            recommendations_text += "\n"
        
        ax6.text(0.05, 0.95, recommendations_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 7. Summary statistics table (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary statistics
        summary_stats = []
        for compressor in df['compressor_name'].unique():
            comp_data = df[df['compressor_name'] == compressor]
            stats = {
                'Compressor': compressor.upper(),
                'Files': len(comp_data),
                'Avg Ratio': f"{comp_data['best_compression_ratio'].mean():.2f}",
                'Best Ratio': f"{comp_data['best_compression_ratio'].max():.2f}",
                'Success Rate': f"{(comp_data['best_fitness'] > 1.0).mean():.1%}",
                'Domains': len(comp_data['data_domain'].unique()),
                'Avg Runtime': f"{comp_data['total_runtime'].mean():.1f}s"
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create table
        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append(list(row.values))
        
        table = ax7.table(cellText=table_data,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code the table by compressor
        for i, compressor in enumerate(summary_df['Compressor']):
            color = self.compressor_colors.get(compressor.lower(), '#ffffff')
            for j in range(len(summary_df.columns)):
                table[(i + 1, j)].set_facecolor(color)
                table[(i + 1, j)].set_alpha(0.3)
        
        plt.suptitle('Multi-Domain Compression Analysis - Executive Summary', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        output_path = self.output_dir / "cross_domain_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_analysis_report(self, df: pd.DataFrame, full_data: Dict) -> str:
        """Create comprehensive markdown analysis report."""
        report_path = self.output_dir / "multi_domain_analysis_report.md"
        
        # Calculate key statistics
        total_files = len(df)
        total_compressors = len(df['compressor_name'].unique())
        total_domains = len(df['data_domain'].unique())
        avg_compression_ratio = df['best_compression_ratio'].mean()
        best_compression_ratio = df['best_compression_ratio'].max()
        
        # Find best performer overall
        best_performer = df.loc[df['best_compression_ratio'].idxmax()]
        
        # Calculate domain-specific winners
        domain_winners = {}
        for domain in df['data_domain'].unique():
            domain_data = df[df['data_domain'] == domain]
            winner_idx = domain_data['best_compression_ratio'].idxmax()
            winner = domain_data.loc[winner_idx]
            domain_winners[domain] = {
                'compressor': winner['compressor_name'],
                'ratio': winner['best_compression_ratio'],
                'file': winner['file_path']
            }
        
        report_content = f"""# Multi-Domain Compression Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of compression performance across multiple data domains and file types. The analysis covers {total_files} files across {total_domains} different data domains, evaluated using {total_compressors} different compression algorithms.

### Key Findings

- **Overall Best Performance:** {best_compression_ratio:.2f}x compression ratio
- **Average Performance:** {avg_compression_ratio:.2f}x compression ratio  
- **Top Performer:** {best_performer['compressor_name'].upper()} on {Path(best_performer['file_path']).name}
- **Most Versatile:** Analysis across {total_domains} data domains

## Performance Summary by Compressor

"""
        
        # Add per-compressor analysis
        for compressor in sorted(df['compressor_name'].unique()):
            comp_data = df[df['compressor_name'] == compressor]
            
            avg_ratio = comp_data['best_compression_ratio'].mean()
            best_ratio = comp_data['best_compression_ratio'].max()
            success_rate = (comp_data['best_fitness'] > 1.0).mean()
            domains_covered = len(comp_data['data_domain'].unique())
            
            report_content += f"""### {compressor.upper()}

- **Files Processed:** {len(comp_data)}
- **Average Compression Ratio:** {avg_ratio:.2f}x
- **Best Compression Ratio:** {best_ratio:.2f}x
- **Success Rate:** {success_rate:.1%}
- **Domains Covered:** {domains_covered}/{total_domains}

"""
        
        report_content += """## Domain-Specific Analysis

The following section details the best-performing compressor for each data domain:

"""
        
        for domain, winner_info in domain_winners.items():
            report_content += f"""### {domain.replace('_', ' ').title()}

- **Best Compressor:** {winner_info['compressor'].upper()}
- **Best Ratio:** {winner_info['ratio']:.2f}x
- **Best File:** {Path(winner_info['file']).name}

"""
        
        # Add recommendations
        report_content += """## Recommendations

Based on the comprehensive analysis across multiple domains:

### General Purpose
- **For Speed:** ZSTD offers the best balance of compression and speed
- **For Maximum Compression:** LZMA typically provides the highest compression ratios
- **For Web Content:** Brotli excels with text-based data

### Domain-Specific Recommendations

"""
        
        # Domain-specific recommendations
        domain_recommendations = {
            'natural_language': 'Brotli or LZMA for text content',
            'source_code': 'LZMA for structured code files',
            'chemical_data': 'LZMA for structured scientific data',
            'genomic_data': 'LZMA for highly repetitive sequences',
            'binary': 'ZSTD for general binary data',
            'numerical_data': 'LZMA for tabular data',
            'log_data': 'LZMA for repetitive log patterns'
        }
        
        for domain, recommendation in domain_recommendations.items():
            if domain in df['data_domain'].values:
                report_content += f"- **{domain.replace('_', ' ').title()}:** {recommendation}\n"
        
        report_content += f"""

## Technical Details

### Data Distribution
- **Total Files Analyzed:** {total_files:,}
- **File Size Range:** {df['file_size'].min():,} - {df['file_size'].max():,} bytes
- **Average File Size:** {df['file_size'].mean():,.0f} bytes

### Performance Metrics
- **Entropy Range:** {df['entropy'].min():.3f} - {df['entropy'].max():.3f}
- **Text Ratio Range:** {df['text_ratio'].min():.3f} - {df['text_ratio'].max():.3f}
- **Repetition Factor Range:** {df['repetition_factor'].min():.3f} - {df['repetition_factor'].max():.3f}

### Analysis Configuration
- **Population Size:** {full_data.get('benchmark_summary', {}).get('ga_config', {}).get('population_size', 'N/A')}
- **Generations:** {full_data.get('benchmark_summary', {}).get('ga_config', {}).get('generations', 'N/A')}
- **Total Runtime:** {full_data.get('benchmark_summary', {}).get('total_runtime', 0):.2f} seconds

## Visualization Outputs

The following visualizations have been generated as part of this analysis:

1. **Compression Performance Analysis** - Overall performance comparison
2. **Domain Comparison Analysis** - Domain-specific performance heatmaps
3. **Compressor Performance Heatmaps** - Detailed metric comparisons
4. **File Size Analysis** - Performance by file size categories
5. **Entropy Correlation Analysis** - File characteristics vs performance
6. **Prediction Accuracy Analysis** - Compressibility prediction evaluation
7. **Performance Radar Chart** - Multi-dimensional comparison
8. **Runtime Analysis** - Timing and efficiency analysis
9. **Cross-Domain Summary** - Executive summary visualization

## Conclusions

This multi-domain analysis demonstrates that compression performance varies significantly across different data types and domains. The choice of compression algorithm should be guided by:

1. **Data characteristics** (entropy, text ratio, repetition patterns)
2. **Performance requirements** (speed vs compression ratio)
3. **Domain-specific patterns** (structured vs unstructured data)
4. **File size considerations** (algorithm efficiency at different scales)

The analysis provides a solid foundation for making informed decisions about compression algorithm selection based on specific use cases and data characteristics.

---
*Report generated by Multi-Domain Compression Analysis System*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description='Multi-Domain Compression Analysis Visualizer')
    parser.add_argument('--results-file', required=True,
                       help='Path to benchmark results JSON file')
    parser.add_argument('--output-dir', default='visualization_output',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    print(f"Creating multi-domain analysis visualizations...")
    print(f"Input: {args.results_file}")
    print(f"Output: {args.output_dir}")
    
    visualizer = MultiDomainVisualizer(args.output_dir)
    
    try:
        created_plots = visualizer.create_comprehensive_analysis(args.results_file)
        
        print(f"\nVisualization completed successfully!")
        print(f"Generated {len(created_plots)} visualizations:")
        
        for plot_type, file_path in created_plots.items():
            print(f"  - {plot_type}: {file_path}")
        
        print(f"\nAll outputs saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()