"""
Reporting and I/O Module

Handles data persistence, progress reporting, and result visualization
for genetic algorithm runs.

Features:
- CSV generation and export
- Progress tracking and logging
- Performance metrics reporting
- Result visualization utilities
- Run statistics and analysis
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from ga_logging import get_logger
import sys


class GAReporter:
    """
    Comprehensive reporting and I/O manager for genetic algorithms.
    
    Handles generation data export, progress tracking, and result analysis
    with support for multiple output formats and real-time monitoring.
    """
    
    def __init__(self, output_dir: str = "ga_results", experiment_name: str = None):
        """
        Initialize GA reporter.
        
        Args:
            output_dir: Directory for output files
            experiment_name: Name of the experiment (auto-generated if None)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"ga_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = get_logger("Reporter")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tracking
        self.start_time = None
        self.generation_data = []
        self.best_fitness_history = []
        self.statistics = {
            'total_generations': 0,
            'total_evaluations': 0,
            'best_overall_fitness': None,
            'convergence_generation': None,
            'total_runtime': 0.0
        }
        
        # Files
        self.log_file = None
        self.summary_file = None
    
    def start_run(self, run_config: Dict[str, Any]):
        """
        Start a new GA run and initialize logging.
        
        Args:
            run_config: Configuration parameters for the run
        """
        self.start_time = time.time()
        
        # Create log file
        log_filename = os.path.join(self.output_dir, f"{self.experiment_name}_log.txt")
        self.log_file = open(log_filename, 'w')
        
        # Log run configuration
        self.log(f"Starting GA run: {self.experiment_name}")
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("Configuration:")
        for key, value in run_config.items():
            self.log(f"  {key}: {value}")
        self.log("-" * 50)
    
    def log(self, message: str, also_print: bool = False):
        """
        Log a message to the log file.
        
        Args:
            message: Message to log
            also_print: Whether to also print to console
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        if self.log_file:
            self.log_file.write(log_message + '\n')
            self.log_file.flush()
        
        if also_print:
            self.logger.info(log_message)
    
    def save_generation_data(self, generation: int, population_with_fitness: List[Tuple[Any, float]], 
                           population_manager: Any, additional_data: Dict[str, Any] = None, 
                           compression_times: List[float] = None):
        """
        Save generation data to CSV and update tracking.
        
        Args:
            generation: Generation number
            population_with_fitness: Population with fitness values
            population_manager: Population manager for decoding
            additional_data: Additional metrics to save
        """
        # CSV filename for this generation
        csv_filename = os.path.join(self.output_dir, f"generation_{generation}.csv")
        
        # Prepare data
        generation_info = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'population_size': len(population_with_fitness),
            'best_fitness': max(fitness for _, fitness in population_with_fitness),
            'avg_fitness': sum(fitness for _, fitness in population_with_fitness) / len(population_with_fitness),
            'worst_fitness': min(fitness for _, fitness in population_with_fitness)
        }
        
        if additional_data:
            generation_info.update(additional_data)
        
        self.generation_data.append(generation_info)
        self.best_fitness_history.append(generation_info['best_fitness'])
        
        # Update statistics
        self.statistics['total_generations'] = generation
        if self.statistics['best_overall_fitness'] is None or generation_info['best_fitness'] > self.statistics['best_overall_fitness']:
            self.statistics['best_overall_fitness'] = generation_info['best_fitness']
        
        # Save CSV
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Get parameter names from first individual
            if population_with_fitness:
                first_individual = population_with_fitness[0][0]
                decoded = population_manager.decode_individual(first_individual)
                
                if decoded:  # decoded is a list of dicts for each model
                    param_names = list(decoded[0].keys()) if decoded[0] else []
                    
                    # Header
                    header = ['Individual_Name', 'Fitness', 'Compression_Time'] + param_names
                    writer.writerow(header)
                    
                    # Data rows
                    for i, (individual, fitness) in enumerate(population_with_fitness):
                        decoded_individual = population_manager.decode_individual(individual)
                        individual_name = individual[1] if len(individual) > 1 else f"Ind_{hash(individual[0])}"
                        
                        # Use first model's parameters (assuming single model)
                        params = decoded_individual[0] if decoded_individual else {}
                        
                        # Get compression time for this individual
                        compression_time = compression_times[i] if compression_times and i < len(compression_times) else 0.0
                        
                        row = [individual_name, fitness, compression_time] + [params.get(param, 'N/A') for param in param_names]
                        writer.writerow(row)
        
        # Log generation summary
        self.log(f"Generation {generation}: Best={generation_info['best_fitness']:.4f}, "
                f"Avg={generation_info['avg_fitness']:.4f}, Pop={generation_info['population_size']}")
    
    def save_run_summary(self, final_result: Any, convergence_info: Dict[str, Any] = None):
        """
        Save comprehensive run summary.
        
        Args:
            final_result: Final result of the GA run
            convergence_info: Information about convergence
        """
        if self.start_time:
            self.statistics['total_runtime'] = time.time() - self.start_time
        
        if convergence_info:
            self.statistics.update(convergence_info)
        
        # Save summary JSON
        summary_filename = os.path.join(self.output_dir, f"{self.experiment_name}_summary.json")
        
        summary_data = {
            'experiment_name': self.experiment_name,
            'end_time': datetime.now().isoformat(),
            'statistics': self.statistics,
            'final_result': str(final_result),  # Convert to string for JSON serialization
            'generation_summary': self.generation_data,
            'fitness_history': self.best_fitness_history
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Log final summary
        self.log("=" * 50)
        self.log("RUN SUMMARY")
        self.log("=" * 50)
        for key, value in self.statistics.items():
            self.log(f"{key}: {value}")
        
        if convergence_info:
            self.log("Convergence Information:")
            for key, value in convergence_info.items():
                self.log(f"  {key}: {value}")
    
    def save_best_parameters_csv(self, population_manager: Any):
        """
        Save the best parameters found across all generations to a summary CSV.
        
        Args:
            population_manager: Population manager for parameter decoding
        """
        if not self.generation_data:
            return
        
        # Find generation with best fitness
        best_gen = max(self.generation_data, key=lambda x: x['best_fitness'])
        best_gen_num = best_gen['generation']
        
        # Load that generation's CSV to get the best individual
        csv_filename = os.path.join(self.output_dir, f"generation_{best_gen_num}.csv")
        
        if os.path.exists(csv_filename):
            best_params_filename = os.path.join(self.output_dir, f"{self.experiment_name}_best_parameters.csv")
            
            try:
                with open(csv_filename, 'r') as infile, open(best_params_filename, 'w', newline='') as outfile:
                    reader = csv.DictReader(infile)
                    rows = list(reader)
                    
                    if rows:
                        # Find row with highest fitness
                        best_row = max(rows, key=lambda x: float(x.get('Fitness', 0)))
                        
                        # Write best parameters
                        fieldnames = ['Parameter', 'Value']
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        # Write metadata
                        writer.writerow({'Parameter': 'Experiment_Name', 'Value': self.experiment_name})
                        writer.writerow({'Parameter': 'Generation', 'Value': best_gen_num})
                        writer.writerow({'Parameter': 'Best_Fitness', 'Value': best_row['Fitness']})
                        writer.writerow({'Parameter': 'Individual_Name', 'Value': best_row['Individual_Name']})
                        writer.writerow({'Parameter': '', 'Value': ''})  # Separator
                        
                        # Write parameters
                        for key, value in best_row.items():
                            if key not in ['Individual_Name', 'Fitness']:
                                writer.writerow({'Parameter': key, 'Value': value})
            
            except Exception as e:
                self.log(f"Error saving best parameters CSV: {e}")
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """
        Generate current progress report.
        
        Returns:
            Progress report dictionary
        """
        if not self.generation_data:
            return {'status': 'No data available'}
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        latest_gen = self.generation_data[-1]
        
        # Calculate improvement trend
        improvement_trend = 0
        if len(self.best_fitness_history) >= 2:
            recent_window = min(10, len(self.best_fitness_history))
            recent_fitnesses = self.best_fitness_history[-recent_window:]
            if len(recent_fitnesses) >= 2:
                improvement_trend = (recent_fitnesses[-1] - recent_fitnesses[0]) / len(recent_fitnesses)
        
        return {
            'current_generation': latest_gen['generation'],
            'elapsed_time_seconds': elapsed_time,
            'best_fitness_so_far': self.statistics['best_overall_fitness'],
            'current_best_fitness': latest_gen['best_fitness'],
            'current_avg_fitness': latest_gen['avg_fitness'],
            'improvement_trend': improvement_trend,
            'generations_completed': len(self.generation_data),
            'status': 'Running' if elapsed_time > 0 else 'Completed'
        }
    
    def export_fitness_history(self, filename: str = None) -> str:
        """
        Export fitness history to CSV.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.experiment_name}_fitness_history.csv")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best_Fitness', 'Avg_Fitness', 'Worst_Fitness'])
            
            for gen_data in self.generation_data:
                writer.writerow([
                    gen_data['generation'],
                    gen_data['best_fitness'],
                    gen_data['avg_fitness'],
                    gen_data['worst_fitness']
                ])
        
        return filename
    
    def create_performance_summary(self, component_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create comprehensive performance summary from all GA components.
        
        Args:
            component_stats: Statistics from all GA components
            
        Returns:
            Performance summary
        """
        summary = {
            'experiment_info': {
                'name': self.experiment_name,
                'output_dir': self.output_dir,
                'total_runtime': self.statistics['total_runtime'],
                'generations_completed': self.statistics['total_generations']
            },
            'performance_metrics': self.statistics.copy(),
            'component_statistics': component_stats
        }
        
        # Add derived metrics
        if self.statistics['total_runtime'] > 0 and self.statistics['total_generations'] > 0:
            summary['performance_metrics']['avg_time_per_generation'] = self.statistics['total_runtime'] / self.statistics['total_generations']
        
        return summary
    
    def cleanup(self):
        """Clean up resources and close files."""
        if self.log_file:
            self.log("Run completed.")
            self.log_file.close()
            self.log_file = None
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()