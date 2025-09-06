import time
import sys
from typing import Dict, List, Any, Tuple
from cache import get_global_cache
from ga_config import GAConfig
from ga_logging import get_logger
from ga_exceptions import GAException
from ga_components.duplicate_prevention import DuplicatePreventionSystem
from ga_components.convergence_detection import ConvergenceDetector
from ga_components.genetic_operations import GeneticOperations
from ga_components.selection import SelectionMethods
from ga_components.population_management import PopulationManager
from ga_components.evaluation import EvaluationEngine
from ga_components.parameter_encoding import ParameterEncoder
from ga_components.reporting import GAReporter
from ga_components.algorithm_optimization import (
    AdaptiveParameterTuning, SmartInitialization, 
    ConvergenceAccelerator, PerformanceMonitor
)


class GeneticAlgorithm:
    def __init__(self, param_values: Dict[str, List[Any]], compressor: Any, config: GAConfig) -> None:
        
        # Store core components
        self.param_values = param_values
        self.compressor = compressor
        self.config = config
        self.logger = get_logger("GeneticAlgorithm")
        
        # Initialize modular components using config values
        self.parameter_encoder = ParameterEncoder(param_values)
        
        self.population_manager = PopulationManager(
            self.parameter_encoder.param_binary_encodings, 
            config.population_size, 
            compressor.nr_models
        )
        
        self.selection_methods = SelectionMethods(
            tournament_size=config.tournament_size,
            elite_ratio=config.elite_ratio
        )
        
        self.genetic_operations = GeneticOperations(
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate
        )
        
        self.evaluation_engine = EvaluationEngine(
            compressor, 
            self.population_manager,
            max_threads=config.max_threads,
            min_fitness=config.min_fitness,
            enable_dynamic_scaling=config.enable_dynamic_thread_scaling
        )
        
        self.duplicate_prevention = DuplicatePreventionSystem(
            self.parameter_encoder.total_parameter_combinations,
            config.population_size, 
            config.generations, 
            enabled=True
        )
        
        self.convergence_detector = ConvergenceDetector(
            config.generations, 
            config.convergence_generations, 
            config.convergence_threshold
        )
        
        self.reporter = GAReporter(
            output_dir=config.output_dir,
            experiment_name=f"ga_run_{int(time.time())}"
        )
        
        # Initialize optimization components based on config
        self.adaptive_tuning = AdaptiveParameterTuning(
            initial_mutation_rate=config.mutation_rate,
            initial_crossover_rate=config.crossover_rate
        ) if config.enable_adaptive_tuning else None
        
        self.convergence_accelerator = ConvergenceAccelerator(
            confidence_threshold=0.95,
            min_improvement=1e-6
        ) if config.enable_convergence_acceleration else None
        
        self.performance_monitor = PerformanceMonitor() if config.enable_performance_monitoring else None
        
        # Initialize population with smart or standard initialization
        if config.enable_smart_initialization:
            self.population = SmartInitialization.create_diverse_population(
                self.parameter_encoder.param_binary_encodings,
                config.population_size,
                self.population_manager.create_individual_name,
                diversity_strategy=config.diversity_strategy
            )
        else:
            self.population = self.population_manager.initialize_population()
        
        # Apply initial diversity enforcement
        self.population = self.duplicate_prevention.enforce_population_diversity(
            self.population, 1, self.parameter_encoder.param_binary_encodings,
            compressor.nr_models, self.population_manager.create_individual_name
        )
        
        # Simple decoding cache to avoid redundant decoding
        self._decode_cache = {}
    
    def _decode_individual_cached(self, individual):
        """Decode individual with caching to avoid redundant computation."""
        gene_code, individual_name = individual
        
        # Use gene_code as cache key (tuple is hashable)
        if gene_code not in self._decode_cache:
            self._decode_cache[gene_code] = self.population_manager.decode_individual(individual)
        
        return self._decode_cache[gene_code]
    
    def run(self) -> Tuple[List[Dict[str, Any]], float]:
        """Runs the genetic algorithm using modular components."""
        # Start experiment reporting
        run_config = {
            'population_size': self.config.population_size,
            'generations': self.config.generations, 
            'mutation_rate': self.config.mutation_rate,
            'crossover_rate': self.config.crossover_rate,
            'offspring_ratio': self.config.offspring_ratio,
            'max_threads': self.config.max_threads,
            'param_values': {k: len(v) for k, v in self.param_values.items()}
        }
        
        self.reporter.start_run(run_config)
        
        init_time = time.time()
        self.logger.info("-" * 100)
        self.logger.log_generation_start(1, self.config.population_size)
        self.compressor.erase_temp_files()
        time.sleep(1)
        
        # Evaluate initial population
        fitness_results, peak_memory = self.evaluation_engine.evaluate_population_parallel(self.population, generation=0)
        population_with_fitness = [(ind, fit) for ind, fit in zip(self.population, fitness_results)]
        
        # Sort by fitness
        self.population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
        
        # Save generation data 
        self.reporter.save_generation_data(1, self.population, self.population_manager)
        
        # Track convergence
        best_fitness = self.population[0][1]
        self.convergence_detector.add_fitness(best_fitness)
        
        # Display best individual
        best_individual = self.population[0][0]
        decoded_best = self._decode_individual_cached(best_individual)
        
        generation_time = time.time() - init_time
        self.logger.log_generation_complete(1, best_fitness, generation_time, peak_memory)
        self.logger.info(f"Best individual from Initial Generation: {decoded_best}")
        self.logger.info("-" * 100)
        
        # Evolution loop with optimization
        generation = 0  # Initialize outside loop for scope
        is_converged = False  # Initialize convergence flag
        reason = "Max generations reached"  # Default reason
        last_fitness = best_fitness  # For performance monitoring
        
        for generation in range(1, self.config.generations):
            init_time = time.time()
            eval_start_time = time.time()
            self.logger.log_generation_start(generation + 1, self.config.population_size)
            
            self.compressor.erase_temp_files()
            time.sleep(1)
            
            # Calculate population diversity for adaptive tuning
            population_diversity = self.population_manager.get_population_diversity(
                [ind[0] for ind in self.population]
            )
            
            # Adaptive parameter tuning (if enabled)
            current_mutation_rate = self.config.mutation_rate
            current_crossover_rate = self.config.crossover_rate
            
            if self.adaptive_tuning:
                current_mutation_rate, current_crossover_rate = self.adaptive_tuning.update_parameters(
                    population_diversity, best_fitness, generation + 1, self.config.generations
                )
                
                # Update genetic operations with adaptive rates
                self.genetic_operations.mutation_rate = current_mutation_rate
                self.genetic_operations.crossover_rate = current_crossover_rate
            
            # Check for restart recommendation (if convergence acceleration enabled)
            should_restart = False
            restart_reason = ""
            if self.convergence_accelerator:
                should_restart, restart_reason = self.convergence_accelerator.should_restart(
                    population_diversity, generation + 1
                )
            
            if should_restart:
                self.logger.info(f"Restarting population: {restart_reason}")
                # Keep best individuals but generate new diverse population
                elite_count = max(2, self.config.population_size // 10)
                elites = [ind[0] for ind in self.population[:elite_count]]
                
                new_population = SmartInitialization.create_diverse_population(
                    self.parameter_encoder.param_binary_encodings,
                    self.config.population_size - elite_count,
                    self.population_manager.create_individual_name,
                    diversity_strategy="structured"
                )
                
                # Combine elites with new population
                elite_individuals = [(elite, f"Elite_{i+1}") for i, elite in enumerate(elites)]
                self.population = elite_individuals + new_population
                
                # Re-evaluate and sort
                fitness_results, _ = self.evaluation_engine.evaluate_population_parallel(
                    [ind for ind, _ in self.population], generation=generation
                )
                self.population = [(ind, fit) for (ind, _), fit in zip(self.population, fitness_results)]
                self.population = sorted(self.population, key=lambda x: x[1], reverse=True)
            
            # Generate offspring through selection, crossover, and mutation
            offspring = []
            num_pairs = self.config.num_offspring // 2
            
            parent_pairs = self.selection_methods.select_parents(self.population, num_pairs)
            
            for parent1, parent2 in parent_pairs:
                # Crossover
                child1, child2 = self.genetic_operations.crossover(
                    parent1[0], parent2[0], generation + 1,
                    self.parameter_encoder.param_binary_encodings,
                    self.population_manager.create_individual_name)
                    
                # Mutation
                child1 = self.genetic_operations.mutate(child1, generation + 1,
                                                       self.population_manager.create_individual_name)
                child2 = self.genetic_operations.mutate(child2, generation + 1,
                                                       self.population_manager.create_individual_name)
                
                offspring.extend([child1, child2])
            
            # Apply diversity enforcement
            offspring = self.duplicate_prevention.enforce_population_diversity(
                offspring, generation + 1, self.parameter_encoder.param_binary_encodings,
                self.compressor.nr_models, self.population_manager.create_individual_name)
            
            # Evaluate offspring and select next generation
            fitness_results, peak_memory = self.evaluation_engine.evaluate_population_parallel(offspring, generation=generation)
            offspring_with_fitness = [(ind, fit) for ind, fit in zip(offspring, fitness_results)]
            
            # Elitist selection for next generation
            self.population = self.selection_methods.elitist_selection(
                self.population, offspring_with_fitness, self.config.population_size)
                
            # Save generation data
            self.reporter.save_generation_data(generation + 1, self.population, self.population_manager)
            
            # Track convergence
            best_fitness = self.population[0][1]
            self.convergence_detector.add_fitness(best_fitness)
            
            # Performance monitoring (if enabled)
            if self.performance_monitor:
                eval_time = time.time() - eval_start_time
                fitness_improvement = best_fitness - last_fitness
                self.performance_monitor.record_generation_metrics(
                    generation_time=time.time() - init_time,
                    evaluation_time=eval_time,
                    memory_usage=peak_memory,
                    fitness_improvement=fitness_improvement
                )
            last_fitness = best_fitness
            
            # Display progress
            best_individual = self.population[0][0]
            decoded_best = self._decode_individual_cached(best_individual)
            generation_time = time.time() - init_time
            
            self.logger.log_generation_complete(generation + 1, best_fitness, generation_time, peak_memory)
            self.logger.info(f"Best individual: {decoded_best}")
            
            # Log optimization status
            if generation % 5 == 0:  # Every 5 generations
                if self.adaptive_tuning:
                    self.logger.info(f"Adaptive rates: mutation={current_mutation_rate:.4f}, crossover={current_crossover_rate:.4f}")
                self.logger.info(f"Population diversity: {population_diversity:.3f}")
                
                # Get performance recommendations
                if self.performance_monitor:
                    recommendations = self.performance_monitor.get_optimization_recommendations()
                    if recommendations:
                        self.logger.info(f"Optimization suggestions: {'; '.join(recommendations[:2])}")
            
            # Check convergence (both standard and accelerated)
            is_converged, reason = self.convergence_detector.check_convergence(generation + 1)
            
            # Check early stopping (if convergence acceleration enabled)
            should_stop_early = False
            early_reason = ""
            if self.convergence_accelerator:
                should_stop_early, early_reason = self.convergence_accelerator.should_stop_early(
                    best_fitness, generation + 1, self.config.generations
                )
            
            if is_converged:
                self.logger.log_convergence(generation + 1, reason)
                break
            elif should_stop_early:
                self.logger.log_convergence(generation + 1, f"Early stop: {early_reason}")
                break
                
            self.logger.info("-" * 100)
        
        # Finalize run
        self.compressor.erase_temp_files()
        
        # Generate comprehensive statistics including optimization metrics
        component_stats = {
            'parameter_encoder': self.parameter_encoder.stats,
            'population_manager': self.population_manager.get_statistics(),
            'selection_methods': self.selection_methods.get_statistics(),
            'genetic_operations': self.genetic_operations.get_statistics(),
            'evaluation_engine': self.evaluation_engine.get_statistics(),
            'duplicate_prevention': self.duplicate_prevention.get_statistics(),
            'convergence_detector': self.convergence_detector.get_statistics()
        }
        
        # Add optimization statistics if components are enabled
        if self.adaptive_tuning:
            component_stats['adaptive_tuning'] = self.adaptive_tuning.get_adaptation_stats()
        if self.convergence_accelerator:
            component_stats['convergence_accelerator'] = self.convergence_accelerator.get_acceleration_stats()
        if self.performance_monitor:
            component_stats['performance_monitor'] = self.performance_monitor.get_performance_summary()
        
        # Save run summary and best parameters
        convergence_info = {
            'converged': is_converged,
            'convergence_reason': reason,
            'final_generation': generation + 1 if generation > 0 else 1
        }
        
        self.reporter.save_run_summary(decoded_best, convergence_info)
        self.reporter.save_best_parameters_csv(self.population_manager)
        self.reporter.export_fitness_history()
        
        # Print performance summary
        performance_summary = self.reporter.create_performance_summary(component_stats)
        self.logger.info("Performance Summary")
        self.logger.info(f"Total runtime: {performance_summary['experiment_info']['total_runtime']:.2f}s")
        self.logger.info(f"Generations completed: {performance_summary['experiment_info']['generations_completed']}")
        if 'avg_time_per_generation' in performance_summary['performance_metrics']:
            self.logger.info(f"Avg time per generation: {performance_summary['performance_metrics']['avg_time_per_generation']:.2f}s")
        
        # Log cache statistics
        cache = get_global_cache()
        try:
            # Get cache stats without using print_stats
            if hasattr(cache, 'hits') and hasattr(cache, 'misses'):
                total = cache.hits + cache.misses
                hit_rate = cache.hits / total if total > 0 else 0.0
                self.logger.log_cache_stats(cache.hits, cache.misses, hit_rate)
            else:
                cache.print_stats()  # Fallback to original method
        except Exception as e:
            self.logger.warning("Could not log cache statistics", exception=e)
        
        # Cleanup
        self.reporter.cleanup()
        
        time.sleep(1)
        return decoded_best, best_fitness
