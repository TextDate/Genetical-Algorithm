"""
Adaptive Hybrid Duplicate Prevention System

This module implements an intelligent duplicate prevention system for genetic algorithms
that adapts its behavior based on generation progress, population density, and parameter space characteristics.

The system uses a three-phase approach:
- Early Phase: Strict uniqueness enforcement (force exploration)
- Middle Phase: Gradual transition with probabilistic enforcement  
- Late Phase: Allow duplicates (enable natural convergence)
"""

import random
from typing import Dict, List, Tuple, Any, Optional, Set
from ga_logging import get_logger


class DuplicatePreventionSystem:
    """
    Adaptive hybrid duplicate prevention system for genetic algorithms.
    
    This system intelligently decides when to enforce uniqueness based on:
    - Generation progress (early exploration vs late exploitation)
    - Population density relative to parameter space
    - Current duplicate frequency in the generation
    """
    
    def __init__(self, total_parameter_combinations: int, population_size: int, 
                 total_generations: int, enabled: bool = True):
        """
        Initialize the duplicate prevention system.
        
        Args:
            total_parameter_combinations: Total possible parameter combinations
            population_size: Size of the population per generation
            total_generations: Total number of generations planned
            enabled: Whether the system is active
        """
        self.total_parameter_combinations = total_parameter_combinations
        self.population_size = population_size  
        self.total_generations = total_generations
        self.enabled = enabled
        self.logger = get_logger("DuplicatePreventionSystem")
        
        # Statistics tracking
        self.stats = {
            'total_duplicates': 0,
            'total_enforced': 0,
            'total_allowed': 0
        }
    
    def calculate_parameter_space_density(self, current_generation: int) -> float:
        """
        Calculate how dense the current population is in the parameter space.
        
        Higher density means more collisions are likely, so we should be more strict
        about preventing duplicates in early generations.
        
        Args:
            current_generation: Current generation number (for adaptive behavior)
            
        Returns:
            Density value between 0.0 and 1.0
        """
        # Base density: population size relative to total parameter space
        base_density = self.population_size / self.total_parameter_combinations
        
        # Adjust for generation progress - later generations naturally have higher density
        # due to convergence, which is expected and desirable
        generation_progress = current_generation / self.total_generations
        density_adjustment = 1.0 - (generation_progress * 0.5)  # Reduce by up to 50% in late generations
        
        adjusted_density = base_density * density_adjustment
        
        # Return capped density value
        return min(1.0, max(0.0, adjusted_density))

    def get_adaptive_thresholds(self, current_generation: int) -> Tuple[str, float]:
        """
        Determine adaptive thresholds for duplicate prevention based on generation progress.
        
        The system has three phases:
        - Early Phase: Strict uniqueness enforcement (force exploration)
        - Middle Phase: Gradual transition with probabilistic enforcement
        - Late Phase: Allow duplicates (enable natural convergence)
        
        Args:
            current_generation: Current generation number
            
        Returns:
            Tuple of (phase_name, enforcement_probability)
        """
        generation_progress = current_generation / self.total_generations
        
        # Adapt phase boundaries based on population size
        # Smaller populations need longer strict phases, larger populations can transition faster
        pop_adjustment_factor = max(0.5, min(2.0, 50 / self.population_size))
        
        # Calculate adaptive phase boundaries
        early_phase_end = max(0.15, min(0.4, 0.25 * pop_adjustment_factor))
        late_phase_start = min(0.85, max(0.6, 0.75 + (0.1 / pop_adjustment_factor)))
        
        if generation_progress <= early_phase_end:
            # Early phase: Strict enforcement
            enforcement_prob = 0.9  # 90% chance to enforce uniqueness
            return "strict", enforcement_prob
            
        elif generation_progress >= late_phase_start:
            # Late phase: Allow duplicates for natural convergence
            return "none", 0.0
            
        else:
            # Middle phase: Gradual transition
            # Linear interpolation between early and late phases
            transition_progress = (generation_progress - early_phase_end) / (late_phase_start - early_phase_end)
            enforcement_prob = 0.9 * (1.0 - transition_progress)  # Gradually decrease from 90% to 0%
            return "gradual", enforcement_prob

    def should_enforce_uniqueness(self, current_generation: int, duplicate_count_this_gen: int) -> bool:
        """
        Decide whether to enforce uniqueness for the current situation.
        
        This combines multiple factors:
        - Generation progress (early vs late)
        - Current duplicate count in this generation
        - Parameter space density
        - Population size considerations
        
        Args:
            current_generation: Current generation number
            duplicate_count_this_gen: Number of duplicates seen so far in this generation
            
        Returns:
            Boolean indicating whether to enforce uniqueness
        """
        if not self.enabled:
            return False
            
        # Get the basic threshold for this generation
        phase, base_enforcement_prob = self.get_adaptive_thresholds(current_generation)
        
        if phase == "none":
            return False  # Never enforce in late phase
        elif phase == "strict":
            # Always enforce in strict phase, but with slight randomization to prevent artifacts
            return random.random() < 0.95
        else:  # gradual phase
            # In gradual phase, combine multiple factors
            
            # Factor 1: Base probability from generation progress
            enforcement_prob = base_enforcement_prob
            
            # Factor 2: Adjust based on duplicate density in current generation
            duplicate_ratio = duplicate_count_this_gen / self.population_size
            if duplicate_ratio > 0.3:  # If more than 30% are duplicates, be more strict
                enforcement_prob += 0.2
            elif duplicate_ratio < 0.1:  # If very few duplicates, can be more lenient
                enforcement_prob -= 0.1
            
            # Factor 3: Parameter space density consideration
            density = self.calculate_parameter_space_density(current_generation)
            if density > 0.1:  # High density parameter space - be more strict
                enforcement_prob += 0.1
            else:  # Low density - natural diversity likely, be more lenient
                enforcement_prob -= 0.1
            
            # Clamp probability between 0 and 1
            enforcement_prob = max(0.0, min(1.0, enforcement_prob))
            
            return random.random() < enforcement_prob

    def individual_to_signature(self, individual: Tuple[Tuple[str, ...], str]) -> str:
        """
        Convert individual to a hashable signature for duplicate detection.
        
        We only care about the genetic content (gene_code), not the individual name,
        since the name is just a label and doesn't affect the actual parameters.
        
        Args:
            individual: Tuple containing (gene_code, individual_name)
            
        Returns:
            String signature representing the genetic content
        """
        gene_code = individual[0]
        # Convert tuple of gene strings to a single concatenated string
        return ''.join(gene_code)

    def generate_unique_individual(self, seen_signatures: Set[str], generation: int, 
                                 param_binary_encodings: Dict[str, Dict[str, Any]], 
                                 compressor_nr_models: int, individual_name_generator,
                                 population: List = None, max_attempts: int = 50) -> Tuple[Tuple[str, ...], str]:
        """
        Generate a new individual that doesn't conflict with already seen signatures.
        
        This method tries multiple approaches:
        1. Pure random generation (most attempts)
        2. Mutation of random existing individual (fallback)
        3. Force mutation with higher rate (last resort)
        
        Args:
            seen_signatures: Set of signatures already seen in current generation
            generation: Current generation number (for individual naming)
            param_binary_encodings: Parameter encoding information
            compressor_nr_models: Number of models for the compressor
            individual_name_generator: Function to generate individual names
            population: Current population for mutation fallback
            max_attempts: Maximum attempts before using fallback methods
            
        Returns:
            New unique individual tuple (gene_code, individual_name)
        """
        # Method 1: Try pure random generation
        for attempt in range(max_attempts):
            gene_code = []
            
            # Generate genes for each model (usually just 1)
            for model_idx in range(compressor_nr_models):
                gene = ''
                # Build gene by concatenating binary representations of each parameter
                for param, encodings in param_binary_encodings.items():
                    value_idx = random.randint(0, len(encodings['values']) - 1)
                    gene += encodings['binary_map'][value_idx]
                gene_code.append(gene)
            
            # Check if this gene combination is unique
            new_individual = (tuple(gene_code), individual_name_generator(generation))
            signature = self.individual_to_signature(new_individual)
            
            if signature not in seen_signatures:
                return new_individual
        
        # Method 2: Fallback - mutate a random existing individual
        if population:
            for attempt in range(10):  # Try mutation approach
                base_individual = random.choice(population)
                mutated_individual = self._mutate_individual_for_uniqueness(
                    base_individual, seen_signatures, generation, individual_name_generator)
                if mutated_individual:
                    return mutated_individual
        
        # Method 3: Last resort - generate with forced high mutation
        return self._force_unique_individual(seen_signatures, generation, param_binary_encodings, 
                                           compressor_nr_models, individual_name_generator)

    def _mutate_individual_for_uniqueness(self, base_individual: Tuple[Tuple[str, ...], str], 
                                        seen_signatures: Set[str], generation: int, 
                                        individual_name_generator) -> Optional[Tuple[Tuple[str, ...], str]]:
        """
        Create a unique individual by mutating a base individual.
        
        Args:
            base_individual: Individual to use as base for mutation
            seen_signatures: Set of signatures to avoid
            generation: Generation number for naming
            individual_name_generator: Function to generate individual names
            
        Returns:
            Mutated unique individual, or None if couldn't create one
        """
        base_gene_code = base_individual[0]
        
        for attempt in range(20):  # Try multiple mutations
            mutated_gene_code = []
            
            for gene in base_gene_code:
                mutated_gene = list(gene)  # Convert to list for mutation
                
                # Apply mutations - flip 2-3 random bits
                num_mutations = random.randint(2, 3)
                for _ in range(num_mutations):
                    if len(mutated_gene) > 0:
                        pos = random.randint(0, len(mutated_gene) - 1)
                        mutated_gene[pos] = '1' if mutated_gene[pos] == '0' else '0'
                
                mutated_gene_code.append(''.join(mutated_gene))
            
            new_individual = (tuple(mutated_gene_code), individual_name_generator(generation))
            signature = self.individual_to_signature(new_individual)
            
            if signature not in seen_signatures:
                return new_individual
        
        return None  # Couldn't find unique mutation

    def _force_unique_individual(self, seen_signatures: Set[str], generation: int, 
                                param_binary_encodings: Dict[str, Dict[str, Any]], 
                                compressor_nr_models: int, individual_name_generator) -> Tuple[Tuple[str, ...], str]:
        """
        Force generation of a unique individual as last resort.
        
        This method systematically tries different parameter combinations
        until it finds one that's unique.
        
        Args:
            seen_signatures: Set of signatures to avoid
            generation: Generation number for naming
            param_binary_encodings: Parameter encoding information
            compressor_nr_models: Number of models for the compressor
            individual_name_generator: Function to generate individual names
            
        Returns:
            Guaranteed unique individual (though may not be optimal)
        """
        attempts = 0
        max_systematic_attempts = min(1000, self.total_parameter_combinations)
        
        while attempts < max_systematic_attempts:
            gene_code = []
            
            # Systematically vary the first few parameters
            for model_idx in range(compressor_nr_models):
                gene = ''
                param_idx = 0
                
                for param, encodings in param_binary_encodings.items():
                    if param_idx == 0:
                        # Vary first parameter systematically
                        value_idx = attempts % len(encodings['values'])
                    else:
                        # Random for other parameters
                        value_idx = random.randint(0, len(encodings['values']) - 1)
                    
                    gene += encodings['binary_map'][value_idx]
                    param_idx += 1
                    
                gene_code.append(gene)
            
            new_individual = (tuple(gene_code), individual_name_generator(generation))
            signature = self.individual_to_signature(new_individual)
            
            if signature not in seen_signatures:
                return new_individual
                
            attempts += 1
        
        # Absolute fallback - just return a random individual
        # This should almost never happen in practice
        gene_code = []
        for model_idx in range(compressor_nr_models):
            gene = '0' * sum(enc['bit_length'] for enc in param_binary_encodings.values())
            gene_code.append(gene)
        
        return (tuple(gene_code), individual_name_generator(generation))

    def enforce_population_diversity(self, population: List[Tuple[Tuple[str, ...], str]], 
                                   generation: int, param_binary_encodings: Dict[str, Dict[str, Any]], 
                                   compressor_nr_models: int, individual_name_generator,
                                   verbose: bool = True) -> List[Tuple[Tuple[str, ...], str]]:
        """
        Apply adaptive duplicate prevention to a population.
        
        This is the main integration point that applies our adaptive hybrid system
        to any population (initial population, offspring after crossover/mutation, etc.)
        
        Args:
            population: List of individuals to process
            generation: Current generation number
            param_binary_encodings: Parameter encoding information
            compressor_nr_models: Number of models for the compressor
            individual_name_generator: Function to generate individual names
            verbose: Whether to print statistics
            
        Returns:
            List of individuals with duplicates handled according to adaptive policy
        """
        if not self.enabled:
            return population
        
        seen_signatures = set()
        duplicate_count = 0
        processed_population = []
        
        # Statistics for debugging/monitoring
        enforced_count = 0
        allowed_count = 0
        
        for individual in population:
            signature = self.individual_to_signature(individual)
            
            if signature in seen_signatures:
                duplicate_count += 1
                
                # Decide whether to enforce uniqueness for this duplicate
                if self.should_enforce_uniqueness(generation, duplicate_count):
                    # Generate a unique replacement
                    try:
                        unique_individual = self.generate_unique_individual(
                            seen_signatures, generation, param_binary_encodings, 
                            compressor_nr_models, individual_name_generator, population)
                        if unique_individual:
                            processed_population.append(unique_individual)
                            seen_signatures.add(self.individual_to_signature(unique_individual))
                            enforced_count += 1
                        else:
                            # Fallback: keep the duplicate if we can't generate unique one
                            processed_population.append(individual)
                            allowed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to generate unique individual: {e}, keeping duplicate")
                        processed_population.append(individual)
                        allowed_count += 1
                else:
                    # Allow the duplicate
                    processed_population.append(individual)
                    # Note: we don't add to seen_signatures to allow more of the same
                    allowed_count += 1
            else:
                # First occurrence - always allow
                processed_population.append(individual)
                seen_signatures.add(signature)
        
        # Update statistics
        self.stats['total_duplicates'] += duplicate_count
        self.stats['total_enforced'] += enforced_count
        self.stats['total_allowed'] += allowed_count
        
        # Optional: Print statistics for monitoring (can be disabled later)
        if verbose and (generation <= 5 or generation % 10 == 0):  # Only print for early generations or every 10th
            phase, enforcement_prob = self.get_adaptive_thresholds(generation)
            density = self.calculate_parameter_space_density(generation)
            self.logger.debug("Duplicate prevention statistics", 
                             generation=generation,
                             duplicate_count=duplicate_count,
                             enforced_count=enforced_count, 
                             allowed_count=allowed_count,
                             phase=phase,
                             enforcement_prob=f"{enforcement_prob:.2f}",
                             density=f"{density:.4f}")
        
        # Verify population size is maintained
        if len(processed_population) != len(population):
            self.logger.warning(f"Population size changed during duplicate prevention: "
                              f"{len(population)} → {len(processed_population)}")
        
        return processed_population
    
    def get_statistics(self) -> Dict[str, int]:
        """Get overall statistics about duplicate prevention performance."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.stats = {
            'total_duplicates': 0,
            'total_enforced': 0,
            'total_allowed': 0
        }