"""
Genetic Operations Module

Core genetic algorithm operations including crossover and mutation.
These are the fundamental building blocks that drive evolutionary search.

Features:
- Uniform crossover with configurable rates
- Bit-flip mutation with adaptive rates
- Parameter-aware genetic operations
- Performance optimized implementations
"""

import random
from typing import Dict, List, Tuple, Any


class GeneticOperations:
    """
    Core genetic operations for evolutionary algorithms.
    
    Handles crossover and mutation operations with support for
    binary-encoded individuals and configurable parameters.
    """
    
    def __init__(self, crossover_rate: float, mutation_rate: float):
        """
        Initialize genetic operations.
        
        Args:
            crossover_rate: Probability of crossover occurring
            mutation_rate: Probability of each bit being mutated
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Statistics tracking
        self.crossover_count = 0
        self.mutation_count = 0
    
    def crossover(self, parent1: Tuple[str, ...], parent2: Tuple[str, ...], 
                  generation: int, param_binary_encodings: Dict[str, Dict[str, Any]], 
                  individual_name_generator) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """
        Perform uniform crossover between two parents.
        
        Args:
            parent1: First parent's gene sequence
            parent2: Second parent's gene sequence
            generation: Current generation (for offspring naming)
            param_binary_encodings: Parameter encoding information
            individual_name_generator: Function to generate individual names
            
        Returns:
            Tuple of two offspring gene sequences
        """
        if random.random() >= self.crossover_rate:
            # No crossover - return parents as-is
            return parent1, parent2
            
        # Calculate parameter boundaries for crossover
        param_boundaries = self._get_parameter_boundaries(param_binary_encodings)
        
        child1_genes = []
        child2_genes = []
        
        # Process each gene (model) separately
        for gene_idx in range(len(parent1)):
            parent1_gene = parent1[gene_idx]
            parent2_gene = parent2[gene_idx]
            
            child1_gene, child2_gene = self._crossover_genes(
                parent1_gene, parent2_gene, param_boundaries)
            
            child1_genes.append(child1_gene)
            child2_genes.append(child2_gene)
        
        self.crossover_count += 1
        
        return tuple(child1_genes), tuple(child2_genes)
    
    def _get_parameter_boundaries(self, param_binary_encodings: Dict[str, Dict[str, Any]]) -> List[int]:
        """
        Calculate parameter boundaries for crossover operations.
        
        Args:
            param_binary_encodings: Parameter encoding information
            
        Returns:
            List of bit positions where parameters end
        """
        boundaries = []
        current_pos = 0
        
        for param, encodings in param_binary_encodings.items():
            current_pos += encodings['bit_length']
            boundaries.append(current_pos)
        
        return boundaries
    
    def _crossover_genes(self, parent1_gene: str, parent2_gene: str, 
                        param_boundaries: List[int]) -> Tuple[str, str]:
        """
        Perform uniform crossover on individual genes respecting parameter boundaries.
        
        Args:
            parent1_gene: First parent's gene string
            parent2_gene: Second parent's gene string
            param_boundaries: Parameter boundary positions
            
        Returns:
            Tuple of two crossed-over gene strings
        """
        child1_gene = ""
        child2_gene = ""
        
        current_pos = 0
        
        # Process each parameter separately to maintain parameter integrity
        for boundary in param_boundaries:
            # Extract parameter bits
            parent1_param = parent1_gene[current_pos:boundary]
            parent2_param = parent2_gene[current_pos:boundary]
            
            # Uniform crossover: randomly choose which parent contributes each parameter
            if random.random() < 0.5:
                child1_gene += parent1_param
                child2_gene += parent2_param
            else:
                child1_gene += parent2_param
                child2_gene += parent1_param
            
            current_pos = boundary
        
        # Validate gene lengths before returning
        expected_length = len(parent1_gene)
        if len(child1_gene) != expected_length:
            raise ValueError(f"Crossover error: Child1 gene length {len(child1_gene)} != parent length {expected_length}")
        if len(child2_gene) != expected_length:
            raise ValueError(f"Crossover error: Child2 gene length {len(child2_gene)} != parent length {expected_length}")
        
        return child1_gene, child2_gene
    
    def mutate(self, individual: Tuple[Tuple[str, ...], str], generation: int, 
              individual_name_generator) -> Tuple[Tuple[str, ...], str]:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate (gene_code, name)
            generation: Current generation
            individual_name_generator: Function to generate new names
            
        Returns:
            Mutated individual
        """
        gene_code, name = individual
        mutated_gene_code = []
        gene_code_mutated = False
        
        for gene in gene_code:
            mutated_gene = ""
            
            # Apply bit-flip mutation to each bit
            for bit in gene:
                if random.random() < self.mutation_rate:
                    # Flip the bit
                    mutated_gene += '1' if bit == '0' else '0'
                    gene_code_mutated = True
                else:
                    # Keep original bit
                    mutated_gene += bit
            
            mutated_gene_code.append(mutated_gene)
        
        if gene_code_mutated:
            self.mutation_count += 1
            # Create new name for mutated individual
            new_name = individual_name_generator(generation)
            return (tuple(mutated_gene_code), new_name)
        else:
            # No mutation occurred - return original
            return individual
    
    def adaptive_mutation(self, individual: Tuple[Tuple[str, ...], str], generation: int,
                         total_generations: int, individual_name_generator, 
                         base_rate: float = None) -> Tuple[Tuple[str, ...], str]:
        """
        Apply adaptive mutation that changes rate based on generation progress.
        
        Args:
            individual: Individual to mutate
            generation: Current generation
            total_generations: Total planned generations
            individual_name_generator: Function to generate names
            base_rate: Base mutation rate (uses instance rate if None)
            
        Returns:
            Mutated individual
        """
        if base_rate is None:
            base_rate = self.mutation_rate
        
        # Adaptive mutation: higher rate early, lower rate late
        generation_progress = generation / total_generations
        adaptive_rate = base_rate * (2.0 - generation_progress)  # 2x early, 1x late
        adaptive_rate = max(0.001, min(0.1, adaptive_rate))  # Clamp between 0.1% and 10%
        
        # Temporarily adjust rate
        original_rate = self.mutation_rate
        self.mutation_rate = adaptive_rate
        
        result = self.mutate(individual, generation, individual_name_generator)
        
        # Restore original rate
        self.mutation_rate = original_rate
        
        return result
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about genetic operations performed."""
        return {
            'crossover_count': self.crossover_count,
            'mutation_count': self.mutation_count
        }
    
    def reset_statistics(self):
        """Reset operation counters."""
        self.crossover_count = 0
        self.mutation_count = 0
    
    def update_rates(self, crossover_rate: float = None, mutation_rate: float = None):
        """
        Update operation rates during evolution.
        
        Args:
            crossover_rate: New crossover rate (if provided)
            mutation_rate: New mutation rate (if provided)
        """
        if crossover_rate is not None:
            self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        if mutation_rate is not None:
            self.mutation_rate = max(0.0, min(1.0, mutation_rate))