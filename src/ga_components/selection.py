"""
Selection Methods Module

Implements various selection strategies for genetic algorithms including
parent selection and survivor selection with elitism support.

Features:
- Tournament selection for parent selection
- Elitist survivor selection
- Fitness-proportionate selection
- Rank-based selection
- Configurable selection pressure
"""

import random
from typing import List, Tuple, Any


class SelectionMethods:
    """
    Collection of selection methods for genetic algorithms.
    
    Provides both parent selection (for reproduction) and survivor selection
    (for next generation formation) with various strategies and elitism support.
    """
    
    def __init__(self, tournament_size: int = 3, elite_ratio: float = 0.1):
        """
        Initialize selection methods.
        
        Args:
            tournament_size: Number of individuals in tournament selection
            elite_ratio: Proportion of population to preserve as elites
        """
        self.tournament_size = tournament_size
        self.elite_ratio = elite_ratio
        
        # Statistics tracking
        self.selection_stats = {
            'tournaments_held': 0,
            'elites_preserved': 0
        }
    
    def tournament_selection(self, population: List[Tuple[Any, float]], 
                           k: int = None) -> Tuple[Any, float]:
        """
        Select an individual using tournament selection.
        
        Args:
            population: Population with fitness values [(individual, fitness), ...]
            k: Tournament size (uses instance default if None)
            
        Returns:
            Selected individual and its fitness
        """
        if k is None:
            k = self.tournament_size
        
        # Ensure tournament size doesn't exceed population size
        k = min(k, len(population))
        
        # Select random individuals for tournament
        tournament_candidates = random.sample(population, k)
        
        # Select best individual from tournament (highest fitness)
        winner = max(tournament_candidates, key=lambda x: x[1])
        
        self.selection_stats['tournaments_held'] += 1
        
        return winner
    
    def select_parents(self, population: List[Tuple[Any, float]], 
                      num_pairs: int) -> List[Tuple[Tuple[Any, float], Tuple[Any, float]]]:
        """
        Select parent pairs for reproduction using tournament selection.
        
        Args:
            population: Population with fitness values
            num_pairs: Number of parent pairs to select
            
        Returns:
            List of parent pairs
        """
        parent_pairs = []
        
        for _ in range(num_pairs):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Ensure parents are different (if population size allows)
            attempts = 0
            while parent1[0] == parent2[0] and attempts < 10 and len(population) > 1:
                parent2 = self.tournament_selection(population)
                attempts += 1
            
            parent_pairs.append((parent1, parent2))
        
        return parent_pairs
    
    def elitist_selection(self, population: List[Tuple[Any, float]], 
                         offspring: List[Tuple[Any, float]], 
                         target_size: int) -> List[Tuple[Any, float]]:
        """
        Select next generation using elitist strategy.
        
        Preserves the best individuals from current population and fills
        the rest with the best offspring.
        
        Args:
            population: Current population with fitness
            offspring: Offspring population with fitness  
            target_size: Desired size of next generation
            
        Returns:
            Next generation population
        """
        # Calculate number of elites to preserve
        num_elites = int(self.elite_ratio * target_size)
        num_elites = max(0, min(num_elites, len(population)))  # Clamp to valid range
        
        # Sort current population by fitness (descending)
        sorted_population = sorted(population, key=lambda x: x[1], reverse=True)
        
        # Select elites from current population
        elites = sorted_population[:num_elites]
        self.selection_stats['elites_preserved'] += len(elites)
        
        # Fill remaining slots with best offspring
        remaining_slots = target_size - num_elites
        
        if remaining_slots > 0:
            # Sort offspring by fitness (descending)
            sorted_offspring = sorted(offspring, key=lambda x: x[1], reverse=True)
            
            # Select best offspring to fill remaining slots
            selected_offspring = sorted_offspring[:remaining_slots]
            
            next_generation = elites + selected_offspring
        else:
            # All slots filled by elites
            next_generation = elites[:target_size]
        
        # Sort final population by fitness (descending)
        next_generation.sort(key=lambda x: x[1], reverse=True)
        
        return next_generation
    
    def fitness_proportionate_selection(self, population: List[Tuple[Any, float]]) -> Tuple[Any, float]:
        """
        Select individual using fitness-proportionate selection (roulette wheel).
        
        Args:
            population: Population with fitness values
            
        Returns:
            Selected individual and fitness
        """
        # Calculate total fitness
        total_fitness = sum(fitness for _, fitness in population)
        
        if total_fitness <= 0:
            # If all fitness values are non-positive, use uniform random selection
            return random.choice(population)
        
        # Generate random number between 0 and total_fitness
        selection_point = random.uniform(0, total_fitness)
        
        # Find individual at selection point
        cumulative_fitness = 0
        for individual, fitness in population:
            cumulative_fitness += fitness
            if cumulative_fitness >= selection_point:
                return (individual, fitness)
        
        # Fallback (shouldn't reach here)
        return population[-1]
    
    def rank_based_selection(self, population: List[Tuple[Any, float]]) -> Tuple[Any, float]:
        """
        Select individual using rank-based selection.
        
        Args:
            population: Population with fitness values
            
        Returns:
            Selected individual and fitness
        """
        # Sort population by fitness (ascending for rank assignment)
        sorted_pop = sorted(population, key=lambda x: x[1])
        
        # Assign ranks (1 = worst, n = best)
        n = len(sorted_pop)
        ranks = list(range(1, n + 1))
        
        # Use ranks for proportionate selection
        total_rank = sum(ranks)
        selection_point = random.uniform(0, total_rank)
        
        cumulative_rank = 0
        for i, (individual, fitness) in enumerate(sorted_pop):
            cumulative_rank += ranks[i]
            if cumulative_rank >= selection_point:
                return (individual, fitness)
        
        # Fallback
        return sorted_pop[-1]
    
    def adaptive_tournament_size(self, generation: int, total_generations: int, 
                                min_size: int = 2, max_size: int = 5) -> int:
        """
        Calculate adaptive tournament size based on generation progress.
        
        Args:
            generation: Current generation
            total_generations: Total planned generations
            min_size: Minimum tournament size
            max_size: Maximum tournament size
            
        Returns:
            Adaptive tournament size
        """
        progress = generation / total_generations
        
        # Start with larger tournaments (higher selection pressure) early,
        # reduce tournament size later to allow more diversity
        size = max_size - (progress * (max_size - min_size))
        
        return max(min_size, min(max_size, int(round(size))))
    
    def diversity_selection(self, population: List[Tuple[Any, float]], 
                           diversity_func, num_select: int) -> List[Tuple[Any, float]]:
        """
        Select individuals based on both fitness and diversity.
        
        Args:
            population: Population with fitness values
            diversity_func: Function to calculate diversity between individuals
            num_select: Number of individuals to select
            
        Returns:
            Selected diverse individuals
        """
        if num_select >= len(population):
            return population[:]
        
        selected = []
        remaining = population[:]
        
        # Select first individual (best fitness)
        best = max(remaining, key=lambda x: x[1])
        selected.append(best)
        remaining.remove(best)
        
        # Select remaining individuals balancing fitness and diversity
        for _ in range(num_select - 1):
            if not remaining:
                break
            
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in remaining:
                # Calculate fitness component (normalized)
                fitness_score = candidate[1]
                
                # Calculate diversity component (minimum distance to selected)
                min_diversity = float('inf')
                for selected_ind, _ in selected:
                    diversity = diversity_func(candidate[0], selected_ind)
                    min_diversity = min(min_diversity, diversity)
                
                # Combined score (balance fitness and diversity)
                combined_score = fitness_score + min_diversity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def get_statistics(self) -> dict:
        """Get selection statistics."""
        return self.selection_stats.copy()
    
    def reset_statistics(self):
        """Reset selection statistics."""
        self.selection_stats = {
            'tournaments_held': 0,
            'elites_preserved': 0
        }
    
    def update_parameters(self, tournament_size: int = None, elite_ratio: float = None):
        """
        Update selection parameters during evolution.
        
        Args:
            tournament_size: New tournament size
            elite_ratio: New elite ratio
        """
        if tournament_size is not None:
            self.tournament_size = max(2, tournament_size)
        if elite_ratio is not None:
            self.elite_ratio = max(0.0, min(1.0, elite_ratio))