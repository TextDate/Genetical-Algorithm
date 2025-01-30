import random
import csv
import os
import concurrent.futures
import time
from functools import lru_cache
import psutil
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(self, param_values, compressor, population_size, generations, mutation_rate, crossover_rate,
                 output_dir="ga_results_all_files", max_threads=8):
        self.param_values = param_values  # Dictionary of parameter names to their valid values
        self.compressor = compressor  # Compressor the GA is trying to optimize
        self.population_size = population_size  # Population size
        self.num_offspring = int(self.population_size * 0.9)  # Number of offspring to generate (90% of population size)
        self.generations = generations  # Number of generations
        self.mutation_rate = mutation_rate  # Probability of mutation
        self.crossover_rate = crossover_rate  # Probability of crossover
        self.output_dir = output_dir  # Output directory for CSV files
        self.max_threads = max_threads

        #self.num_workers = psutil.cpu_count(logical=False) // 2

        self.mutation_counts = []  # Track mutations per generation
        self.crossover_counts = []  # Track crossovers per generation
        self.individual_counter = 0  # Counter for assigning unique individual names

        self.param_binary_encodings = self._encode_parameters()  # Encode params to binary

        self.population = self._initialize_population()  # Initialize population

        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory

    def _encode_parameters(self):
        """Encodes parameters to binary."""
        param_encodings = {}
        for param, values in self.param_values.items():
            num_bits = len(bin(len(values) - 1)[2:])  # Number of bits required
            binary_representations = {i: format(i, f'0{num_bits}b') for i in range(len(values))}
            param_encodings[param] = {
                'values': values,
                'binary_map': binary_representations,
                'bit_length': num_bits
            }
        return param_encodings

    def _decode_individual(self, individual):
        """Decodes binary gene to parameter values."""
        gene_code = individual[0]
        decoded_individual = []
        for gene in gene_code:
            decoded_gene = {}
            current_pos = 0
            for param, encodings in self.param_binary_encodings.items():
                bit_length = encodings['bit_length']
                binary_value = gene[current_pos:current_pos + bit_length]
                value_idx = int(binary_value, 2)
                if value_idx >= len(encodings['values']):  # Ensure valid index
                    value_idx = random.randint(0, len(encodings['values']) - 1)  # Correct the index if invalid
                decoded_gene[param] = encodings['values'][value_idx]
                current_pos += bit_length
            decoded_individual.append(decoded_gene)
        return tuple(decoded_individual)

    def _initialize_population(self):
        """Initializes population with binary genes ensuring uniqueness."""
        population = []
        while len(population) != self.population_size:
            gene_code = []
            while len(gene_code) != self.compressor.nr_models:
                gene = ''
                for param, encodings in self.param_binary_encodings.items():
                    value_idx = random.randint(0, len(encodings['values']) - 1)
                    gene += encodings['binary_map'][value_idx]
                if gene not in gene_code:
                    gene_code.append(gene)

            if gene_code not in population:  # rever se isto está bem, provavelmente não
                population.append((tuple(gene_code), self._create_individual_name(1)))

        return population

    def _create_individual_name(self, generation):
        """Create a unique name for an individual based on the generation and order of creation."""
        self.individual_counter += 1
        return f"Gen{generation}_Ind{self.individual_counter}"

    @lru_cache(maxsize=None)
    def _evaluate_fitness(self, individual):
        """Evaluates fitness of an individual."""
        decoded_individual = self._decode_individual(individual)
        return self.compressor.evaluate(decoded_individual, individual[1])

    def _evaluate_population_in_parallel(self, population):
        """Evaluates the fitness of the entire population in parallel with tqdm progress bar."""
        fitness_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            # Use tqdm to wrap the executor's map
            for fitness in tqdm(executor.map(self._evaluate_fitness, population), total=len(population),
                                desc="Evaluating Population"):
                fitness_results.append(fitness)
        return fitness_results

    def _select_parents(self):
        """Selects parents using tournament selection."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0], tournament[1]

    def _crossover(self, parent1, parent2, generation):
        """Performs crossover between two parents at the parameter level, ensuring parameters change when possible."""

        if random.random() < self.crossover_rate:
            num_params = len(self.param_binary_encodings)
            param_boundaries = []

            current_pos = 0
            for param, encodings in self.param_binary_encodings.items():
                bit_length = encodings['bit_length']
                param_boundaries.append((current_pos, current_pos + bit_length))
                current_pos += bit_length

            child1_gene_code = []
            child2_gene_code = []
            swapped = False

            for gene_index in range(len(parent1[0])):  # Iterate over each gene
                #print("Parent1:", parent1)
                #print("Parent2", parent2)

                parent1_gene = parent1[0][gene_index]
                parent2_gene = parent2[0][gene_index]

                # Ensure parent genes are strings for slicing
                parent1_gene_str = ''.join(parent1_gene)
                parent2_gene_str = ''.join(parent2_gene)

                child1_gene = ''
                child2_gene = ''

                # Go through each parameter within the gene
                for start, end in param_boundaries:
                    parent1_param = parent1_gene_str[start:end]
                    parent2_param = parent2_gene_str[start:end]

                    if parent1_param != parent2_param:
                        if random.random() < self.crossover_rate:
                            child1_gene += parent2_param
                            child2_gene += parent1_param
                            swapped = True
                        else:
                            child1_gene += parent1_param
                            child2_gene += parent2_param
                    else:
                        child1_gene += parent1_param
                        child2_gene += parent2_param

                if not swapped:
                    crossover_point = random.randint(0, num_params - 1)
                    start, end = param_boundaries[crossover_point]
                    child1_gene = child1_gene[:start] + parent2_gene_str[start:end] + child1_gene[end:]
                    child2_gene = child2_gene[:start] + parent1_gene_str[start:end] + child2_gene[end:]
                    self.crossover_counts[-1] += 1

                child1_gene_code.append(child1_gene)
                child2_gene_code.append(child2_gene)

            #print("Child 1: ", child1_gene_code)
            #print("Child 2: ", child2_gene_code)
            child1 = (tuple(child1_gene_code), self._create_individual_name(generation))
            child2 = (tuple(child2_gene_code), self._create_individual_name(generation))

            return child1, child2
        else:
            return parent1, parent2

    def _mutate(self, individual, generation):
        """Applies mutation to an individual."""
        # Gene Code (each in individual can have n models) and name are the elements of the tuple
        #print("Individual to mutate:", individual)
        gene_code, name = individual
        mutated_gene_code = []
        gene_code_mutated = False

        # Applying mutation to each gene in the Gene Code
        for gene in gene_code:
            mutated = False
            gene_list = list(gene)
            for i in range(len(gene_list)):
                if random.random() < self.mutation_rate:
                    self.mutation_counts[-1] += 1
                    # Bit flipping to apply mutation
                    gene_list[i] = '1' if gene_list[i] == '0' else '0'
                    mutated = True
                    gene_code_mutated = True
            mutated_gene = ''.join(gene_list)

            if mutated:
                mutated_gene_code.append(mutated_gene)
            else:
                mutated_gene_code.append(gene)

        if gene_code_mutated:
            return tuple(mutated_gene_code), self._create_individual_name(generation)
        else:
            return individual

    def _select_next_generation(self, offspring):
        """Selects the next generation of individuals using elitism."""

        # Calculate the number of elites to preserve (10% of the original population)
        num_elites = int(0.1 * self.population_size)

        # Keep the top 10% of the current generation
        elites = self.population[:num_elites]

        # Sort the offspring by fitness, from best to worst
        offspring_with_fitness = [
            (individual, fitness) for individual, fitness in
            zip(offspring, self._evaluate_population_in_parallel(offspring))
        ]
        sorted_offspring = sorted(offspring_with_fitness, key=lambda x: x[1], reverse=True)

        # Fill the rest of the population with the best offspring
        remaining_population_size = self.population_size - num_elites
        top_offspring = sorted_offspring[:remaining_population_size]

        self.population = elites + top_offspring
        self.population = sorted(self.population, key=lambda x: x[1], reverse=True)

    def _save_generation_to_csv(self, generation):
        """Saves the current generation's population to a CSV file."""
        filename = os.path.join(self.output_dir, f"generation_{generation}.csv")

        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            header = ["Rank", "Individual"]

            for n in range(self.compressor.nr_models):
                header.append(f"Gene{n + 1}")
                header += [f"{param}_Gene{n + 1}" for param in self.param_values.keys()]

            header.append("Fitness")
            writer.writerow(header)

            # Write each individual, with its rank and fitness
            for rank, (individual, fitness) in enumerate(self.population, start=1):
                decoded_gene_code = self._decode_individual(individual)
                row = [rank, individual[1]]

                for gene_index, gene in enumerate(decoded_gene_code):
                    row.append(individual[0][gene_index])  # Add the gene binary code
                    row += [value for value in gene.values()]  # Add the parameter values for this gene

                # Append the fitness score
                row.append(fitness)
                writer.writerow(row)

    def run(self):
        """Runs the genetic algorithm."""
        init_time = time.time()
        print(f"Initial Generation")
        # The initial generation will never have mutation or crossovers
        self.mutation_counts.append(0)
        self.crossover_counts.append(0)

        # Evaluate fitness for each individual and store it with the individual
        fit_time = time.time()
        population_with_fitness = [
            (individual, fitness) for individual, fitness in
            zip(self.population, self._evaluate_population_in_parallel(self.population))
        ]
        print(f"Took {time.time() - fit_time} to evaluate Initial Generation")

        # Sort the population by fitness, best to worst
        self.population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

        # Save the initial generation on CSV
        self._save_generation_to_csv(1)

        # Display the best individual from the initial generation
        best_individual = self.population[0][0]
        decoded_best_individual = self._decode_individual(best_individual)
        print(f"Best individual from Initial Generation: {decoded_best_individual}")
        best_fitness = self.population[0][1]
        print(f"Fitness: {best_fitness}")
        print(f"Total time took: {time.time() - init_time}")
        # Delete the temporary files created by the initial generation
        self.compressor.erase_temp_files()

        # Start the process for the next generations
        for generation in range(1, self.generations):
            init_time = time.time()
            print(f"Generation {generation + 1}")
            # Reset mutation and crossover for each generation
            self.mutation_counts.append(0)
            self.crossover_counts.append(0)

            # Generate offspring
            offspring = []
            # Divided by two because each iteration of the for cycle creates two new individuals
            offspring_gen_time = time.time()
            for i in range(self.num_offspring // 2):
                parent1, parent2 = self._select_parents()
                child1, child2 = self._crossover(parent1[0], parent2[0], generation + 1)
                # Applying mutation to the newly created individuals
                child1 = self._mutate(child1, generation + 1)
                child2 = self._mutate(child2, generation + 1)
                # Adding the new individuals to the offspring list
                offspring.extend([child1, child2])
            print(f"Took {time.time() - offspring_gen_time} to create offspring from Generation {generation}")

            # Select the next generation
            fit_time = time.time()
            self._select_next_generation(offspring)
            print(f"Took {time.time() - fit_time} to evaluate Generation {generation}")

            # Save the generation to a CSV file
            self._save_generation_to_csv(generation + 1)

            # Display the best individual so far
            best_individual = self.population[0][0]
            decoded_best_individual = self._decode_individual(best_individual)
            print(f"Best individual: {decoded_best_individual}")
            best_fitness = self.population[0][1]
            print(f"Fitness: {best_fitness}")
            print(f"Total time took: {time.time() - init_time}")

            # Erase the temporary files created this generation
            self.compressor.erase_temp_files()

        return decoded_best_individual, best_fitness
