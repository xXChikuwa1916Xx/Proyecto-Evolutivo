import random
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

class MagicSquare:
    def __init__(self, size):
        self.size = size
        self.values = [[-1 for _ in range(size)] for _ in range(size)]
        self.used_numbers = set()
        self.available_numbers = list(range(1, self.size * self.size + 1))

    def set_value(self, row, col, value):
        available_values = set(self.available_numbers)
        available_values.difference_update(self.used_numbers)
        if not available_values:
            value = self.used_numbers.pop()
        elif value in self.used_numbers:
            value = available_values.pop()
        self.used_numbers.add(value)
        self.values[row][col] = value

    def get_value(self, row, col):
        return self.values[row][col]

    def get_size(self):
        return self.size

    def get_magic_constant(self):
        return (self.size * (self.size * self.size + 1)) // 2

    def get_row_sum(self, row):
        return sum(self.values[row])

    def get_col_sum(self, col):
        return sum(self.values[i][col] for i in range(self.size))

    def get_diag_sum(self, is_main_diag):
        if is_main_diag:
            return sum(self.values[i][i] for i in range(self.size))
        else:
            return sum(self.values[i][self.size - i - 1] for i in range(self.size))

    def get_fitness(self):
        fitness = 0
        for i in range(self.size):
            fitness += abs(self.get_magic_constant() - self.get_row_sum(i))
            fitness += abs(self.get_magic_constant() - self.get_col_sum(i))
        fitness += abs(self.get_magic_constant() - self.get_diag_sum(True))
        fitness += abs(self.get_magic_constant() - self.get_diag_sum(False))
        return fitness

    def get_fitness_percentual(self):
        return 100 * (1 - self.get_fitness() / (self.size * self.size * self.size))
 

    def get_fitness_percentuales(self):
        return self.get_fitness() / (self.size + self.size + 2)

    def __str__(self):
        s = ""
        for i in range(self.size):
            row_sum = self.get_row_sum(i)
            for j in range(self.size):
                s += f"{self.values[i][j]:2d} "
            s += f"= {row_sum}\n"
        for i in range(self.size):
            col_sum = self.get_col_sum(i)
            s += f"= {col_sum} "
        s += "\n"
        s += f"Diagonal principal: = {self.get_diag_sum(True)}\n"
        s += f"Diagonal secundaria: = {self.get_diag_sum(False)}\n"
        s += f"Constante mágica: {self.get_magic_constant()}\n"
        return s

class GeneticAlgorithm:
    MUTATION_RATE = 0.02
    TOURNAMENT_SIZE = 3
    
    def __init__(self, population_size, num_generations, size):
        self.population_size = population_size
        self.num_generations = num_generations
        self.size = size
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        population = set()
        while len(population) < self.population_size:
            square = MagicSquare(self.size)
            for i in range(self.size):
                for j in range(self.size):
                    square.set_value(i, j, random.randint(1, self.size * self.size))
            population.add(square)
        return population

    def evolve(self):
        # Inicializar listas para almacenar la aptitud de cada individuo y la
        # aptitud promedio de la población en cada generación
        
        self.average_fitnesses = []
        self.avg_fitness_list = []
        self.max_fitness_list = []

        for i in range(self.num_generations):
            self.population = self.reproduce()
            self.population = self.mutate()

            # Calcular la aptitud de cada individuo y la aptitud promedio de la
            # población en esta generación y almacenarlas en las listas
            individual_fitnesses = [ind.get_fitness() for ind in self.population]
            self.average_fitnesses.append(sum(individual_fitnesses) / len(individual_fitnesses))
            average_fitnesses=(sum(individual_fitnesses) / len(individual_fitnesses))

            #calculando adaptación promedio
            total = 0
            for i in self.population:
                total += i.get_fitness_percentual()
            avg_fitness = total/len(self.population)

            #calculando el fitness máximo
            best_fitness = max(self.population, key=lambda x: x.get_fitness_percentual()).get_fitness_percentual()

            #guardando el mejor individuo
            best_ind = max(self.population, key=lambda x: x.get_fitness_percentual())

            self.avg_fitness_list.append(avg_fitness)
            self.max_fitness_list.append(max(self.population, key=lambda x: x.get_fitness_percentual()).get_fitness_percentual())
            
    def reproduce(self):
        new_population = set()
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            new_population.add(child)
        return new_population

    def tournament_selection(self):
        tournament = set()
        while len(tournament) < self.TOURNAMENT_SIZE:
            individual = random.sample(self.population, 1)[0]
            tournament.add(individual)
        return min(tournament, key=lambda x: x.get_fitness())

    def random_selection(self):
        return random.sample(self.population, 1)[0]

    def crossover(self, parent1, parent2):
        child = MagicSquare(self.size)
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.5:
                    child.set_value(i, j, parent1.get_value(i, j))
                else:
                    child.set_value(i, j, parent2.get_value(i, j))
        return child

    def mutate(self):
        new_population = set()
        for square in self.population:
            if random.random() < self.MUTATION_RATE:
                square = self.mutate_square(square)
            new_population.add(square)
        return new_population

    def mutate_square(self, square):
        i = random.randint(0, self.size - 1)
        j = random.randint(0, self.size - 1)
        square.set_value(i, j, random.randint(1, self.size * self.size))
        return square

    def get_best_solution(self):
        self.population = list(self.population)
        self.population.sort(key=lambda x: x.get_fitness())
        return self.population[0]
        
if __name__ == "__main__":
    size = int(input("Ingresa el tamaño del cuadrado mágico: "))
    population_size = int(input("Ingresa el tamaño de la población: "))
    num_generations = int(input("Ingresa el número de generaciones máxima: "))
    ga = GeneticAlgorithm(population_size, num_generations, size)
    ga.evolve()
    best_square = ga.get_best_solution()
    print("Cuadrado mágico con la mayor aptitud:")
    print(best_square)
    print("Porcentaje de adaptacion del cuadrado mágico:")
    print(best_square.get_fitness_percentual())
    print("Porcentaje de aptitud del cuadrado mágico:")
    print(best_square.get_fitness_percentuales())

    # Crear la gráfica
    plt.plot(ga.average_fitnesses, color='red', linestyle='solid', label="Promedio de Aptitud")
    plt.plot(ga.avg_fitness_list, color='blue', linestyle='solid', label="Adaptacion promedio")
    plt.plot(ga.max_fitness_list, color='green', linestyle='dashed', label="Adaptacion maxima")
    plt.title("Evolucion del fitness por generacion")
    plt.xlabel("Generacion")
    plt.ylabel("Adaptacion")
    plt.legend()
    plt.show()
