from pyeasyga.pyeasyga import GeneticAlgorithm


def fitness(individual, data):
    fitness = 0
    x = int("".join([str(a) for a in individual]),2)/10
    y= x
    if -100 <= x <= 100:
        return y
    return 0


# Provide sample data

#1.Determine the size of the individuals

data = [1,1,1,1,1,1,1,1]

ga = GeneticAlgorithm(data, maximise_fitness=False)
ga.fitness_function = fitness
ga.run()

print(ga.best_individual())
