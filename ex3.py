from pyeasyga.pyeasyga import GeneticAlgorithm


def fitness(individual, data):
    fitness = 0
    x = int("".join([str(a) for a in individual]),2)/10
    y= -x**2
    if -10 <= x <= 10:
        return y
    return 0


# Provide sample data

#1.Determine the size of the individuals

data = [1,1,1,1,1,1,1,1]

ga = GeneticAlgorithm(data, maximise_fitness=False)
ga.fitness_function = fitness
ga.run()

best_y, best_individual = ga.best_individual()
best_x = int("".join([str(a) for a in best_individual]), 3)/10
print(ga.best_individual())

#Determine the highest value of Y in the equationy=x3+ (x−1)2in the range (-3, 1) with an interval of0.001
#