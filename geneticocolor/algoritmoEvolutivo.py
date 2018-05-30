from deap import creator, base, tools, algorithms
from collections import OrderedDict, Counter
import numpy
from random import random, randint
from oRGB import RGB_to_oRGB, oRGB_to_RGB, randcolor_RGB, randcolor_oRGB
from time import time

def _centeroid(pointArray):
    length = pointArray.shape[0]
    sum_x = numpy.sum(pointArray[:, 0])
    sum_y = numpy.sum(pointArray[:, 1])
    res = numpy.array([sum_x, sum_y])
    return res

def _distances_centroid(N, K, points, points_classes):
    geom_distances = numpy.zeros([K,K])
    centroids = numpy.zeros([K,2])
    for i in range (0, K):
        points_i = [points[index] for index in range(0,N) if points_classes[index] == i]
        centroids[i] = _centeroid(numpy.array(points_i))
    for i in range (0, K):
        j = i + 1
        while (j < K):
            geom_distances[i][j] = numpy.linalg.norm(centroids[i] - centroids[j])
            geom_distances[j][i] = geom_distances[i][j]
            j += 1
    return geom_distances

def _distances(N, K, points, points_classes):
    geom_distances = numpy.zeros([N, N])
    for i in range(0, N):
        j = i + 1
        while(j < N):
            if points_classes[i] != points_classes[j]:
                geom_distances[i][j] = numpy.linalg.norm(points[i] - points[j])
                geom_distances[j][i] = geom_distances[i][j]
            j += 1
    return geom_distances

def _objective(geom_distances, points_classes, counter_classes, indiv):
    K = len(indiv)
    N = geom_distances.shape[0]
    color_distances = numpy.zeros([K, K])
    res = 0 # resultado

    for i in range(0,N): # para todos los points
        ki = points_classes[i]
        j = i + 1
        while(j < N):
            kj = points_classes[j]
            if geom_distances[i][j] != 0:
                if (color_distances[ki][kj] == 0):
                    color_distances[ki][kj] = numpy.linalg.norm(numpy.array(indiv[ki]) - numpy.array(indiv[kj]))
                    # color_distances[kj][ki] = color_distances[ki][kj]
                res +=  color_distances[ki][kj] / geom_distances[i][j]
            j += 1
    return res,

def _objective_norm(geom_distances, points_classes, counter_classes, indiv):
    K = len(indiv)
    N = geom_distances.shape[0]
    color_distances = numpy.zeros([K, K])
    res = 0 # resultado

    for i in range(0,N): # para todos los points
        ki = points_classes[i]
        j = i + 1
        resi = 0
        while(j < N):
            kj = points_classes[j]
            if geom_distances[i][j] != 0:
                if (color_distances[ki][kj] == 0):
                    color_distances[ki][kj] = numpy.linalg.norm(numpy.array(indiv[ki]) - numpy.array(indiv[kj]))
                    # color_distances[kj][ki] = color_distances[ki][kj]
                resi += (color_distances[ki][kj] / geom_distances[i][j]) / counter_classes[kj]
            j += 1
        res += resi / counter_classes[ki]
    return res,

def _objective_cent(geom_distances, points_classes, counter_classes, indiv):
    K = len(indiv)
    color_distances = numpy.zeros([K, K])
    res = 0
    for i in range (0, K): # for each color
        j = i + 1
        while j < K:
            if geom_distances[i][j] != 0:
                if color_distances[i][j] == 0:
                    color_distances[i][j] = numpy.linalg.norm(numpy.array(indiv[i]) - numpy.array(indiv[j]))
                    # color_distances[j][i] = color_distances[i][j]
                res += color_distances[i][j] / geom_distances[i][j]
            j += 1      
    return res,

def _mutation(indiv, indpb):
    K = len(indiv)
    for color in indiv:
        if random() < indpb:
            selected_class = randint(0, K-1)
            indiv[selected_class] = randcolor_oRGB()
    return indiv,

def _values_params():
    mating_mutation_prob = 0.7
    mutation_prob = 0.1
    tournsize = 3
    mating_prob = 0.5
    TAM_POPULATION = 10
    NGEN = 20

    return (mutation_prob, tournsize, mating_prob, mating_mutation_prob, NGEN, TAM_POPULATION)

def generate(x, y, points_classes, verbose=False, return_fitness_solution = False,
    return_evolution_data=False, NGEN=None):
    classes = list(OrderedDict.fromkeys(points_classes))
    points_classes = numpy.array([classes.index(i) for i in points_classes])
    points = numpy.array(zip(x,y))
    counter_classes = Counter(points_classes)

    # Number of points and number of classses
    N = points.shape[0]
    K = len(classes)

    if K == 1:
        return randcolor_RGB

    distances = _distances
    objective = _objective_norm
    if N > 100:
        distances = _distances_centroid
        objective = _objective_cent

    mutation_prob, tournsize, mating_prob, mating_mutation_prob, NGEN_default, TAM_POPULATION = _values_params()

    if NGEN is None:
        NGEN = NGEN_default

    # Configuration for optimization - maximization
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))

    # Individual definition
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('color', randcolor_oRGB)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.color, n=K)
    toolbox.register('Population', tools.initRepeat, list, toolbox.individual)

    # Geometric distances
    geom_distances = distances(N, K, points, points_classes)

    # Toolbox configuration
    toolbox.register('evaluate', objective, geom_distances, points_classes, counter_classes)
    toolbox.register('select', tools.selTournament, tournsize=tournsize)    
    toolbox.register('mate', tools.cxOnePoint)
    toolbox.register('mutate', _mutation, indpb=mutation_prob)

    # Initial population
    population = toolbox.Population(TAM_POPULATION)
    MIN_GEN = NGEN
    gen = 0

    # Genetic Algorithm
    maxims = []
    best_indivs = []

    start_time = time()

    while time() - start_time < 2.5 or gen < MIN_GEN:
        offspring = algorithms.varAnd(population, toolbox, cxpb=mating_prob, 
            mutpb=mating_mutation_prob)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))        

        top = tools.selBest(population, k=1)
        top_fitness = objective(geom_distances, points_classes, counter_classes, top[0])

        if verbose:
            print "Generation:", gen, "Best fitness:", top_fitness

        maxims.append(top_fitness)
        best_indivs.append(top[0])
        gen += 1

    max_fitness = max(maxims)
    winner = best_indivs[maxims.index(max_fitness)]
    winner_RGB = [list(oRGB_to_RGB(color[0],color[1],color[2])) for color in winner]
    array_colores = [winner_RGB[points_classes[i]] for i in range(0,N)]

    evolution_data = [list(range(0,gen)),maxims]

    if return_fitness_solution or return_evolution_data:
        res = list()
        res.append(array_colores)
        if return_fitness_solution:
            res.append(max_fitness)
        if return_evolution_data:
            res.append(evolution_data)
        return res

    return array_colores
