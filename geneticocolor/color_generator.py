"""
color_generator
===============

Generate a set of colors for N points with K classes, using oRGB space color.
"""

from deap import creator, base, tools, algorithms
from collections import OrderedDict, Counter
from geneticocolor.oRGB import RGB_to_oRGB, oRGB_to_RGB, randcolor_RGB, randcolor_oRGB
from random import random, randint
from time import time
import numpy


def _centroid(pointArray):
    length = pointArray.shape[0]
    size = pointArray.shape[1]
    return numpy.array([sum(pointArray[:,i])/length for i in range(0,size)])

def _distances_centroid(N, K, points, points_classes):
    geom_distances = numpy.zeros([K,K])
    centroids = numpy.zeros([K,2])
    for i in range (0, K):
        points_i = [points[index] for index in range(0,N) if points_classes[index] == i]
        centroids[i] = _centroid(numpy.array(points_i))
    for i in range (0, K):
        j = i + 1
        while (j < K):
            d = numpy.linalg.norm(centroids[i] - centroids[j])
            geom_distances[i][j] = d
            geom_distances[j][i] = d
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

def _objective_norm(geom_distances, points_classes, counter_classes, indiv):
    K = len(indiv)
    N = geom_distances.shape[0]
    color_distances = numpy.zeros([K, K])
    res = 0

    for i in range(0,N):
        ki = points_classes[i]
        j = i + 1
        resi = 0
        while(j < N):
            kj = points_classes[j]
            if geom_distances[i][j] != 0:
                if (color_distances[ki][kj] == 0):
                    color_distances[ki][kj] = numpy.linalg.norm(numpy.array(indiv[ki]) - numpy.array(indiv[kj]))
                resi += (color_distances[ki][kj] / geom_distances[i][j]) / counter_classes[kj]
            j += 1
        res += resi / counter_classes[ki]
    return res,

def _objective_cent(geom_distances, points_classes, counter_classes, indiv):
    K = len(indiv)
    color_distances = numpy.zeros([K, K])
    res = 0
    for i in range (0, K):
        j = i + 1
        while j < K:
            if geom_distances[i][j] != 0:
                if color_distances[i][j] == 0:
                    color_distances[i][j] = numpy.linalg.norm(numpy.array(indiv[i]) - numpy.array(indiv[j]))
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

def _values_params(mode, NGEN, TAM_POPULATION, mating_prob,  mating_mutation_prob,
    mutation_prob, tournsize):

    _mating_mutation_prob = 0.7
    _mutation_prob = 0.1
    _tournsize = 3
    _mating_prob = 0.5
    _TAM_POPULATION = 10
    _NGEN = 20
    _use_time = True

    if mode == 'custom':
        _NGEN = NGEN
        _TAM_POPULATION = TAM_POPULATION
        _mating_mutation_prob = mating_mutation_prob
        _mutation_prob = mutation_prob
        _tournsize = tournsize
        _mating_prob = mating_prob
        _use_time = False

    if mode == 'optimized':
        _NGEN = 300
        _use_time = False

    return (_NGEN, _TAM_POPULATION, _mating_prob, _mating_mutation_prob, _mutation_prob, _tournsize, _use_time)

def generate(x, y, points_classes,
    mode = 'fast',
    verbose = False,
    return_fitness_solution = False,
    return_evolution_data = False,
    NGEN = 20,
    TAM_POPULATION = 10,
    mating_prob = 0.5,
    mating_mutation_prob = 0.7,
    mutation_prob = 0.1,
    tournsize = 3):
    
    """ Convert oRGB color into sRGB color.

    Parameterss
    -----------
    x : float list
        list of x components.
    y : float list
        list of y components.
    mode : string
        Avaible values: 'fast', 'custom' and 'optimized'.
        Default value is 'fast'.
    verbose: bool
        Flag for displaying log messages. Default value is False.    
    return_fitness_solution: bool
        If true, the function will also return the fitness value of the solution.
        Default value is False.
    return_evolution_data: bool
        Default value is False.
        If true, the function will also return a list with best fitness value for every generation.
        Default value is False.
    NGEN: int
        Number of generations in the genetic algorithm.
        Default value for 'fast' mode is 20 and for 'optimized' is 300.
    TAM_POPULATION: int.
        Size of the genetic algorithm population.
        Default value for 'fast' and 'optimized' modes is 10.
    mating_prob: float
        Probability of mating in crossover function. Value between 1 and 0. 
        Default value for 'fast' and 'optimized' modes is 0.5
    mating_mutation_prob: float
        Probability of mutation during crossover function. Value between 1 and 0.
        Default value for 'fast' and 'optimized' modes is 0.7
    mutation_prob: float
        Probability of mutation in mutation function. Value between 1 and 0.
        Default value for 'fast' and 'optimized' modes is 0.1
    tournsize: int
        Size of the tournament in selection function.
        Default value for 'fast' and 'optimized' modes is 3

    Returns
    -------
    list : [c0, c1, c2, ..., ck-1]
        List of size K (the number of point classes) with the generated colors in RGB space.
        Each color is a three element list with values inside interval [0-1]

    Notes
    -----
    There are three modes for using this function:
    -   'fast': option by default. Uses default values for algorithm params.
    -   'optimized': same as 'fast', but with more generations. This option is
        indicated for cases with K >= 20. Notice that it will increase executing time.
    -   'custom': it will take params indicated in the call of the function.

    Fast mode is indicated for 20 diferent classes or below. It will provide best solution generated in 2.5s or at least 20 generations.
    In case there is a need for using diferent params, custom mode allows customization of all parameters.
    If only more executions are needed, optimized mode uses 300 generations, but that will also increase notably execution time.
    """

    classes = list(OrderedDict.fromkeys(points_classes))
    points_classes = numpy.array([classes.index(i) for i in points_classes])
    points = numpy.array(list(zip(x,y)))

    counter_classes = Counter(points_classes)

    # Number of points and number of classses
    N = points.shape[0]
    K = len(classes)

    if K == 1:
        return randcolor_RGB()

    distances = _distances
    objective = _objective_norm
    if N > 100:
        distances = _distances_centroid
        objective = _objective_cent

    NGEN, TAM_POPULATION, mating_prob,  mating_mutation_prob, mutation_prob, tournsize, use_time = _values_params(mode,
        NGEN, TAM_POPULATION, mating_prob,  mating_mutation_prob, mutation_prob, tournsize)

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

    while (use_time and (time() - start_time < 2.5 or gen < MIN_GEN)) or (not use_time and gen < MIN_GEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=mating_prob, 
            mutpb=mating_mutation_prob)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))        

        top = tools.selBest(population, k=1)
        top_fitness = objective(geom_distances, points_classes, counter_classes, top[0])

        if verbose:
            print ("Generation:", gen, "Best fitness:", top_fitness)

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
