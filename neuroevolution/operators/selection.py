import numpy as np


reduce_dict = lambda x, max_pop_size, division: {
    k:v for index,(k,v) in enumerate(x.items())
    if index<=(max_pop_size//(division*2))-1
}

def selection(population:dict, max_pop_size:int, division:float = 5):
    """Operator that represent the natural selection where they will only survive
    the 20% of the offspring with the best fitness, that is the minimum loss.
    """
    if len(population.keys()) <= max_pop_size//6:
      return population

    new_pop = {}
    new_pop = reduce_dict(population, max_pop_size, division)
    for _ in range(len(population.keys())//(division*2)):
      idx = np.random.randint(0,len(population.keys()))
      key = list(population.keys())[idx]
      new_pop[key] = population[key]
    population = new_pop
    return population