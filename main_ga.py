from metaheuristics.genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.solve(1000)
    vs = ga.population.values()
    vs = list(vs)
    print(len(vs))
    breakpoint()
    for i in vs:
        print("--------")
        print(i['value'])
        # print(i['array'])