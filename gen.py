from parse import parse_from_txt
from developer_chromosome import DeveloperChromosome
from k_point_crossover import k_point_crossover
from task import Task
import random
import matplotlib.pyplot as plt

# константы задачи
ONE_MAX_LENGTH = 1000    # длина подлежащему оптимизации гену
FILE_NAME = 'input.txt'  # имя файла для конфигурации

# константы генетического алгоритма
POPULATION_SIZE = 50     # количество индивидуумов в популяции
P_CROSSOVER = 0.9        # вероятность скрещивания
P_MUTATION = 0.1         # вероятность мутации индивидуума
CROSS_INTERVAL_COUNT = 2 # Количсетво интервалов для скрещивания
MAX_GENERATIONS = 100    # максимальное количество поколений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

result = parse_from_txt(FILE_NAME)
ONE_MAX_LENGTH = result.n_task
tasks = []

for i in range(ONE_MAX_LENGTH):
    task = Task(result.task_difficult[i], result.task_timing[i])
    tasks.append(task)

class FitnessMax():
    def __init__(self):
        self.values = [0]

class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()

def oneMaxFitness(individual):
    max_time = 0
    for item in individual:
        if (item.timing_for_task > max_time):
            max_time = item.timing_for_task

    return 100000 / max_time, # кортеж

def individualCreator():
    developer_index = random.randint(1, result.n_developers)
    developer_cof = result.developers_cof[developer_index - 1] 
    
    return Individual([DeveloperChromosome(developer_index, developer_cof, tasks[i].time * developer_cof[tasks[i].difficult - 1], tasks[i]) for i in range(ONE_MAX_LENGTH)])

def populationCreator(n = 0):
    return list([individualCreator() for i in range(n)])

population = populationCreator(n=POPULATION_SIZE)
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []

def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring

def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1)-3)
    child1[s:], child2[s:] = child2[s:], child1[s:]

def mutChangeChrom(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            rand_index_developer = random.randint(1, result.n_developers)
            prev = mutant[indx]
            dev_cof = result.developers_cof[rand_index_developer - 1]
            new_chromosome = DeveloperChromosome(rand_index_developer, dev_cof, prev.task.time * dev_cof[prev.task.difficult - 1], prev.task)
            mutant[indx] = new_chromosome


fitnessValues = [individual.fitness.values[0] for individual in population]

while generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            interval = int(ONE_MAX_LENGTH / CROSS_INTERVAL_COUNT)
            child1, child2 = k_point_crossover(child1, child2, [random.randint((interval * i) + 1, (interval * (i + 1)) - -1) for i in range(0, CROSS_INTERVAL_COUNT)])

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutChangeChrom(mutant, 1 / ONE_MAX_LENGTH)

    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index].fitness.values, "\n")
    best_ind = population[best_index]
    data = []
    for i in range(0, len(best_ind)):
        data.append(best_ind[i].number)
    all_time = 0
    for item in range(0, len(best_ind)):
        all_time += best_ind[item].timing_for_task
    print("Score = ", 1000 / all_time, "\n")
    print("Лучший индивидуум = ", data, "\n")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()