# IPython log file

get_ipython().magic(u'logstart')
get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tspsolver.tsp_generator import TSPGenerator
from tspsolver.ga.simulator import Simulator
from tspsolver.tuning import GeneticAlgorithmParameterEstimation

OUTPUT_TABLE_DIR = "../report/tables/"
OUTPUT_FIG_DIR = "../report/figures/"
NUM_DATASETS = 5 # number of datasets to take the mean fitness over 
NUM_POINTS = 50  # number of points to use in each dataset
tuner = GeneticAlgorithmParameterEstimation(NUM_DATASETS, NUM_POINTS)
params = {
    "num_epochs": [1000],
    "num_elites": [0, 1, 2],
    "generator": ["SimplePopulationGenerator"],
    "generator_population_size": [40],
    "selector": ["TournamentSelection"],
    "selector_tournament_size": [10],
    "crossover": ["OrderCrossover"],
    "crossover_pcross": [0.9],
    "mutator": ["InversionMutation"],
    "mutator_pmutate": [0.2]
}
elite_results = tuner.perform_grid_search(params)
elite_results
gen = TSPGenerator(NUM_POINTS)
data = gen.generate()

all_fitness = []
for i, row in elite.iterrows():
    params = row.to_dict()
    sim = Simulator(**params)
    sim.evolve(data)
    all_fitness.append(sim.get_min_fitness()[::10])
    
df = pd.DataFrame(np.array(all_fitness))
gen = TSPGenerator(NUM_POINTS)
data = gen.generate()

all_fitness = []
for i, row in elite_results.iterrows():
    params = row.to_dict()
    sim = Simulator(**params)
    sim.evolve(data)
    all_fitness.append(sim.get_min_fitness()[::10])
    
df = pd.DataFrame(np.array(all_fitness))
df
df.T.plot()
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
plt.savefig(OUTPUT_FIG_DIR + 'elite_convergence.png', bbox_inches='tight')
elite_results[['num_elites', 'fitness']].to_latex(OUTPUT_TABLE_DIR + "selection_vs_pop_size2.tex")
elite_results[['num_elites', 'fitness']].to_latex(OUTPUT_TABLE_DIR + "elite_fitness.tex")
params = {
    "num_epochs": [1000],
    "num_elites": [0, 1, 2],
    "generator": ["SimplePopulationGenerator"],
    "generator_population_size": [40],
    "selector": ["TournamentSelection"],
    "selector_tournament_size": [10],
    "crossover": ["OrderCrossover"],
    "crossover_pcross": [0.9],
    "mutator": ["InversionMutation"],
    "mutator_pmutate": [0.2]
}
elite_results = tuner.perform_grid_search(params)
elite_results
elite_results[['num_elites', 'fitness']].to_latex(OUTPUT_TABLE_DIR + "elite_fitness.tex")
params = {
    "num_epochs": [1000],
    "num_elites": [1],
    "generator": ["SimplePopulationGenerator", "KNNPopulationGenerator"],
    "generator_population_size": [40],
    "selector": ["TournamentSelection"],
    "selector_tournament_size": [10],
    "crossover": ["OrderCrossover"],
    "crossover_pcross": [0.9],
    "mutator": ["InversionMutation"],
    "mutator_pmutate": [0.2]
}
elite_results = tuner.perform_grid_search(params)
knn_results = tuner.perform_grid_search(params)
knn_results
params = {
    "num_epochs": [1000],
    "num_elites": [1],
    "generator": ["SimplePopulationGenerator", "KNNPopulationGenerator"],
    "generator_random_proportion": [0.3, 0.5, 0.6],
    "generator_population_size": [40],
    "selector": ["TournamentSelection"],
    "selector_tournament_size": [10],
    "crossover": ["OrderCrossover"],
    "crossover_pcross": [0.9],
    "mutator": ["InversionMutation"],
    "mutator_pmutate": [0.2]
}
knn_results = tuner.perform_grid_search(params)
knn_results
knn_results.loc[0]
knn_results.iloc[0]
knn_results.iloc[[0,3,4,5]]
t = knn_results.iloc[[0,3,4,5]]
t = knn_results.iloc[[0,3,4,5]].T
t = knn_results.iloc[[0,3,4,5]].T
t = knn_results.iloc[[0,3,4,5]]
t = knn_results.iloc[[0,3,4,5]]
t
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_radnom_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_random_proportion', 'fitness']]
t.iloc[0]['generator_random_proportion'] = np.NaN
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_random_proportion', 'fitness']]
t = .iloc[0]['generator_random_proportion'] = np.NaN
t
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_random_proportion', 'fitness']]
t = t.iloc[0]['generator_random_proportion'] = np.NaN
t
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_random_proportion', 'fitness']]
t.iloc[0] = t.iloc[0]['generator_random_proportion'] = np.NaN
t
t = knn_results.iloc[[0,3,4,5]]
t[['generator', 'generator_random_proportion', 'fitness']]
t.iloc[0]['generator_random_proportion'] = np.NaN
t
t = knn_results.iloc[[0,3,4,5]]
t.iloc[0]['generator_random_proportion'] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t.iloc[0]['generator_random_proportion'] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t.iloc[0]['generator_random_proportion'] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t['generator_random_proportion', :0] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t[0, 'generator_random_proportion'] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t[0, 2] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t[0, 1] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t['generator_random_proportion'][0] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t['generator_random_proportion'][0] = np.NaN
t[['generator', 'generator_random_proportion', 'fitness']]
t = knn_results.iloc[[0,3,4,5]]
t['generator_random_proportion'][0] = np.NaN
t.index = np.arange(4)
t[['generator', 'generator_random_proportion', 'fitness']]
t[['generator', 'generator_random_proportion', 'fitness']].to_latex(OUTPUT_TABLE_DIR + "knn_fitness.tex")
gen = TSPGenerator(NUM_POINTS)
data = gen.generate()

all_fitness = []
for i, row in knn_results.iterrows():
    params = row.to_dict()
    sim = Simulator(**params)
    sim.evolve(data)
    all_fitness.append(sim.get_min_fitness()[::10])
    
df = pd.DataFrame(np.array(all_fitness))
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
# plt.savefig(OUTPUT_FIG_DIR + 'elite_convergence.png', bbox_inches='tight')
gen = TSPGenerator(NUM_POINTS)
data = gen.generate()
p = knn_results.iloc[[0,3,4,5]]
all_fitness = []
for i, row in p.iterrows():
    params = row.to_dict()
    sim = Simulator(**params)
    sim.evolve(data)
    all_fitness.append(sim.get_min_fitness()[::10])
    
df = pd.DataFrame(np.array(all_fitness))
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
# plt.savefig(OUTPUT_FIG_DIR + 'elite_convergence.png', bbox_inches='tight')
params = {
    "num_epochs": [1000],
    "num_elites": [1],
    "generator": ["SimplePopulationGenerator", "KNNPopulationGenerator"],
    "generator_random_proportion": [0.3, 0.5, 0.6],
    "generator_population_size": [40],
    "selector": ["TournamentSelection"],
    "selector_tournament_size": [10],
    "crossover": ["OrderCrossover"],
    "crossover_pcross": [0.9],
    "mutator": ["InversionMutation"],
    "mutator_pmutate": [0.2]
}
NUM_DATASETS = 5 # number of datasets to take the mean fitness over 
NUM_POINTS = 100  # number of points to use in each dataset
tuner = GeneticAlgorithmParameterEstimation(NUM_DATASETS, NUM_POINTS)
knn_results = tuner.perform_grid_search(params)
t = knn_results.iloc[[0,3,4,5]]
t['generator_random_proportion'][0] = np.NaN
t.index = np.arange(4)
t[['generator', 'generator_random_proportion', 'fitness']]
t[['generator', 'generator_random_proportion', 'fitness']].to_latex(OUTPUT_TABLE_DIR + "knn_fitness.tex")
gen = TSPGenerator(NUM_POINTS)
data = gen.generate()
p = knn_results.iloc[[0,3,4,5]]
all_fitness = []
for i, row in p.iterrows():
    params = row.to_dict()
    sim = Simulator(**params)
    sim.evolve(data)
    all_fitness.append(sim.get_min_fitness()[::10])
    
df = pd.DataFrame(np.array(all_fitness))
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
plt.savefig(OUTPUT_FIG_DIR + 'knn_convergence.png', bbox_inches='tight')
df
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
plt.savefig(OUTPUT_FIG_DIR + 'knn_convergence.png', bbox_inches='tight')
df
t
df['generator-name'] = selection_results[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])), axis=1)
t
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t['generator-name'][0] = t['generator']
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t['generator-name'][0] = t['generator']
t
t['generator-name'] = t[['generator', 'generator_random_proportion']].apply(lambda x: '-'.join([x[0], str(x[1])]), axis=1)
t['generator-name'][0] = t['generator'][0]
t
df.index = t['generator-name']
ax = df.T.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness")
plt.savefig(OUTPUT_FIG_DIR + 'knn_convergence.png', bbox_inches='tight')
