# TSP Solver
[![Build Status](https://magnum.travis-ci.com/samueljackson92/tsp-solver.svg?token=BaPtpk9DsGYbbzV8h1jS&branch=master)](https://magnum.travis-ci.com/samueljackson92/tsp-solver)

Solving Travelling Salesman Problems with Genetic Algorithms. Originally for SEM6120. Documentation for this project is located in the doc/build/html folder.

## Installation
Installation is easiest using the ```pip``` package manager. Python version 2.7.9+ automatically ship with ```pip```. If you're using an older version of Python you can find the installation instructions [here](http://pip.readthedocs.org/en/stable/installing/).

To install the program locally ```cd``` into the top level directory of the probject and run the following:

```pip install -e .```

All of the project dependancies should be installed automatically. If for some reason they are not, the full list of dependancies are:

 - numpy
 - scipy
 - pandas
 - scikit-learn
 - matplotlib
 - click

## Running the Program

There are three major commands: ```generate```, ```solve``` and ```tune```. 

### Generating Datasets
The generate command will create a uniformly random TSP datasetand output it to the specified CSV file.

```bash
tspsolver generate dataset.csv
```
Optionally, it can also take a parameter specifying the number of cities to generate (default is 10):

```bash
tspsovler generate -n 30 dataset.csv
```

### Solving TSP problems
The ```solve``` command can be used to run the genetic algorithm and produce solutions to the problems. The command takes a JSON parameter file as an arguement. Optionally you can provide a dataset file (such as produced by the ```generate``` command) or specify a random number of cities to generate. When the command terminates, two plots are produced. One shows the min, mean, and max fitness across all generations, the second shows the best solution found over all datasets.

Example Parameter File:
```json
{
    "num_epochs": 1000,
    "num_elites": 2,
    "generator": "SimplePopulationGenerator",
    "generator_population_size": 20,
    "selector": "RouletteWheelSelection",
    "selector_tournament_size": 5,
    "crossover": "OrderCrossover",
    "crossover_pcross": 0.9,
    "mutator": "InversionMutation",
    "mutator_pmutate": 0.1
}
```

Solving a TSP problem with a randomly generated TSP dataset:

```bash
tspsolver solve -n 20  params.json
```

Solving a TSP problem with an exisitng dataset:
```bash
tspsolver solve -f dataset.csv params.json
```

### Tuning Parameters
The final command can be used to run a range of parameter configurations over a number of different datasets. This can be useful to examine the effects of different parameter datasets. Each configuration is run on ```n``` different randomly generated datasets and the median result is take to represent the whoel. The results for all datasets are saved to a CSV file. The configuration that produced the best results is also saved to a JSON file.

This command takes a special parameter file that specifies ranges of parameters. An example is shown below:

```json
{
    "num_epochs": [1000],
    "num_elites": [0],
    "generator": ["SimplePopulationGenerator"],
    "generator_population_size": [20],
    "selector": ["TournamentSelection"],
    "selector_tournament_size": [5],
    "crossover": ["OrderCrossover"],
    "crossover_pcross": [0.6, 0.7, 0.8, 0.9],
    "mutator": ["InversionMutation"],
    "mutator_pmutate": [0.01, 0.05, 0.1, 0.2]
}
```

This will run the genetic algorithm with a varying range of crossover and mutation probabilities. An example of running the tuning command is as follows:

```bash
tspsolver tune -d 5 -n 50 tuning_params.json results.csv best.json
```

In the above command the ```-d``` command specifies the number of datasets to generate for each parameter configuration. The ```-n``` flag specifies the number of random generated points to use for each dataset. ```tuning_params.json``` is the special parameter file with ranges. ```results.csv``` is the csv file created with all parameter results. ```best.json``` is the generated parameter file containing the parameters that produced the best run.
