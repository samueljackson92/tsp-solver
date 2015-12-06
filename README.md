# TSP Solver
[![Build Status](https://magnum.travis-ci.com/samueljackson92/tsp-solver.svg?token=BaPtpk9DsGYbbzV8h1jS&branch=master)](https://magnum.travis-ci.com/samueljackson92/tsp-solver)

Solving Travelling Salesman Problems with Genetic Algorithms. Originally for SEM6120.

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
    "crossover_use_rog": false,
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
