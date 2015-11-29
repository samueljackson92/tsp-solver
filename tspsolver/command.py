import click
import numpy as np
import logging
import os.path
import json
import sys

from tsp_generator import TSPGenerator
from tuning import GeneticAlgorithmParameterEstimation
from ga.simulator import Simulator
from ga.plotting import plot_fitness, plot_solution

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def generate_random_dataset(num_points):
    logger.info('Generating %d uniformly random points' % num_points)
    generator = TSPGenerator(num_points)
    return generator.generate()


def load_dataset(data_file):
    logger.info('Loading dataset from file %s' % data_file)
    data = np.loadtxt(data_file, delimiter=',')
    logger.info('Loaded %d points from %s' % (data.shape[0], data_file))
    return data


def save_dataset(data, data_file):
    np.savetxt(data_file, data, delimiter=",", header="x,y")
    logger.info("%d points saved to file: %s" % (data.shape[0], data_file))


def load_parameter_file(file_name):
    if os.path.exists(file_name) and os.path.isfile(file_name):
        with open(file_name) as param_file:
            try:
                parameters = json.load(param_file)
            except ValueError as e:
                logger.error("Failed to parse parameter file:\n %s", str(e))
                sys.exit(1)
    else:
        logger.error("Could not open parameter file: %s" % file_name)
        sys.exit(1)

    return parameters


def save_parameter_file(file_name, parameters):
    with open(file_name, 'w') as param_file:
        try:
            json.dump(parameters, param_file, indent=4, separators=(',', ': '))
        except ValueError as e:
            logger.error("Failed to save parameter file:\n %s", str(e))
            sys.exit(1)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--num_points', '-n', default=10,
              help='Number of points to generate')
@click.argument('output-file')
def generate(output_file, num_points):
    generator = TSPGenerator(num_points)
    logger.info("Generating %d points" % num_points)

    data, _ = generator.generate()
    save_dataset(data, output_file)


@cli.command()
@click.argument("parameter-file")
@click.option('--dataset_file', '-f',
              help="File containg a dataset to use.")
@click.option('--num-points', '-n', default=10,
              help="Number of points to generate.")
def solve(parameter_file, dataset_file, num_points):
    if dataset_file is not None:
        data = load_dataset(dataset_file)
    else:
        data = generate_random_dataset(num_points)

    params = load_parameter_file(parameter_file)
    sim = Simulator(**params)
    solution = sim.evolve(data)

    plot_fitness(sim)
    plot_solution(data, solution)


@cli.command()
@click.argument("parameter-file")
@click.argument("output-file")
@click.option('--num-points', '-n', default=10,
              help="Number of points to generate.")
@click.option('--num-datasets', '-d', default=3,
              help="Number of points to generate.")
def tune(parameter_file, output_file, num_points, num_datasets):
    params = load_parameter_file(parameter_file)
    tuner = GeneticAlgorithmParameterEstimation(num_datasets=num_datasets,
                                                dataset_size=num_points)
    best_params = tuner.perform_grid_search(params)
    save_parameter_file(output_file, best_params)
