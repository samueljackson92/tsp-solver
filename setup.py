try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import tspsolver

config = {
    'description': 'TSP Solver with GAs',
    'author': 'Samuel Jackson',
    'url': 'http://github.com/samueljackson92/tsp-solver',
    'download_url': 'http://github.com/samueljackson92/tsp-solver',
    'author_email': 'samueljackson@outlook.com',
    'version': tspsolver.__version__,
    'install_requires': [
	    'numpy',
	    'scipy'
    ],
    'packages': ['tspsolver'],
    'name': 'tspsolver'
}

setup(**config)
