try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'TSP Solver with GAs',
    'author': 'Samuel Jackson',
    'url': 'http://github.com/samueljackson92/tsp-solver',
    'download_url': 'http://github.com/samueljackson92/tsp-solver',
    'author_email': 'samueljackson@outlook.com',
    'version': '0.1.0',
    'install_requires': [
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'Click'
    ],
    'entry_points': '''
        [console_scripts]
        tspsolver=tspsolver.command:cli
    ''',
    'packages': ['tspsolver'],
    'name': 'tspsolver'
}

setup(**config)
