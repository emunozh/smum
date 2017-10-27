from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'VERSION')) as f:
    version = f.read().strip()

install_requires = [
    'pandas',
    'numpy',
    'scipy',
    'statsmodels',
    'matplotlib',
    'seaborn',
    'Theano',
    'pymc3',
    'rpy2',
    'ipfn'
]

setup(
    name='umsm',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='An urban metabalism spatial microsimulation model.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/emunozh/um',

    # Author details
    author='M. Esteban Munoz H.',
    author_email='emunozh@gmail.com',

    # Choose your license
    license='GPL3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Scientific community',

        # Pick your license as you wish (should match "license" above)
        'License :: GLP3 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    include_package_data=True,

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,

    setup_requires=[
        "setuptools_git >= 1.0",
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
        # 'dev': ['nose', 'flake8'],
        # 'test': ['nose', 'flake8'],
        # 'gdal': ['GDAL', 'Shapely>=1.5.13'],
    # },
)
