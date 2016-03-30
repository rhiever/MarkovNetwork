#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('MarkovNetwork/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='MarkovNetwork',
    version=package_version,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/rhiever/MarkovNetwork',
    license='License :: OSI Approved :: MIT License',
    description=('Markov Networks for neural computing'),
    long_description='''
A Python implementation of Markov Networks for neural computing.

Contact
=============
If you have any questions or comments about MarkovNetwork, please feel free to contact me via:

E-mail: rso@randalolson.com

or Twitter: https://twitter.com/randal_olson

This project is hosted at https://github.com/rhiever/MarkovNetwork
''',
    zip_safe=True,
    install_requires=['numpy'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Artificial Life'
    ],
    keywords=['markov network', 'neural computing', 'artificial neural network', 'artificial intelligence', 'evolved intelligence'],
)
