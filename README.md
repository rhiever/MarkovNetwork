[![Build Status](https://travis-ci.org/rhiever/MarkovNetwork.svg?branch=master)](https://travis-ci.org/rhiever/MarkovNetwork)
[![Code Health](https://landscape.io/github/rhiever/MarkovNetwork/master/landscape.svg?style=flat)](https://landscape.io/github/rhiever/MarkovNetwork/master)
[![Coverage Status](https://coveralls.io/repos/rhiever/MarkovNetwork/badge.svg?branch=master&service=github)](https://coveralls.io/github/rhiever/MarkovNetwork?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://badge.fury.io/py/MarkovNetwork.svg)](https://badge.fury.io/py/MarkovNetwork)

# Markov Network

[![Join the chat at https://gitter.im/rhiever/MarkovNetwork](https://badges.gitter.im/rhiever/MarkovNetwork.svg)](https://gitter.im/rhiever/MarkovNetwork?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Python implementation of Markov Networks for neural computing.

## License

Please see the [repository license](https://github.com/rhiever/MarkovNetwork/blob/master/LICENSE) for the licensing and usage information for datacleaner.

Generally, we have licensed the MarkovNetwork package to make it as widely usable as possible.

## Installation

MarkovNetwork is built to use NumPy arrays for fast array processing. As such, we recommend installing the [Anaconda Python distribution](https://www.continuum.io/downloads) prior to installing MarkovNetwork. However, MarkovNetwork should work fine with any basic install of Python.

Once the prerequisites are installed, datacleaner can be installed with a simple `pip` command:

```
pip install MarkovNetwork
```

## Usage

When creating an instance of a MarkovNetwork, you can pass the following parameters:

```
num_input_states: int (required)
    The number of input states in the Markov Network
num_memory_states: int (required)
    The number of internal memory states in the Markov Network
num_output_states: int (required)
    The number of output states in the Markov Network
random_genome_length: int (default: 10000)
    Length of the genome if it is being randomly generated
    This parameter is ignored if "genome" is not None
seed_num_markov_gates: int (default: 4)
    The number of Markov Gates with which to seed the Markov Network
    It is important to ensure that randomly-generated Markov Networks have at least a few Markov Gates to begin with
    May sometimes result in fewer Markov Gates if the Markov Gates are randomly seeded in the same location
    This parameter is ignored if "genome" is not None
probabilistic: bool (default: True)
    Flag indicating whether the Markov Gates are probabilistic or deterministic
genome: array-like (default: None)
    An array representation of the Markov Network to construct
    All values in the array must be integers in the range [0, 255]
    If None, then a random Markov Network will be generated
```

The following code creatives a deterministic MarkovNetwork, provides some input, activates the network, then retrieves the output:

```python
from MarkovNetwork import MarkovNetwork

my_mn = MarkovNetwork(num_input_states=2,
                      num_memory_states=4,
                      num_output_states=2,
                      random_genome_length=8000,
                      seed_num_markov_gates=5,
                      probabilistic=False)

my_mn.update_input_states([1, 0])
my_mn.activate_network()
output_states = my_mn.get_output_states()
```

You can repeat this process multiple times with different input:

```python
from MarkovNetwork import MarkovNetwork

my_mn = MarkovNetwork(num_input_states=2,
                      num_memory_states=4,
                      num_output_states=2,
                      random_genome_length=8000,
                      seed_num_markov_gates=5,
                      probabilistic=False)

my_mn.update_input_states([1, 0])
my_mn.activate_network()
output_states1 = my_mn.get_output_states()

my_mn.update_input_states([0, 1])
my_mn.activate_network()
output_states2 = my_mn.get_output_states()
```

If you want to allow the MarkovNetwork to activate multiple times with the same inputs, you can pass a `num_activations` parameter to `activate_network()`:

```python
from MarkovNetwork import MarkovNetwork

my_mn = MarkovNetwork(num_input_states=2,
                      num_memory_states=4,
                      num_output_states=2,
                      random_genome_length=8000,
                      seed_num_markov_gates=5,
                      probabilistic=False)

my_mn.update_input_states([1, 0])
my_mn.activate_network(num_activations=20)
output_states = my_mn.get_output_states()
```

Finally, you can seed a MarkovNetwork with a pre-existing byte string by passing the `genome` parameter:

```python
from MarkovNetwork import MarkovNetwork
import numpy as np

my_mn_genome = np.random.randint(0, 256, 15000)
my_mn = MarkovNetwork(num_input_states=2,
                      num_memory_states=4,
                      num_output_states=2,
                      probabilistic=False,
                      genome=my_mn_genome)
```

## Having problems with the MarkovNetwork package?

Before you file a bug report, please [check the existing issues](https://github.com/rhiever/MarkovNetwork/issues?utf8=%E2%9C%93&q=is%3Aissue) to make sure that your issue hasn't already been filed or solved. If the bug is unreported, please [file a new issue](https://github.com/rhiever/MarkovNetwork/issues/new) and describe your bug in detail.

## Contributing to the MarkovNetwork package

We welcome you to [check the existing issues](https://github.com/rhiever/MarkovNetwork/issues/) for bugs or enhancements to work on. If you have an idea for an extension to the MarkovNetwork package, please [file a new issue](https://github.com/rhiever/MarkovNetwork/issues/new) so we can discuss it.

## Citing MarkovNetwork

If you use the MarkovNetwork package as part of your workflow in a scientific publication, please consider citing the following publication that describes Markov Networks in detail.

Randal S. Olson, David B. Knoester, and Christoph Adami. "Evolution of swarming behavior is shaped by how predators attack." *Artificial Life Journal*, to appear in Spring 2016.

```
@misc{Olson2016SelfishHerd,
author = {Olson, Randal S. and Knoester, David B. and Adami, Christoph},
title = {Evolution of swarming behavior is shaped by how predators attack},
howpublished={arXiv e-print. http://arxiv.org/abs/1310.6012},
year={2016}
}
```

You can also cite the repository directly using the following DOI:

[![DOI](https://zenodo.org/badge/20747/rhiever/MarkovNetwork.svg)](https://zenodo.org/badge/latestdoi/20747/rhiever/MarkovNetwork)
