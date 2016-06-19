# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from __future__ import print_function
import numpy as np
from MarkovNetwork import MarkovNetwork

def test_init():
    """MarkovNetwork initializer"""
    np.random.seed(3938472)
    test_mn = MarkovNetwork(num_input_states=4,
                            num_memory_states=5,
                            num_output_states=6,
                            random_genome_length=8000,
                            seed_num_markov_gates=2,
                            probabilistic=False,
                            genome=None)


    assert test_mn.num_input_states == 4
    assert test_mn.num_memory_states == 5
    assert test_mn.num_output_states == 6
    assert len(test_mn.states) == 4+5+6
    assert len(test_mn.genome) == 8000
    assert len(test_mn.markov_gates) == 2
    assert np.max([len(x) for x in test_mn.markov_gate_input_ids]) <= MarkovNetwork.max_markov_gate_inputs
    assert np.max([len(x) for x in test_mn.markov_gate_output_ids]) <= MarkovNetwork.max_markov_gate_outputs

def test_init_seed_genome():
    """MarkovNetwork initializer with seeded genome"""
    np.random.seed(4303423)
    seed_genome = np.random.randint(0, 256, 10000)
    seed_genome[0:2] = np.array([42, 213])

    test_mn = MarkovNetwork(num_input_states=4,
                            num_memory_states=5,
                            num_output_states=6,
                            probabilistic=False,
                            genome=seed_genome)

    assert np.all(test_mn.genome == seed_genome)
    assert len(test_mn.markov_gates) == 1

def test_init_seed_bad_genome():
    """MarkovNetwork initializer with bad seeded genome"""
    np.random.seed(4303423)
    seed_genome = np.random.randint(0, 256, 10000)
    seed_genome[0:2] = np.array([42, 213])
    seed_genome[-10:-8] = np.array([42, 213])

    test_mn = MarkovNetwork(num_input_states=4,
                            num_memory_states=5,
                            num_output_states=6,
                            probabilistic=False,
                            genome=seed_genome)

    assert np.all(test_mn.genome == seed_genome)
    assert len(test_mn.markov_gates) == 1

def test_activate_network():
    """MarkovNetwork.activate()"""
    np.random.seed(32480)
    test_mn = MarkovNetwork(2, 4, 2)
    test_mn.states[0:2] = np.array([1, 1])
    test_mn.activate_network()
    assert np.all(test_mn.states[-2:] == np.array([1, 0]))

def test_activate_network_bad_input():
    """MarkovNetwork.activate() with bad input"""
    np.random.seed(32480)
    test_mn = MarkovNetwork(2, 4, 2)
    test_mn.states[0:2] = np.array([-7, 2])
    test_mn.activate_network()
    assert np.all(test_mn.states[-2:] == np.array([1, 0]))

def test_update_input_states():
    """MarkovNetwork.test_update_input_states()"""
    np.random.seed(98342)
    test_mn = MarkovNetwork(2, 4, 2)
    test_mn.update_input_states([1, 1])
    assert np.all(test_mn.states[:2] == np.array([1, 1]))

def test_update_input_states_bad_input():
    """MarkovNetwork.test_update_input_states() with bad input"""
    np.random.seed(98342)
    test_mn = MarkovNetwork(2, 4, 2)
    test_mn.update_input_states([-7, 2])
    assert np.all(test_mn.states[:2] == np.array([1, 1]))

def test_update_input_states_invalid_input():
    """MarkovNetwork.test_update_input_states() with invalid input"""
    np.random.seed(98342)
    test_mn = MarkovNetwork(2, 4, 2)
    try:
        test_mn.update_input_states([1, 1, 0])
    except Exception as e:
        assert type(e) is ValueError

def test_get_output_states():
    """MarkovNetwork.get_output_states()"""
    np.random.seed(32480)
    test_mn = MarkovNetwork(2, 4, 2)
    test_mn.update_input_states([1, 1])
    test_mn.activate_network()
    assert np.all(test_mn.get_output_states() == np.array([1, 0]))

def test_get_output_states_bad_input():
    """MarkovNetwork.get_output_states() with bad input"""
    np.random.seed(32480)
    test_mn = MarkovNetwork(2, 4, 2)
    test_mn.update_input_states([-7, 2])
    test_mn.activate_network()
    assert np.all(test_mn.get_output_states() == np.array([1, 0]))
