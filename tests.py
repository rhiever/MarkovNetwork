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
    """Ensure that the Markov Network initilizer works"""
    
    test_mn = MarkovNetwork(num_input_states=4,
                            num_memory_states=5,
                            num_output_states=6,
                            seed_num_markov_gates=2,
                            probabilistic=False,
                            genome=None)

    assert test_mn.num_input_states == 4
    assert test_mn.num_memory_states == 5
    assert test_mn.num_output_states == 6
    assert len(test_mn.states) == 4+5+6
    assert len(test_mn.markov_gates) == 2
    assert np.max([len(x) for x in test_mn.markov_gate_input_ids]) <= MarkovNetwork.max_markov_gate_inputs
    assert np.max([len(x) for x in test_mn.markov_gate_output_ids]) <= MarkovNetwork.max_markov_gate_outputs
