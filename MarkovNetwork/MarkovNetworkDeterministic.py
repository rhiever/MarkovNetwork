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

from ._version import __version__

class MarkovNetworkDeterministic(object):

    """A deterministic Markov Network for neural computing."""

    def __init__(self, num_sensor_states, num_memory_states, num_output_states, num_markov_gates=4):
        """Sets up a randomly-generated deterministic Markov Network

        Parameters
        ----------
        num_sensor_states: int
            The number of sensory input states that the Markov Network will use
        num_memory_states: int
            The number of internal memory states that the Markov Network will use
        num_output_states: int
            The number of output states that the Markov Network will use
        num_markov_gates: int (default: 4)
            The number of Markov Gates to seed the Markov Network with
            It is important to ensure that randomly-generated Markov Networks have at least a few Markov Gates to begin with

        Returns
        -------
        None

        """
        self.num_sensor_states = num_sensor_states
        self.num_memory_states = num_memory_states
        self.num_output_states = num_output_states
        self.states = np.zeros(num_sensor_states + num_memory_states + num_output_states)
        self.markov_gates = []
        self.genome = np.random.randint(0, 256, np.random.randint(1000, 5000))
        
        # Seed the random genome with num_markov_gates Markov Gates
        for _ in range(num_markov_gates):
            start_index = np.random.randint(0, int(len(self.genome) * 0.8))
            self.genome[start_index] = 42
            self.genome[start_index + 1] = 213

    def __init__(self, num_sensor_states, num_memory_states, num_output_states, genome):
        """Sets up a deterministic Markov Network using the provided genome

        Parameters
        ----------
        num_sensor_states: int
            The number of sensory input states that the Markov Network will use
        num_memory_states: int
            The number of internal memory states that the Markov Network will use
        num_output_states: int
            The number of output states that the Markov Network will use
        genome: array-like
            Array representation of the Markov Network
            All values in the array must be integers in the range [0, 255]

        Returns
        -------
        None

        """
        self.num_sensor_states = num_sensor_states
        self.num_memory_states = num_memory_states
        self.num_output_states = num_output_states
        self.states = np.zeros(num_sensor_states + num_memory_states + num_output_states)
        self.markov_gates = []
        self.genome = genome

    def setup_markov_network(self):
        """Interprets the internal genome into the corresponding Markov Gates

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass

    def activate_network(self):
        """Activates the Markov Network

        Parameters
        ----------
        ggg: type (default: ggg)
            ggg

        Returns
        -------
        None

        """
        pass

    def update_sensor_states(self, sensory_input):
        """Updates the sensor states with the provided sensory inputs

        Parameters
        ----------
        sensory_input: array-like
            An array of integers containing the sensory inputs for the Markov Network
            len(sensory_input) must be equal to num_sensor_states

        Returns
        -------
        None

        """
        if len(sensory_input) != self.num_sensor_states:
            raise ValueError('Invalid number of sensory inputs provided')
        pass
        
    def get_output_states(self):
        """Returns an array of the current output state's values

        Parameters
        ----------
        None

        Returns
        -------
        output_states: array-like
            An array of the current output state's values

        """
        return self.states[-self.num_output_states:]