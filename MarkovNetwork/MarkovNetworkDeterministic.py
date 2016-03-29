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

    def __init__(self, num_input_states, num_memory_states, num_output_states, num_markov_gates=4, genome=None):
        """Sets up a randomly-generated deterministic Markov Network

        Parameters
        ----------
        num_input_states: int
            The number of sensory input states that the Markov Network will use
        num_memory_states: int
            The number of internal memory states that the Markov Network will use
        num_output_states: int
            The number of output states that the Markov Network will use
        num_markov_gates: int (default: 4)
            The number of Markov Gates to seed the Markov Network with
            It is important to ensure that randomly-generated Markov Networks have at least a few Markov Gates to begin with
        genome: array-like (optional)
            An array representation of the Markov Network to construct
            All values in the array must be integers in the range [0, 255]
            This option overrides the num_markov_gates option

        Returns
        -------
        None

        """
        self.num_input_states = num_input_states
        self.num_memory_states = num_memory_states
        self.num_output_states = num_output_states
        self.states = np.zeros(num_input_states + num_memory_states + num_output_states)
        self.markov_gates = []
        self.markov_gate_input_ids = []
        self.markov_gate_output_ids = []
        
        if genome is None:
            self.genome = np.random.randint(0, 256, np.random.randint(1000, 5000))

            # Seed the random genome with num_markov_gates Markov Gates
            for _ in range(num_markov_gates):
                start_index = np.random.randint(0, int(len(self.genome) * 0.8))
                self.genome[start_index] = 42
                self.genome[start_index + 1] = 213
        else:
            self.genome = np.array(genome)
            
        self._setup_markov_network()

    def _setup_markov_network(self):
        """Interprets the internal genome into the corresponding Markov Gates

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for index_counter in range(self.genome.shape[0] - 1):
            # Sequence of 42 then 213 indicates a new Markov Gate
            if self.genome[index_counter] == 42 and self.genome[index_counter + 1] == 213:
                internal_index_counter = index_counter + 2
                
                # Determine the number of inputs and outputs for the Markov Gate
                num_inputs = self.genome[internal_index_counter] % 4
                internal_index_counter += 1
                num_outputs = self.genome[internal_index_counter] % 4
                internal_index_counter += 1
                
                # Make sure that the genome is long enough to encode this Markov Gate
                if internal_index_counter + 8 + (2 ** self.num_input_states) * (2 ** self.num_output_states) > self.genome.shape[0]:
                    print('Genome is too short to encode this Markov Gate -- skipping')
                    continue
                
                # Determine the states that the Markov Gate will connect its inputs and outputs to
                input_state_ids = self.genome[internal_index_counter:internal_index_counter + 4][:self.num_input_states]
                internal_index_counter += 4
                output_state_ids = self.genome[internal_index_counter:internal_index_counter + 4][:self.num_output_states]
                internal_index_counter += 4
                
                self.markov_gate_input_ids.append(input_state_ids)
                self.markov_gate_output_ids.append(output_state_ids)
                
                markov_gate = self.genome[internal_index_counter:internal_index_counter + (2 ** self.num_input_states) * (2 ** self.num_output_states)]
                markov_gate = markov_gate.reshape((2 ** self.num_input_states, 2 ** self.num_output_states))
                
                for row_index in range(markov_gate.shape):
                    row_max = markov_gate[row_index, :].max()
                    markov_gate[row_index, :] = np.zeros()
                break

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
            len(sensory_input) must be equal to num_input_states

        Returns
        -------
        None

        """
        if len(sensory_input) != self.num_input_states:
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


if __name__ == '__main__':
    np.random.seed(29382)
    test = MarkovNetworkDeterministic(2, 4, 3)
