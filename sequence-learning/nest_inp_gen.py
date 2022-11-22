import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import sys
import nest

matplotlib.use('TkAgg')
sys.path.append('../../')

import nest_utils as utils
from GLOBALS import *

class InputGenerator(object):
    """
    Contains all functionality to stimulate an assigned target network. Takes keywords:

    - **r_noise** [Hz]: Noise firing rate
    - **r_input** [Hz]: Input firing rate
    - **n_patterns**: Number of different patterns
    - **pattern_sequences**: List of pattern sequences
    - **pattern_mode**: Mode of how patterns are sampled from *pattern_sequences* during presentation. Can be either
      ``random_iterate`` or ``random_independent``
    - **p_switch**: Switching probability for *pattern_mode*
    - **t_pattern** [ms]: List containing the durations for all patterns
    - **t_noise_range** [ms]: Range from which the noise phase duration is randomly chosen from
    - **use_noise** [Bool]: States whether noise should be produced or not
    - **use_input** [Bool]: States whether inputs should be produced or not
    """
    def __init__(self, target_network, **kwds):
        # Number of input channels
        self.n = target_network.n_inputs
        # Target network
        self.target_network = target_network
        # Poisson firing rate of the noise during pattern presentation (in Hz)
        self.r_noise_pattern = kwds.get('r_noise_pattern', 4)
        # Standard poisson firing rate of the noise (in Hz)
        self.r_noise = kwds.get('r_noise', 5)  # poisson noise rate during pattern presentation
        # Input firing rate (in Hz)
        self.r_input = kwds.get('r_input', 3)
        # Number of patterns
        self.n_patterns = kwds.get('n_patterns', 3)

        # Pattern sequences (contains lists of pattern sequences; their presentation order is determined [elsewhere])
        self.pattern_sequences = kwds.get('pattern_sequences', [[0, 1], [2]])
        # Pattern mode (can be either 'random_independent' or 'random_iterate')
        self.pattern_mode = kwds.get('pattern_mode', 'random_iterate')
        # Switching probability for pattern picking
        self.p_switch = kwds.get('p_switch', 1.0)
        # Pattern durations
        self.t_pattern = kwds.get('t_pattern', [300.0] * self.n_patterns)
        # Range from which the noise phase duration is randomly chosen from (in ms)
        self.t_noise_range = kwds.get('t_noise_range', [100.0, 500.0])
        # Dictionary of stored patterns
        self.pattern_list = []
        # Spiketrain for all n input channels
        self.spiketrain = [[]] * self.n
        # Tuple storing the sequence index and the index of the current pattern
        self.current_pattern_index = [0, 0]

        # Parameters used for `get_order`
        # List of time points where the phase changes (noise, patternA, patternB, ...)
        self.phase_times = [0]
        # List of length [duration] containing the index of the pattern for each time step (or -1 if noise)
        self.pattern_trace = []
        self.next_pattern_length = [self.t_pattern[0]]

        # NodeCollection of spike_generators
        self.spike_generators = None
        # NodeCollection of inhomogeneous_poisson_generators
        self.poisson_generators = None
        # NodeCollection of parrot_neurons used for poisson noise and input patterns
        self.parrots = nest.Create('parrot_neuron', self.n)
        self.connect_parrots()  # connect parrots to network with stdp and w/o stp

        # Spikerecorder for poisson noise
        self.noiserecorder = None

        self.use_noise = kwds.get('use_noise', True)
        # self.use_input = kwds.get('use_input', True)

        # Create noise
        if self.use_noise:
            self.generate_noise()

    def connect_parrots(self):
        # Connect parrots to target network
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'p': 1.0,
            'allow_autapses': False,
        }
        syn_dict = {"synapse_model": SYNAPSE_MODEL_NAME,
                    'delay': 1.,
                    'U': 0.5,
                    'u': 0.5,
                    'use_stp': 0  # TODO for some reason, input synapses are not dynamic.
                    }
        nest.Connect(self.parrots, self.target_network.get_node_collections(), conn_dict, syn_dict)

    def generate_noise(self) -> None:
        """Creates and connects poisson generators to target network to stimulate it with poisson noise."""
        # Create n poisson input channels with firing rate r_noise
        self.poisson_generators = nest.Create('inhomogeneous_poisson_generator', self.n)
        # Connect one poisson generator to each parrot neuron
        nest.Connect(self.poisson_generators, self.parrots, 'one_to_one')

        # Update connection weights to random values
        utils.randomize_outgoing_connections(self.parrots)

    def create_patterns(self) -> None:
        """Creates poisson patterns according to the InputGenerator's

        - number of patterns **n_patterns**,
        - pattern durations **t_pattern** (in ms),
        - pattern firing rate **r_input** (in Hz)

        and stores them in **pattern_list**."""
        self.pattern_list = []
        for i in range(self.n_patterns):
            pattern = []
            for j in range(self.n):
                pattern.append(utils.generate_poisson_spiketrain(self.t_pattern[i], self.r_input))
            self.pattern_list.append(pattern)

    def generate_input(self, duration, t_origin=0.0, force_refresh_patterns=False):
        """Generates Input for a given duration. Needs to be run for every simulation

        - duration: duration of input (in ms)
        -
        """
        # Create new patterns if none have been created yet, or it is demanded explicitly
        if not self.pattern_list or force_refresh_patterns:
            self.create_patterns()

        if self.spike_generators is None:
            # create n spike_generators if none exist yet
            self.spike_generators = nest.Create('spike_generator', self.n, params={'allow_offgrid_times': False,
                                                                                   'origin': t_origin})
            # Connect spike generators to parrots
            nest.Connect(self.spike_generators, self.parrots, 'one_to_one')

            # Randomize connection weights
            utils.randomize_outgoing_connections(self.spike_generators)  # TODO Is this needed?

        noise_rate_times = []
        noise_rate_values = []
        # generate a list of spiketrains that alternate between noise phase and pattern presentation phase
        t = nest.biological_time
        spiketrain_list = [[]] * self.n  # list to store the spiketrain of each input channel
        current_pattern_id = self.pattern_sequences[self.current_pattern_index[0]][self.current_pattern_index[1]]
        while t < nest.biological_time + duration:
            # Randomly draw the duration of the noise phase
            t_noise_phase = self.t_noise_range[0] + np.random.rand()*(self.t_noise_range[1]-self.t_noise_range[0])
            t_noise_phase = np.round(t_noise_phase, decimals=0)

            # Get noise and pattern times for poisson gens
            noise_rate_times += [t + nest.resolution]
            noise_rate_values += [self.r_noise]  # noise rate during noise phase
            noise_rate_times += [t+t_noise_phase]
            noise_rate_values += [self.r_noise_pattern]  # noise rate during pattern presentation

            # append pattern spike times to spiketrain list
            for i in range(self.n):  # iterate over input channels
                st = np.add(t+t_noise_phase, self.pattern_list[current_pattern_id][i])
                spiketrain_list[i] = spiketrain_list[i] + st.tolist()

            # Append phase times (for get_order)
            self.phase_times += [int(t+t_noise_phase), int(t+t_noise_phase + self.t_pattern[current_pattern_id])]
            # Append next pattern length (for get_order)
            self.next_pattern_length += [int(self.t_pattern[current_pattern_id]), int(self.t_pattern[current_pattern_id])]
            # Append pattern trace (for get_order)
            self.pattern_trace += [-1]*int(t_noise_phase)  # append noise id
            self.pattern_trace += [current_pattern_id]*int(self.t_pattern[current_pattern_id])  # append input pattern id

            t += t_noise_phase + self.t_pattern[current_pattern_id]

            # Update the pattern to present next round
            current_pattern_id = self.get_next_pattern_id()

        # cutoff values over t=origin+duration
        t_threshold = nest.biological_time + duration
        for i in range(len(spiketrain_list)):
            threshold_index = np.searchsorted(spiketrain_list[i], t_threshold)
            spiketrain_list[i] = spiketrain_list[i][0: threshold_index]

        for i in range(len(self.spiketrain)):
            self.spiketrain[i] = self.spiketrain[i] + spiketrain_list[i]
            self.spiketrain[i] = np.unique(self.spiketrain[i]).tolist()  # avoid redundant spike times

        # Set noise and pattern times for poisson gens
        if self.use_noise:
            self.poisson_generators.set({'rate_times': noise_rate_times, 'rate_values': noise_rate_values})

        # Assign spiketrain_list to spike_generators
        for i in range(self.n):
            self.spike_generators[i].spike_times = np.round(self.spiketrain[i], decimals=0)  # TODO fix non descending order issue

    def get_next_pattern_id(self) -> int:
        # if sequence is not over just progress to next id in sequence
        if not self.current_pattern_index[1] + 1 >= len(self.pattern_sequences[self.current_pattern_index[0]]):
            self.current_pattern_index[1] += 1
        else:
            # if sequence is over pick new sequence from pattern_sequences using rules
            if self.pattern_mode == 'random_independent':
                print("Error: random_independent switching mode not implemented yet")  # TODO
            elif self.pattern_mode == 'random_iterate':
                # with probability p_switch move on to the next sequence/repeat the current sequence with 1-p_switch
                if np.random.rand() <= self.p_switch:
                    self.current_pattern_index[0] = (self.current_pattern_index[0]+1) % len(self.pattern_sequences)
                self.current_pattern_index[1] = 0  # reset index to beginning of sequence
        return self.pattern_sequences[self.current_pattern_index[0]][self.current_pattern_index[1]]

    def get_patterns(self):
        return self.pattern_list

    def set_patterns(self, patterns):
        self.pattern_list = []
        self.pattern_list += patterns

    def visualize_spiketrain(self, st):
        """Visualizes a given spiketrain"""
        fig = plt.figure()
        fig, ax = plt.subplots()
        for i in range(len(st)):
            ax.scatter(st[i], [i]*len(st[i]), color=(i/(len(st)), 0.0, i/(len(st))))
            # ax.plot(st[i], [i]*len(st[i]), ".", color='orange')
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Input channels")
        ax.axis([0, np.amax(st)[0]*1.5, -1, self.n])