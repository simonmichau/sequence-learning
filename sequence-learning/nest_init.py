import nest_utils as utils
from GLOBALS import *
import nest

def nest_init():
    # Generate NEST code
    utils.generate_nest_code(NEURON_MODEL, SYNAPSE_MODEL)
    # Verify that module installation was successful
    print(SYNAPSE_MODEL_NAME, " installed: ", SYNAPSE_MODEL_NAME in nest.synapse_models)
    print(NEURON_MODEL_NAME, " installed: ", NEURON_MODEL_NAME in nest.node_models)
    # Set parameters for nest simulator
    nest.resolution = 1.
    nest.set_verbosity('M_ERROR')
    nest.print_time = True
    nest.SetKernelStatus({'resolution': 1.,
                          'use_compressed_spikes': False,
                          "local_num_threads": 6,
                          'rng_seed': rng_seed
                          })  # no touch!
    utils.SYNAPSE_MODEL_NAME = SYNAPSE_MODEL_NAME
