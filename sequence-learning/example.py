from GLOBALS import *
from nest_network import *
from nest_inp_gen import *
from nest_recorder import *
from nest_utils import *

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
# Initialize a new Network object
grid = Network(grid_shape=(10, 5), k_min=2, k_max=10, n_inputs=100)

# Visualize the network in 2D and 3D
grid.visualize_circuits()
grid.visualize_circuits_3d_barchart()

# Set the max_neuron_gid property (important!)
grid.get_node_collections().max_neuron_gid = max(grid.get_node_collections().global_id)  # critical

# Initialize a new InputGenerator for our Network
inpgen = InputGenerator(grid, r_noise=5, r_input=5, r_noise_pattern=2, use_noise=True, t_noise_range=[500.0, 800.0],
                        n_patterns=2, t_pattern=[300.]*2, pattern_sequences=[[0],[1]])

# Initialize a new Recorder for our Network
recorder = Recorder(grid, save_figures=False, show_figures=True, create_plot=False)
# id_list = recorder.run_network(inpgen=inpgen, t_sim=1, dt_rec=None, title="Simulation #1") #readout_size=30

# Set a property of the Recorder to a new value
recorder.set(create_plot=True)

# Run a simulation
id_list = recorder.run_network(inpgen=inpgen, t_sim=3000, dt_rec=None, title="Test #1", train=True, order_neurons=False)
# Run another simulation and record from the same neurons as in the last run
recorder.run_network(inpgen=inpgen, id_list=id_list, t_sim=1000, dt_rec=100, title="Test #2", train=True, order_neurons=False)

# Verify rate fractions
recorder.verify_rate_fractions()
# Reset the recorders
recorder.reset_recorders(inpgen)

# Run a last simulation and also plot the previous runs
recorder.set(plot_history=True)
recorder.run_network(inpgen=inpgen, t_sim=3000, dt_rec=None, title="History", train=False, order_neurons=True)
