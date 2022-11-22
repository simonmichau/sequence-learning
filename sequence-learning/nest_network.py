import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import sys
import nest

matplotlib.use('TkAgg')
sys.path.append('../../')

from GLOBALS import *
import nest_utils as utils
from nest_inp_gen import *
from nest_recorder import *
# from nest_utils import *
# from pynestml.frontend.pynestml_frontend import *

# rng_seed = np.random.randint(0, 1000, 1)[0]
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": rng_seed})
np.random.seed(rng_seed)


class WTACircuit:
    """
    Object to store a NodeCollection for a 2D position. All neurons in the object are connected to each other using
    RATE_CONN_SYN_MODEL connections to allow lateral inhibition.
    """
    def __init__(self, nc, pos):
        self.nc = nc
        self.pos = pos
        self.k = self.get_size()
        self.form_WTA()

    def get_pos(self):
        """Returns the (x, y) position of the WTA circuit"""
        return self.pos

    def get_x(self):
        """Returns the x coordinate of the WTA circuit"""
        return self.pos[0]

    def get_y(self):
        """Returns the y coordinate of the WTA circuit"""
        return self.pos[1]

    def get_node_collection(self):
        """Returns the NodeCollection nc"""
        return self.nc

    def get_size(self):
        """Returns the size of the NodeCollection nc"""
        try:
            return len(self.nc.get('global_id'))
        except TypeError:
            return 1

    def form_WTA(self):
        """Connects all neurons within the same WTA via rate_connection_instantaneous connections"""
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    nest.Connect(self.nc[i], self.nc[j], "all_to_all", {"synapse_model": RATE_CONN_SYN_MODEL})


class Network(object):
    def __init__(self, **kwds):
        # FUNCTIONAL VARIABLES
        # Dimensions of the grid of WTA circuits
        self.grid_shape = kwds.get('grid_shape', (10, 5))
        self.n, self.m = self.grid_shape
        # Upper and lower bound for randomly drawn number k of neurons in each WTA circuit
        self.k_min = kwds.get('k_min', 2)
        self.k_max = kwds.get('k_max', 10)
        # Number of external input channels
        self.n_inputs = kwds.get('n_inputs', 50)
        # parameter lambda of exponential distance distribution
        self.lam = kwds.get('lam', 0.088)
        self.n_conn = 1000
        # List containing all WTA circuits
        self.circuits = []
        # NodeCollection containing all neurons of the grid
        self.neuron_collection = None
        # ADMINISTRATIVE VARIABLES
        self.save_figures = kwds.get('save_figures', True)
        self.show_figures = kwds.get('show_figures', True)
        # Whether STP & STDP should be enabled for recurrent connections
        self.use_stp = True
        self.use_stdp = True
        # Create WTA circuits
        self.create_grid()
        # Establish interneuron connections
        self.form_connections()

        # Initialize recorders
        self.weight_recorder = list()
        self.epsp_recorder = list()
        self.weight_recorder_manual = list()
        self.multimeter = None
        self.spikerecorder = None

    def get_circuit_grid(self) -> np.ndarray:
        """Returns a (nxm) array containing the neuron frequencies per grid point"""
        data = np.zeros((self.m, self.n))
        for circuit in self.circuits:
            data[circuit.get_pos()[1], circuit.get_pos()[0]] = circuit.get_size()
        return data

    def get_node_collections(self, slice_min=None, slice_max=None):
        """Return a slice of **self.circuits** as a **NodeCollection**"""
        if slice_min is None:
            slice_min = 0
        if slice_max is None:
            slice_max = len(self.circuits) + 1
        s = slice(slice_min, slice_max)

        id_list = []
        for circuit in self.circuits[s]:
            try:
                id_list += circuit.get_node_collection().get()['global_id']
            except TypeError:
                id_list += [circuit.get_node_collection().get()['global_id']]
        return nest.NodeCollection(id_list)

    def get_pos_by_id(self, node_id: int):
        """Returns the position of the WTA circuit which contains the node with the given ID"""
        for i in self.circuits:
            if node_id in i.nc.get()['global_id']:
                return i.get_pos()

    def get_wta_by_id(self, node_id: int):
        """Returns the WTA circuit which contains the node with the given ID"""
        for i in self.circuits:
            if node_id in i.nc.get()['global_id']:
                return i
        raise RuntimeError(f"Neuron {node_id} not in any circuit?")

    def refresh_neurons(self):
        """Refreshes self.neurons based on self.circuits"""
        self.neuron_collection = self.get_node_collections()

    def create_grid(self) -> list:
        """
        Create a **WTACircuit** object for every point on the (nxm) grid and returns all those objects in a list

        - **K**: number of neurons in a WTA circuit, randomly drawn with lower and upper bound [k_min, k_max]
        """
        circuit_list = []
        for m in range(self.m):
            for n in range(self.n):
                K = np.random.randint(self.k_min, self.k_max + 1)
                nc = nest.Create(NEURON_MODEL_NAME, K,
                                 {'tau_m': 20.0, 'use_variance_tracking': 1.,
                                  'use_stdp': int(self.use_stdp), 'rate_fraction': 1./K})
                circuit_list.append(WTACircuit(nc, (n, m)))
                print(f"Position and size of WTA circuit: ({n}, {m}) - {K}")
        self.circuits = circuit_list
        self.refresh_neurons()
        return circuit_list

    def form_connections(self) -> None:
        """Connect every WTA circuit """
        conn_dict = {'rule': 'pairwise_bernoulli',
                     'p': 1.0,
                     'allow_autapses': False}
        syn_dict = {
            "synapse_model": SYNAPSE_MODEL_NAME,
            'delay': 1.,
            'use_stp': int(self.use_stp)
        }
        # Create distance matrix
        self.distances = np.zeros((len(self.circuits), len(self.circuits)))
        for i in range(len(self.circuits)):
            self.circuits[i].get_pos()
            for j in range(len(self.circuits)):
                self.distances[i][j] = math.sqrt((self.circuits[i].get_x() - self.circuits[j].get_x()) ** 2
                                                 + (self.circuits[i].get_y() - self.circuits[j].get_y()) ** 2)

        # Iterate over each WTACircuit object and establish connections to every other population with distance
        # dependent probability p(d)
        eps = 1
        count = 0  # circuit-to-circuit connections
        while count < self.n_conn:
            d = np.random.exponential(scale=1.0 / self.lam)
            I, J = np.where((self.distances <= d + eps) & (self.distances > d - eps))
            if len(I) > 0:
                i = np.random.randint(len(I))
                if I[i] != J[i]:
                    conns = nest.Connect(self.circuits[I[i]].get_node_collection(),
                                         self.circuits[J[i]].get_node_collection(),
                                         conn_dict, syn_dict, return_synapsecollection=True)

                    U_mean = 0.5
                    tau_d_mean = 110.
                    tau_f_mean = 5.

                    Us = U_mean + U_mean / 2 * np.random.randn(len(conns))
                    tau_ds = tau_d_mean + tau_d_mean / 2 * np.random.randn(len(conns))
                    tau_fs = tau_f_mean + tau_f_mean / 2 * np.random.randn(len(conns))
                    conns.set({
                        'U': np.maximum(Us, 0),
                        'tau_d': np.maximum(tau_ds, 1.),
                        'tau_f': np.maximum(tau_fs, 1.),
                    })
                    count += 1
        print(f"Created {count} circuit-to-circuit connections.")

        # Randomize weights of each WTA circuit
        for i in range(len(self.circuits)):
             utils.randomize_outgoing_connections(self.circuits[i].get_node_collection())

    def visualize_circuits(self) -> None:
        """Creates a **pcolormesh** visualizing the number of neurons k per WTA circuit on the grid"""
        data = self.get_circuit_grid()

        fig, ax = plt.subplots()
        im = ax.imshow(data)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(self.n))
        ax.set_yticks(np.arange(self.m))

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(self.m):
            for j in range(self.n):
                text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")

        ax.set_title("Number of neurons of each WTA circuit on the (%dx%d) grid" % (self.n, self.m))
        ax.set_xlabel("%d neurons total" % np.sum(data))
        fig.tight_layout()

        if self.save_figures:
            plt.savefig("grid_visualization.png")
        if self.show_figures:
            plt.show()

    def visualize_circuits_3d(self) -> None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        x = []
        y = []
        z = []
        data = self.get_circuit_grid()
        for i in range(len(self.get_circuit_grid())):
            for j in range(len(self.get_circuit_grid()[i])):
                x.append(j)
                y.append(i)
                z.append(data[i][j])

        # Trimesh
        ax.plot_trisurf(x, y, z, color='blue')
        # Scatterplot
        ax.scatter3D(x, y, z, c=z, cmap='cividis')
        # Select Viewpoint
        ax.view_init(30, -90)
        if self.save_figures:
            plt.savefig("grid_visualization_3d.png")
        if self.show_figures:
            plt.show()

    def visualize_circuits_3d_barchart(self):
        # setup the figure and axes
        fig = plt.figure(figsize=(5, 3))
        ax1 = fig.add_subplot(111, projection='3d')
        #ax2 = fig.add_subplot(122, projection='3d')

        # fake data
        data = self.get_circuit_grid()
        _x = np.arange(10)
        _y = np.arange(5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = data.flatten()
        bottom = np.zeros_like(top)
        width = depth = 1

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
        #ax1.set_title('Shaded')
        ax1.set_xlabel("N")
        ax1.set_ylabel("M")
        ax1.set_zlabel("Neurons")

        #ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
        #ax2.set_title('Not Shaded')

        plt.show()
