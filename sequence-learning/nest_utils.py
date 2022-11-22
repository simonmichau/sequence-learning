import numpy as np
import nest
try:
    from pynestml.frontend.pynestml_frontend import generate_nest_target
except:
    pass
from GLOBALS import *


def generate_poisson_spiketrain(t_duration, rate) -> list:
    """Generates a list of poisson generated spike times for a given

    :rtype: list

    - duration **t_duration** (in ms) and
    - firing rate **rate** (in Hz)
    """
    n = t_duration * rate * 1e-3  # expected number of spikes in [0, t_duration)
    scale = 1 / (rate * 1e-3)  # expected time between spikes
    isi = np.random.exponential(scale, int(np.ceil(n)))  # list of [ceil(n)] input spike intervals
    spikes = np.add.accumulate(isi)
    # Hypothetical position of t_duration in spikes list
    i = np.searchsorted(spikes, t_duration)

    # Add or remove spikes
    extra_spikes = []
    if i == len(spikes):
        t_last = spikes[-1]
        while True:
            t_last += np.random.exponential(scale, 1)[0]
            if t_last >= t_duration:
                break
            else:
                extra_spikes.append(t_last)
        spikes = np.concatenate((spikes, extra_spikes))
    else:
        # Cutoff spike times outside of spike duration
        spikes = np.resize(spikes, (i,))  # spikes[:i]
    a = spikes
    b = list(spikes)
    return spikes.tolist()


def randomize_outgoing_connections(nc):
    """Randomizes the weights of outgoing connections of a NodeCollection **nc**"""
    conns = nest.GetConnections(nc)
    random_weight_list = []
    for i in range(len(conns)):
        random_weight_list.append(-np.log(np.random.rand()))
    conns.set(weight=random_weight_list)


def disable_stdp(nc):
    """Disables STDP for a given NodeCollection"""
    nc.set({'use_stdp': float(False)})


def enable_stdp(nc):
    """Disables STDP for a given NodeCollection"""
    nc.set({'use_stdp': float(True)})


def disable_stp(nc):
    synapses = nest.GetConnections(nc, synapse_model="stdp_stp__with_iaf_psc_exp_wta")
    synapses.set({'use_stp': float(False)})


def enable_stp(nc):
    synapses = nest.GetConnections(nc, synapse_model="stdp_stp__with_iaf_psc_exp_wta")
    synapses.set({'use_stp': float(True)})


def update_presyn_ids(network):
    """
    For each neuron, update the ids the presynaptic sources. This is needed for accurate weight updates as in Klampfl.
    """
    assert SYNAPSE_MODEL_NAME is not None
    node_ids = network.get_node_collections()

    for gid in node_ids:
        sources = nest.GetConnections(target=gid, synapse_model=SYNAPSE_MODEL_NAME).source
        gid.set({'presyn_ids': np.array(sources).astype(float)})


def generate_nest_code(neuron_model: str, synapse_model: str, target="nestml_target"):
    """Generates the code for 'iaf_psc_exp_wta' neuron model and 'stdp_stp' synapse model."""
    module_name = "nestml_modified_master_module"
    target += '__modmaster'

    nest.Install(module_name)
    mangled_neuron_name = neuron_model + "__with_" + synapse_model
    mangled_synapse_name = synapse_model + "__with_" + neuron_model
    print("Created ", mangled_neuron_name, " and ", mangled_synapse_name)
