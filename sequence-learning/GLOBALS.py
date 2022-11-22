"""Global variables"""

NEURON_MODEL = 'iaf_psc_exp_wta'
SYNAPSE_MODEL = 'stdp_stp'
RATE_CONN_SYN_MODEL = 'rate_connection_instantaneous'
NEURON_MODEL_NAME = NEURON_MODEL + "__with_" + SYNAPSE_MODEL
SYNAPSE_MODEL_NAME = SYNAPSE_MODEL + "__with_" + NEURON_MODEL

rng_seed = 3  # random number generator seed
