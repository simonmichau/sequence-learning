import pprint
import tqdm
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import nest

from GLOBALS import *
from nest_utils import disable_stp, enable_stp, disable_stdp, enable_stdp, update_presyn_ids


class Recorder:
    """
    Recorder class
    """
    def __init__(self, network, id_list: list = None, **kwargs):
        # Network to record from
        self.network = network
        # global IDs of neurons to record
        self.id_list = id_list
        # number of recorded neurons
        if self.id_list is not None:
            self.n_rec_neurons = len(id_list)
        else:
            self.n_rec_neurons = 0

        self.create_plot = kwargs.get('create_plot', True)  # Note that the following plot/figure related parameters are redundant if False
        self.save_figures = kwargs.get('save_figures', False)
        self.show_figures = kwargs.get('show_figures', True)
        self.plot_history = kwargs.get('plot_history', False)
        self.order_neurons = kwargs.get('order_neurons', True)

        # Recording time interval
        self.dt_rec = kwargs.get('dt_rec', None)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def verify_rate_fractions(self):
        print("Verifying rate fraction consistency...")
        for c in self.network.circuits:
            rfracs = np.empty((0, self.network.multimeter[0].get('n_events')))
            for gid in c.nc.global_id:
                rfracs = np.append(rfracs, np.array([np.array(self.network.multimeter[gid - 1].get('events')['rate_fraction'])]), axis=0)

            if not np.all(np.isclose(np.sum(rfracs[:, 2:], axis=0), 1.)):
                print(np.sum(rfracs[:, 2:], axis=0)[-10:])

            assert np.all(np.isclose(np.sum(rfracs[:, 2:], axis=0), 1.))

        print("Rate fraction consistency passed!")

    def reset_recorders(self, inpgen):
        """
        Resets recording devices
        """
        self.network.multimeter.set(n_events=0)
        self.network.spikerecorder.set(n_events=0)
        inpgen.noiserecorder.set(n_events=0)

    def run_network(self, id_list: list = None, node_collection=None, readout_size: int = None,
                    inpgen=None, t_sim: float = 5000.0, title=None, train=True, dt_rec=None, order_neurons=True):
        """
        Simulates given **NodeCollection** for **t_sim** and plots the recorded spikes, membrane potential and presented
        patterns. Requires an **InputGenerator** object for pattern input generation.

        Readout modes (listed after priority):
            1. "Watch List": a list of Node IDs is specified and measured. Useful when observing readout over several measurements
            2. "Node Collection": A specified NodeCollection is measured
            3. "Random k": k nodes are randomly sampled from the network and measured
            4. "All": Measure all nodes in network
        """
        assert SYNAPSE_MODEL_NAME is not None
        NUMBER_OF_SPIKES = {}

        # Determine NodeCollection to record from (and also id_list for return value)
        if id_list is not None:  # Readout Mode 1
            node_collection = nest.NodeCollection(id_list)
        elif node_collection is not None:  # Readout Mode 2
            id_list = list(node_collection.get('global_id'))
        elif readout_size is not None:  # Readout Mode 3
            global_ids = self.network.get_node_collections().get('global_id')
            id_list = []
            while len(id_list) < readout_size:
                id_list.append(random.randrange(min(global_ids), max(global_ids)))
                id_list = list(set(id_list))  # remove duplicates
            id_list.sort()
            node_collection = nest.NodeCollection(id_list)
        else:  # Readout Mode 4
            node_collection = self.network.get_node_collections()
            id_list = list(self.network.get_node_collections().get('global_id'))
        self.id_list = id_list
        self.n_rec_neurons = len(self.id_list)

        # Create new multimeter and spikerecorder if none exist yet and connect them to the node_collection
        if self.network.multimeter is None:
            self.network.multimeter = nest.Create('multimeter', len(node_collection))
            self.network.multimeter.set(record_from=['V_m', 'rate_fraction'])
            nest.Connect(self.network.multimeter, node_collection, 'one_to_one')
        if self.network.spikerecorder is None:  # TODO
            self.network.spikerecorder = nest.Create('spike_recorder', len(node_collection))
            nest.Connect(node_collection, self.network.spikerecorder, 'one_to_one')
        if inpgen.noiserecorder is None and inpgen is not None:
            if inpgen.use_noise:
                inpgen.noiserecorder = nest.Create('spike_recorder')
                nest.Connect(inpgen.parrots, inpgen.noiserecorder)

        # Run simulation (with or without input)
        if inpgen is None:
            nest.Simulate(t_sim)
        else:
            if train:
                self.simulate(inpgen, t_sim, dt_rec)
            else:
                self.test(inpgen, t_sim, dt_rec)

        if self.create_plot:
            print("Plotting...")
            start_time = time.time()
            # Initialize plot
            fig, axes = plt.subplots(2, 1, sharex=not self.plot_history)  # only sync x axis if not the whole history is observed
            fig.set_figwidth(8)
            fig.set_figheight(6)

            # # MULTIMETER
            # dmm = self.network.multimeter.get()
            # Vms = dmm["events"]["V_m"]
            # rend = dmm["events"]["rate_fraction"]
            # ts = dmm["events"]["times"]
            #
            # # SPIKERECORDER
            # dSD = self.network.spikerecorder.get("events")
            # evs = dSD["senders"]
            # ts_ = dSD["times"]
            #
            # # NOISERECORDER
            # if inpgen.use_noise:
            #     nr = inpgen.noiserecorder.get("events")
            #     evs__ = nr["senders"]
            #     ts__ = nr["times"]
            #
            # # filter the indices after t_sim_start
            t_sim_start = nest.biological_time - t_sim
            # multimeter_time_window = np.where(ts > t_sim_start)[0]
            # spikerecorder_time_window = np.where(ts_ > t_sim_start)[0]
            # if inpgen.use_noise:
            #     noiserecorder_time_window = np.where(ts__ > t_sim_start)[0]

            rend_events = self.network.multimeter.get('events')
            # order the neurons by their mean activation time
            if order_neurons:
                print("Ordering neurons...")
                p = np.array(inpgen.phase_times)
                I = np.array(inpgen.pattern_trace)
                t = np.array(inpgen.next_pattern_length)
                neuron_order, _, _ = self.get_order(p, I, t, rend_events, tstart=int(t_sim_start), nsteps=int(t_sim))
                print(neuron_order)
                print("Done ordering neurons.")
            else:
                # neuron_order = np.unique(dmm["events"]["senders"])
                neuron_order = np.array(self.network.get_node_collections().global_id) - 1

            # n_senders = len(np.unique(dmm["events"]["senders"]))
            # all_senders = np.unique(dmm["events"]["senders"])
            # neuron_order = neuron_order[:100]

            data_multimeter = self.network.multimeter.get('events')
            data_spike_recorder = self.network.spikerecorder.get('events')
            n_plot_weights = 20
            cnt_weight_plot = 0

            # neuron_gids = self.network.get_node_collections().global_id
            for plot_idx, sorted_neuron_idx_rel0 in tqdm.tqdm(enumerate(neuron_order), desc="Sorting neurons:",
                                                              total=len(neuron_order)): #enumerate(np.unique(dmm["events"]["senders"])):  # iterate over all sender neurons
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

                # neuron_gid = neuron_gids[sorted_neuron_idx_rel0]
                # start_time_1 = time.time()


                # SPIKES
                # indices = np.where(dSD["senders"] == neuron_gid)[0]
                # if not self.plot_history:  # remove all indices from outside of spikerecorder_time_window
                    # indices = [i for i in indices if i in spikerecorder_time_window]
                    # indices = indices[np.where((indices >= spikerecorder_time_window.min())
                    #                            & (indices <= spikerecorder_time_window.max()))[0]]
                # axes[1].plot(ts_[indices], evs[indices], ".", ms=1)
                # axes[1].plot(ts_[indices], [idx] * len(ts_[indices]), ".", ms=3)
                ts_spikes = data_spike_recorder[sorted_neuron_idx_rel0]['times']  # spike times of current neuron
                axes[0].plot(ts_spikes, [plot_idx] * len(ts_spikes), ".", ms=3)  # use plot_idx as y index

                # neuron_wta_pos = self.network.get_pos_by_id(neuron_gid)
                # if neuron_wta_pos not in NUMBER_OF_SPIKES:
                #     if neuron_wta_pos is None:
                #         print(f"Neuron {neuron_gid} is in WTA with invalid position", neuron_gid)
                #         assert neuron_wta_pos is not None, f"Neuron {neuron_gid} is in WTA with invalid position"
                #     NUMBER_OF_SPIKES[neuron_wta_pos] = 0
                # NUMBER_OF_SPIKES[neuron_wta_pos] += len(ts_spikes)

                # run_time = time.time() - start_time_1
                #print("Membrane potential and Spikes complete in %s" % run_time)

                #print("EPSPs and weights complete in %s" % run_time)

            # SPIKES FROM PARROTS (NOISE + PATTERNS)  # TODO: set all axes[0] back to axes[4] here
            if inpgen.use_noise:
                nr = inpgen.noiserecorder.get("events")
                evs__ = nr["senders"]
                ts__ = nr["times"]
                parrots_start_id = min(evs__)  # smallest id of a parrot neuron
                for parrot in np.unique(nr["senders"]):
                    indices = np.where(nr["senders"] == parrot)[0]
                    # if not self.plot_history:  # remove all indices from outside of noiserecorder_time_window
                    #     indices = [i for i in indices if i in noiserecorder_time_window]
                    # axes[2].plot(ts__[indices], evs__[indices]-parrots_start_id, ".", color='black')
                    axes[1].plot(ts__, evs__-parrots_start_id, ".", color='black')

            # PRESENTED PATTERNS # TODO: set all axes[0] back to axes[4] here
            time_shift = nest.biological_time - t_sim
            if inpgen is not None:
                st = inpgen.spiketrain
                for i in range(len(st)):
                    # ax3.scatter(np.add(time_shift, st[i]), [i] * len(st[i]), color=(i / (len(st)), 0.0, i / (len(st))))
                    axes[1].plot(np.add(st[i], 10), [i] * len(st[i]), ".", color='red', ms=8)
                if not self.plot_history:
                    axes[1].set_xlim(time_shift, nest.biological_time)

            axes[0].set_title("t_sim= %d, t_start= %d" % (t_sim, (nest.biological_time - t_sim)))
            #axes[0].set_ylabel("Membrane potential (mV)")
            axes[0].set_title("Network spike events")
            axes[1].set_title("Input spike events")
            axes[1].set_xlabel("time [ms]")
            #fig.suptitle(title, fontsize=20)
            #axes[1].set_ylabel("Spike ID")
            #axes[2].set_title("EPSP traces")
            # axes[2].legend()
            #axes[3].set_title("Recurrent weights")
            #axes[3].set_ylabel("w")
            # axes[3].legend()
            #axes[4].set_ylabel("Input channels")
            #axes[4].set_xlabel("time (ms)")


            fig.tight_layout()

            run_time = time.time() - start_time
            print("Plotting complete in %s" % run_time)
            if self.save_figures:
                plt.savefig("simulation_%ds.png" % int(time.time()))
            if self.show_figures:
                plt.show()

            if 1: #plot_total_spike_activity
                fig, axes = plt.subplots(1, 1)

            print("#############################")
            print("#  Number of spikes: ")
            pprint.PrettyPrinter().pprint(NUMBER_OF_SPIKES)
            print("#############################")

        self.id_list = id_list
        return id_list

    def record_variables_step(self):
        """
        Extract and store input weights from the variables stored in the postsynaptic neurons
        """
        print("Recording step")
        if self.id_list is not None:  # limit recorded nodes to the ones from id_list
            target = nest.NodeCollection(self.id_list)
            inp_conns = nest.GetConnections(synapse_model="stdp_stp__with_iaf_psc_exp_wta", target=target)
        else:
            inp_conns = nest.GetConnections(synapse_model="stdp_stp__with_iaf_psc_exp_wta")
        postsyn_weights = nest.GetStatus(self.network.get_node_collections(), 'weights')
        postsyn_epsps = nest.GetStatus(self.network.get_node_collections(), 'epsp_trace')

        t_cur = nest.biological_time

        inp_conns_src = inp_conns.source
        inp_conns_tgt = inp_conns.target
        inp_conns_tgt_min = min(inp_conns_tgt)

        for idx, tgt in enumerate(inp_conns_tgt):
            src = inp_conns_src[idx]  # global ID of source node
            rel0_idx = tgt - inp_conns_tgt_min  # array index relative to 0
            self.network.weight_recorder.append((t_cur, src, tgt, postsyn_weights[rel0_idx][src]))
            self.network.epsp_recorder.append((t_cur, src, tgt, postsyn_epsps[rel0_idx][src]))

    def test(self, inpgen, t, dt_rec=None):
        # Disable STP and STDP
        disable_stp(self.network.neuron_collection)
        disable_stdp(self.network.neuron_collection)
        # Pre-generate input
        inpgen.generate_input(t, t_origin=nest.biological_time)
        update_presyn_ids(self.network)  # IMPORTANT - always set this after input generation

        if dt_rec is None:
            self.record_variables_step()
            nest.Simulate(t)
            self.record_variables_step()
        else:
            for t_ in range(int(t / dt_rec)):
                self.record_variables_step()
                nest.Simulate(dt_rec)
                print("step %s/%s complete." % (t_, t / dt_rec))
            self.record_variables_step()

    def simulate(self, inpgen, t, dt_rec=None):
        """Pre-generates input patterns for duration of simulation and then runs the simulation"""
        # Enable STP and STDP
        if self.network.use_stp:
            enable_stp(self.network.neuron_collection)
        enable_stdp(self.network.neuron_collection)
        # Pre-generate input
        inpgen.generate_input(t, t_origin=nest.biological_time)
        print("update_presyn_ids...")
        update_presyn_ids(self.network)  # IMPORTANT - always set this after input generation

        if dt_rec is None:
            self.record_variables_step()
            nest.Simulate(t)
            self.record_variables_step()
        else:
            for t_ in range(int(t / dt_rec)):
                self.record_variables_step()
                nest.Simulate(dt_rec)
                print("Step %s/%s complete. Time passed since simulation start: %s" % (
                t_, int(t / dt_rec), nest.biological_time))
            self.record_variables_step()

    def get_order(self, p, I, t, r_fracs_events, tstart, nsteps):
        """
        :param p: time points of pattern phases (start and stop point)
            [e.g., [0, 340, 640, 957, 1257, 1671, 1971, 2376, 2676]]
        :param I: unfolded pattern ids, or -1 during noise phase - arrays of size nsteps
        :param t: same dim as p; duration of each pattern phase, twice
        :param r: rate fractions for all neurons in all WTA
        :param tstart:
        :param nsteps:
        :return:
        """
        # clip inputs to correct length
        filtered_indices = np.where((tstart <= p) & (p <= tstart + nsteps))
        p = p[filtered_indices]
        t = t[filtered_indices]
        I = I[tstart:tstart+nsteps]

        # # unflatten rate fractions
        n_timepoints = len(r_fracs_events[0]['rate_fraction'])
        # first, r will be N x T, but will be transposed afterwards
        r = np.empty((0, n_timepoints), int)
        # for i in np.arange(len(self.id_list)*tstart, len(r_fracs), len(self.id_list)):
        #     r = np.append(r, np.array([r_fracs[i:i+len(self.id_list)]]), axis=0)
        for idx, d in enumerate(r_fracs_events):
            r = np.append(r, np.array([d['rate_fraction']]), axis=0)
        # eliminate initial infinity line
        r = r.T  # need to transpose here for TxN
        r[1, :] = r[2, :]
        r[0, :] = r[2, :]

        pt = 0
        order = np.arange(self.n_rec_neurons)
        times = np.zeros(order.shape)
        if self.order_neurons:
            for pti, ptt in enumerate(p[::-1]):  # iterate over p in reverse
                pt = ptt - tstart
                pi = np.max(I[pt])  #
                pl = t[-pti-1]  # pattern length / duration
                if pi < 0:
                    continue
                #pl = int(self.tPattern[pi]/self.dt)
                if pt+pl <= nsteps:
                    break
            Tord = np.arange(pt, pt+pl).astype(int)
            tmp = np.sum(r[Tord, :].T*np.exp(np.arange(pl)/(pl/(2*np.pi))*1j), axis=1)
            tmp /= np.sum(r[Tord, :].T, axis=1)
            angles = np.angle(tmp)
            angles[angles<0] += 2*np.pi
            weighted_rates_max_time = angles/(2*np.pi/pl)
            assert(weighted_rates_max_time.shape == (self.n_rec_neurons,))
            order = np.argsort(weighted_rates_max_time)
            times = pt+weighted_rates_max_time[order]
        else:
            Tord = np.arange(pt, pt).astype(int)
        return order, times, Tord

    # def get_order(self, p, I, t, r_fracs, tstart, nsteps):
    #     """
    #     :param p: time points of pattern phases (start and stop point)
    #         [e.g., [0, 340, 640, 957, 1257, 1671, 1971, 2376, 2676]]
    #     :param I: unfolded pattern ids, or -1 during noise phase - arrays of size nsteps
    #     :param t: same dim as p; duration of each pattern phase, twice
    #     :param r: rate fractions for all neurons in all WTA
    #     :param tstart:
    #     :param nsteps:
    #     :return:
    #     """
    #     # clip inputs to correct length
    #     filtered_indices = np.where((tstart <= p) & (p <= tstart + nsteps))
    #     p = p[filtered_indices]
    #     t = t[filtered_indices]
    #     I = I[tstart:tstart+nsteps]
    #
    #     # unflatten rate fractions
    #     rate_fractions = np.empty((0, len(self.id_list)), int)
    #     for i in np.arange(len(self.id_list)*tstart, len(r_fracs), len(self.id_list)):
    #         rate_fractions = np.append(rate_fractions, np.array([r_fracs[i:i+len(self.id_list)]]), axis=0)
    #     # eliminate initial infinity line
    #     rate_fractions[1, :] = rate_fractions[2, :]
    #     rate_fractions[0, :] = rate_fractions[2, :]
    #
    #     pattern_time = 0
    #     order = np.arange(self.n_rec_neurons)  # array to be ordered, should correspond to global_ids?
    #     times = np.zeros(order.shape)
    #     if self.order_neurons:
    #         for pti, ptt in enumerate(p[::-1]):  # iterate over p in reverse
    #             pattern_time = ptt - tstart
    #             pattern_idx = np.max(I[pattern_time])  #
    #             pattern_len = t[-pti-1]  # pattern length / duration
    #             if pattern_idx < 0:
    #                 continue
    #             #pl = int(self.tPattern[pi]/self.dt)
    #             if pattern_time+pattern_len <= nsteps:
    #                 break
    #         Tord = np.arange(pattern_time, pattern_time + pattern_len).astype(int)
    #         tmp = np.sum(rate_fractions[Tord, :].T * np.exp(np.arange(pattern_len)/(pattern_len/(2*np.pi))*1j), axis=1)
    #         tmp /= np.sum(rate_fractions[Tord, :].T, axis=1)
    #         angles = np.angle(tmp)
    #         angles[angles<0] += 2*np.pi
    #         weighted_rates_max_time = angles/(2*np.pi/pattern_len)
    #         assert(weighted_rates_max_time.shape == (self.n_rec_neurons,))
    #         order = np.argsort(weighted_rates_max_time)
    #         times = pattern_time+weighted_rates_max_time[order]
    #     else:
    #         Tord = np.arange(pattern_time, pattern_time).astype(int)
    #     return order, times, Tord
