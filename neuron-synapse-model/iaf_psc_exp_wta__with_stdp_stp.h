/**
 *  iaf_psc_exp_wta__with_stdp_stp.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Generated from NESTML at time: 2022-09-17 19:34:29.747854
**/
#ifndef IAF_PSC_EXP_WTA__WITH_STDP_STP
#define IAF_PSC_EXP_WTA__WITH_STDP_STP

#include "config.h"

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

// Includes from sli:
#include "dictdatum.h"

namespace nest {
    namespace iaf_psc_exp_wta__with_stdp_stp_names {
        const Name _r("r");
        const Name _V_m("V_m");
        const Name _time_cnt("time_cnt");
        //const Name _rise_time_kernel__X__all_spikes("rise_time_kernel__X__all_spikes");
        //const Name _decay_time_kernel__X__all_spikes("decay_time_kernel__X__all_spikes");
        const Name _tau_m("tau_m");
        const Name _tau_syn("tau_syn");
        const Name _R_max("R_max");
        const Name _eta("eta");
        const Name _use_variance_tracking("use_variance_tracking");
        const Name _use_stdp("use_stdp");

        // to record from
        const Name _epsp_trace("epsp_trace");
        const Name _weights("weights");

        const Name _fixed_spiketimes("fixed_spiketimes");  // for toy model, set manually
        const Name _presyn_ids("presyn_ids");  // for toy model, set manually

        const Name _normalization_sum("normalization_sum");
        const Name _rate_fraction("rate_fraction");
        const Name _rate("rate");
        const Name _max_neuron_gid("max_neuron_gid");
    }
}


#include "nest_time.h"

// entry in the spiking history
class histentry__iaf_psc_exp_wta__with_stdp_stp {
public:
    histentry__iaf_psc_exp_wta__with_stdp_stp(double t,
                                              size_t access_counter)
            : t_(t), access_counter_(access_counter) {
    }

    double t_;              //!< point in time when spike occurred (in ms)
    size_t access_counter_; //!< access counter to enable removal of the entry, once all neurons read it
};


/* BeginDocumentation
  Name: iaf_psc_exp_wta__with_stdp_stp.

  Description:

    

  Parameters:
  The following parameters can be set in the status dictionary.
tau_m [ms]  Capacitance of the membrane
 Membrane time constant
tau_syn [ms]  Time constant of excitatory synaptic current
R_max [Hz]  Maximum rate within current WTA circuit


  Dynamic state variables:
r [integer]  counts number of tick during the refractory period
V_m [mV]  Membrane potential
time_cnt [integer]  TODO temporary


  Sends: nest::SpikeEvent

  Receives: Spike,  DataLoggingRequest
*/
class iaf_psc_exp_wta__with_stdp_stp : public nest::ArchivingNode {
public:
    /**
     * The constructor is only used to create the model prototype in the model manager.
    **/
    iaf_psc_exp_wta__with_stdp_stp();

    /**
     * The copy constructor is used to create model copies and instances of the model.
     * @node The copy constructor needs to initialize the parameters and the state.
     *       Initialization of buffers and interal variables is deferred to
     *       @c init_buffers_() and @c pre_run_hook() (or calibrate() in NEST 3.3 and older).
    **/
    iaf_psc_exp_wta__with_stdp_stp(const iaf_psc_exp_wta__with_stdp_stp &);

    /**
     * Destructor.
    **/
    ~iaf_psc_exp_wta__with_stdp_stp();

    // -------------------------------------------------------------------------
    //   Import sets of overloaded virtual functions.
    //   See: Technical Issues / Virtual Functions: Overriding, Overloading,
    //        and Hiding
    // -------------------------------------------------------------------------

    using nest::Node::handles_test_event;
    using nest::Node::handle;
    using nest::Node::sends_secondary_event;

    /**
     * Used to validate that we can send nest::SpikeEvent to desired target:port.
    **/
    nest::port send_test_event(nest::Node &target, nest::rport receptor_type, nest::synindex, bool);

    // -------------------------------------------------------------------------
    //   Functions handling incoming events.
    //   We tell nest that we can handle incoming events of various types by
    //   defining handle() for the given event.
    // -------------------------------------------------------------------------


    void handle(nest::SpikeEvent &);        //! accept spikes
    void handle(nest::DataLoggingRequest &);//! allow recording with multimeter
    void handle(nest::InstantaneousRateConnectionEvent &);//! accept continuous rate events
    nest::port handles_test_event(nest::SpikeEvent &, nest::port);

    nest::port handles_test_event(nest::DataLoggingRequest &, nest::port);

    nest::port handles_test_event(nest::InstantaneousRateConnectionEvent &, nest::port);

    void sends_secondary_event(nest::InstantaneousRateConnectionEvent &) {

    }

    // -------------------------------------------------------------------------
    //   Functions for getting/setting parameters and state values.
    // -------------------------------------------------------------------------

    void get_status(DictionaryDatum &) const;

    void set_status(const DictionaryDatum &);
    // support for spike archiving

    /**
     * \fn void get_history(long t1, long t2,
     * std::deque<Archiver::histentry__>::iterator* start,
     * std::deque<Archiver::histentry__>::iterator* finish)
     * return the spike times (in steps) of spikes which occurred in the range
     * (t1,t2].
     * XXX: two underscores to differentiate it from nest::Node::get_history()
     */
    void get_history__(double t1,
                       double t2,
                       std::deque<histentry__iaf_psc_exp_wta__with_stdp_stp>::iterator *start,
                       std::deque<histentry__iaf_psc_exp_wta__with_stdp_stp>::iterator *finish);

    /**
     * Register a new incoming STDP connection.
     *
     * t_first_read: The newly registered synapse will read the history entries
     * with t > t_first_read.
     */
    void register_stdp_connection(double t_first_read, double delay);

    // -------------------------------------------------------------------------
    //   Getters/setters for state block
    // -------------------------------------------------------------------------

    inline long get_r() const {
        return S_.r;
    }

    inline void set_r(const long __v) {
        S_.r = __v;
    }

    inline double get_rate_fraction() const {
        return V_.rate_fraction;
    }

    inline void set_rate_fraction(const double __v) {
        V_.rate_fraction = __v;
    }

    inline double get_rate() const {
        return V_.rate;
    }

    inline void set_rate(const double __v) {
        V_.rate = __v;
    }

    inline double get_eta() const {
        return V_.eta;
    }

    inline void set_eta(const double __v) {
        V_.eta = __v;
    }

    inline double get_V_m() const {
        return S_.V_m;
    }

    inline void set_V_m(const double __v) {
        S_.V_m = __v;
    }

    inline long get_time_cnt() const {
        return S_.time_cnt;
    }

    inline void set_time_cnt(const long __v) {
        S_.time_cnt = __v;
    }

    inline std::vector<double> get_presyn_ids() const {
        std::vector<double> presyn_ids;

        for (auto &it: V_.activeSources) {
            presyn_ids.push_back(it);
        }
        return presyn_ids;
    }

    //  inline void set_presyn_ids(const std::vector<double> __v)
    //  {
    //    V_.presyn_ids = __v;
    //  }

    inline std::vector<double> get_fixed_spiketimes() const {
        return V_.fixed_spiketimes;
    }

    inline void set_fixed_spiketimes(const std::vector<double> __v) {
        V_.fixed_spiketimes = __v;
    }

    inline std::vector<double> get_epsp_traces() const {
        return V_.yt_epsp_traces;
    }

    inline std::vector<double> get_weights() const {
        return V_.localWeights_Wk;
    }

    //inline double get_rise_time_kernel__X__all_spikes() const {
    //    return S_.rise_time_kernel__X__all_spikes;
    //}

    //inline void set_rise_time_kernel__X__all_spikes(const double __v) {
    //    S_.rise_time_kernel__X__all_spikes = __v;
    //}

    //inline double get_decay_time_kernel__X__all_spikes() const {
    //    return S_.decay_time_kernel__X__all_spikes;
    //}

    //inline void set_decay_time_kernel__X__all_spikes(const double __v) {
    //    S_.decay_time_kernel__X__all_spikes = __v;
    //}


    // -------------------------------------------------------------------------
    //   Getters/setters for parameters
    // -------------------------------------------------------------------------

    inline double get_max_neuron_gid() const {
        return P_.max_neuron_gid;
    }

    inline void set_max_neuron_gid(const double __v) {
        P_.max_neuron_gid = __v;
    }

    inline double get_tau_m() const {
        return P_.tau_m;
    }

    inline void set_tau_m(const double __v) {
        P_.tau_m = __v;
    }

    inline double get_use_variance_tracking() const {
        return P_.use_variance_tracking;
    }

    inline void set_use_variance_tracking(const double __v) {
        P_.use_variance_tracking = __v;
    }

    inline double get_use_stdp() const {
        return P_.use_stdp;
    }

    inline void set_use_stdp(const double __v) {
        P_.use_stdp = __v;
    }

    inline double get_tau_syn() const {
        return P_.tau_syn;
    }

    inline void set_tau_syn(const double __v) {
        P_.tau_syn = __v;
    }

    inline double get_R_max() const {
        return P_.R_max;
    }

    inline void set_R_max(const double __v) {
        P_.R_max = __v;
    }


    // -------------------------------------------------------------------------
    //   Getters/setters for internals
    // -------------------------------------------------------------------------

    inline double get___h() const {
        return V_.__h;
    }

    inline void set___h(const double __v) {
        V_.__h = __v;
    }

    inline double get___P__V_m__V_m() const {
        return V_.__P__V_m__V_m;
    }

    inline void set___P__V_m__V_m(const double __v) {
        V_.__P__V_m__V_m = __v;
    }

    //inline double get___P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes() const {
    //    return V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes;
    //}

    //inline void set___P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes(const double __v) {
    //    V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = __v;
    //}

    //inline double get___P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes() const {
    //    return V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes;
    //}

    //inline void set___P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes(const double __v) {
    //    V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = __v;
    //}


    inline double get_normalization_sum() const {
        /*
        Compute normalization sum on the fly
        */
        double norm_sum = 0;
        for (auto &it: V_.presyn_u_t) {
//            norm_sum += std::exp(it - V_.normalization_max_prev);  // add u(t) of presyn nodes
            norm_sum += std::exp(it);  // add u(t) of presyn nodes
        }
//        norm_sum += std::exp(get_V_m() - V_.normalization_max_prev);  // add u(t) of this neuron
        norm_sum += std::exp(get_V_m());  // add u(t) of this neuron
        return norm_sum;
    }

    /* getters/setters for variables transferred from synapse */

protected:
    // support for spike archiving

    /**
     * record spike history
     */
    void set_spiketime(nest::Time const &t_sp, double offset = 0.0);

    /**
     * return most recent spike time in ms
     */
    inline double get_spiketime_ms() const;

    /**
     * clear spike history
     */
    void clear_history();

private:
    void recompute_internal_variables(bool exclude_timestep = false);
    // support for spike archiving

    // number of incoming connections from stdp connectors.
    // needed to determine, if every incoming connection has
    // read the spikehistory for a given point in time
    size_t n_incoming_;

    double max_delay_;

    double last_spike_;

    // spiking history needed by stdp synapses
    std::deque <histentry__iaf_psc_exp_wta__with_stdp_stp> history_;

    // cache for initial values

private:


    /**
     * Reset internal buffers of neuron.
    **/
    void init_buffers_();

    /**
     * Initialize auxiliary quantities, leave parameters and state untouched.
    **/
    void pre_run_hook();

    /**
     * Take neuron through given time interval
    **/
    void update(nest::Time const &, const long, const long);

    void evolve_epsps(nest::Time const &, const long);

    void evolve_weights(nest::Time const &, const long);

    // The next two classes need to be friends to access the State_ class/member
    friend class nest::RecordablesMap<iaf_psc_exp_wta__with_stdp_stp>;

    friend class nest::UniversalDataLogger<iaf_psc_exp_wta__with_stdp_stp>;

    struct TraceTracker_ {
        long deliveryTime;
        double w;
        unsigned long id_;
        nest::rport port;
        // this allows a one simulation step difference in spike delivery, in order to allow correct processing;
        // spike_generator and poisson_generators call the handle() function with a 1 step difference, and we must
        // compensate for this by waiting for 1 extra step for the spike arrival during the update().
        bool one_step_mercy;
        bool is_exc;

    };

    /**
     * Free parameters of the neuron.
     *
     *
     *
     * These are the parameters that can be set by the user through @c `node.set()`.
     * They are initialized from the model prototype when the node is created.
     * Parameters do not change during calls to @c update() and are not reset by
     * @c ResetNetwork.
     *
     * @note Parameters_ need neither copy constructor nor @c operator=(), since
     *       all its members are copied properly by the default copy constructor
     *       and assignment operator. Important:
     *       - If Parameters_ contained @c Time members, you need to define the
     *         assignment operator to recalibrate all members of type @c Time . You
     *         may also want to define the assignment operator.
     *       - If Parameters_ contained members that cannot copy themselves, such
     *         as C-style arrays, you need to define the copy constructor and
     *         assignment operator to copy those members.
    **/
    struct Parameters_ {
        //!  Membrane time constant
        double tau_m;
        //!  Time constant of excitatory synaptic current
        double tau_syn;
        //!  Maximum rate within current WTA circuit
        double R_max;
        double use_variance_tracking;
        double use_stdp;
        double max_neuron_gid;
        /**
         * Initialize parameters to their default values.
        **/
        Parameters_();
    };

    /**
     * Dynamic state of the neuron.
     *
     *
     *
     * These are the state variables that are advanced in time by calls to
     * @c update(). In many models, some or all of them can be set by the user
     * through @c `node.set()`. The state variables are initialized from the model
     * prototype when the node is created. State variables are reset by @c ResetNetwork.
     *
     * @note State_ need neither copy constructor nor @c operator=(), since
     *       all its members are copied properly by the default copy constructor
     *       and assignment operator. Important:
     *       - If State_ contained @c Time members, you need to define the
     *         assignment operator to recalibrate all members of type @c Time . You
     *         may also want to define the assignment operator.
     *       - If State_ contained members that cannot copy themselves, such
     *         as C-style arrays, you need to define the copy constructor and
     *         assignment operator to copy those members.
    **/
    struct State_ {
        //!  counts number of tick during the refractory period
        long r;
        //!  Membrane potential
        double V_m;
        //!  TODO temporary
        long time_cnt;
        //double rise_time_kernel__X__all_spikes;
        //double decay_time_kernel__X__all_spikes;

        State_();
    };

    struct DelayedVariables_ {
    };

    /**
     * Internal variables of the neuron.
     *
     *
     *
     * These variables must be initialized by @c pre_run_hook (or calibrate in NEST 3.3 and older), which is called before
     * the first call to @c update() upon each call to @c Simulate.
     * @node Variables_ needs neither constructor, copy constructor or assignment operator,
     *       since it is initialized by @c pre_run_hook() (or calibrate() in NEST 3.3 and older). If Variables_ has members that
     *       cannot destroy themselves, Variables_ will need a destructor.
    **/
    struct Variables_ {
        double __h;
        double __P__V_m__V_m;
        //double __P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes;
        //double __P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes;

        std::vector<double> yt_epsp_traces = std::vector<double>(
                1000);  // synaptic activation traces, for each incoming synapse
        std::vector<double> yt_epsp_decay = std::vector<double>(
                1000);  // synaptic activation traces, for each incoming synapse == EPSP1
        std::vector<double> yt_epsp_rise = std::vector<double>(
                1000);  // synaptic activation traces, for each incoming synapse == EPSP2
        std::vector<double> preSynWeights = std::vector<double>(
                1000); // weights related to STP and only scaling the y(t)
        std::vector<double> localWeights_Wk = std::vector<double>(
                1000); // weights corresponding to w_k and only updated upon postsyn spikes
        std::vector<double> Q = std::vector<double>(1000); //  adaptive learning rate (expected value of weight^2)
        std::vector<double> S = std::vector<double>(1000); //  adaptive learning rate (expected value of weight)
        std::vector<double> fixed_spiketimes;  // for toy model
        std::vector<double> presyn_u_t;  // vector of u_i(t) from presynaptic nodes

        double rate_fraction;
        double rate;
        double eta;
        //double normalization_sum;
        double normalization_max;
        double normalization_max_prev;

        std::list <TraceTracker_> spikeEvents;  // TODO add comm
        std::set<unsigned long> activeSources;  // set with ids of connected presyn neurons
    };

    /**
     * Buffers of the neuron.
     * Usually buffers for incoming spikes and data logged for analog recorders.
     * Buffers must be initialized by @c init_buffers_(), which is called before
     * @c pre_run_hook() (or calibrate() in NEST 3.3 and older) on the first call to @c Simulate after the start of NEST,
     * ResetKernel or ResetNetwork.
     * @node Buffers_ needs neither constructor, copy constructor or assignment operator,
     *       since it is initialized by @c init_nodes_(). If Buffers_ has members that
     *       cannot destroy themselves, Buffers_ will need a destructor.
    **/
    struct Buffers_ {
        Buffers_(iaf_psc_exp_wta__with_stdp_stp &);

        Buffers_(const Buffers_ &, iaf_psc_exp_wta__with_stdp_stp &);

        /**
         * Logger for all analog data
        **/
        nest::UniversalDataLogger<iaf_psc_exp_wta__with_stdp_stp> logger_;

        inline nest::RingBuffer &get_all_spikes() { return all_spikes; }

        //!< Buffer for input (type: pA)
        nest::RingBuffer all_spikes;
        double all_spikes_grid_sum_;

    };

    // -------------------------------------------------------------------------
    //   Getters/setters for inline expressions
    // -------------------------------------------------------------------------

    //inline double get_epsp_decay() const {
    //    return get_decay_time_kernel__X__all_spikes();
    //}

    //inline double get_epsp_rise() const {
    //    return get_rise_time_kernel__X__all_spikes();
    //}

    //inline double get_y() const {
    //    return ((get_decay_time_kernel__X__all_spikes()) - (get_rise_time_kernel__X__all_spikes())) / 1.0;
    //}


    // -------------------------------------------------------------------------
    //   Getters/setters for input buffers
    // -------------------------------------------------------------------------

    inline nest::RingBuffer &get_all_spikes() { return B_.get_all_spikes(); };

    // -------------------------------------------------------------------------
    //   Member variables of neuron model.
    //   Each model neuron should have precisely the following four data members,
    //   which are one instance each of the parameters, state, buffers and variables
    //   structures. Experience indicates that the state and variables member should
    //   be next to each other to achieve good efficiency (caching).
    //   Note: Devices require one additional data member, an instance of the
    //   ``Device`` child class they belong to.
    // -------------------------------------------------------------------------


    Parameters_ P_;        //!< Free parameters.
    State_ S_;        //!< Dynamic state.
    DelayedVariables_ DV_;       //!< Delayed state variables.
    Variables_ V_;        //!< Internal Variables
    Buffers_ B_;        //!< Buffers.

    //! Mapping of recordables names to access functions
    static nest::RecordablesMap<iaf_psc_exp_wta__with_stdp_stp> recordablesMap_;

}; /* neuron iaf_psc_exp_wta__with_stdp_stp */

inline nest::port
iaf_psc_exp_wta__with_stdp_stp::send_test_event(nest::Node &target, nest::rport receptor_type, nest::synindex, bool) {
    // You should usually not change the code in this function.
    // It confirms that the target of connection @c c accepts @c nest::SpikeEvent on
    // the given @c receptor_type.
    nest::SpikeEvent e;
    e.set_sender(*this);
    return target.handles_test_event(e, receptor_type);
}

inline nest::port iaf_psc_exp_wta__with_stdp_stp::handles_test_event(nest::SpikeEvent &, nest::port receptor_type) {
    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c SpikeEvent on port 0. You need to extend the function
    // if you want to differentiate between input ports.
    if (receptor_type != 0) {
        throw nest::UnknownReceptorType(receptor_type, get_name());
    }
    return 0;
}

inline nest::port
iaf_psc_exp_wta__with_stdp_stp::handles_test_event(nest::DataLoggingRequest &dlr, nest::port receptor_type) {
    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c DataLoggingRequest on port 0.
    // The function also tells the built-in UniversalDataLogger that this node
    // is recorded from and that it thus needs to collect data during simulation.
    if (receptor_type != 0) {
        throw nest::UnknownReceptorType(receptor_type, get_name());
    }

    return B_.logger_.connect_logging_device(dlr, recordablesMap_);
}


inline nest::port iaf_psc_exp_wta__with_stdp_stp::handles_test_event(nest::InstantaneousRateConnectionEvent &dlr,
                                                                     nest::port receptor_type) {
    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c DataLoggingRequest on port 0.
    // The function also tells the built-in UniversalDataLogger that this node
    // is recorded from and that it thus needs to collect data during simulation.
    if (receptor_type != 0) {
        throw nest::UnknownReceptorType(receptor_type, get_name());
    }

    return 0;
}


inline void iaf_psc_exp_wta__with_stdp_stp::get_status(DictionaryDatum &__d) const {
    // parameters
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_tau_m, get_tau_m());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_max_neuron_gid, get_max_neuron_gid());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_tau_syn, get_tau_syn());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_R_max, get_R_max());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_use_variance_tracking, get_use_variance_tracking());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_use_stdp, get_use_stdp());

    // initial values for state variables in ODE or kernel
    def<long>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_r, get_r());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_V_m, get_V_m());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_eta, get_eta());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_rate_fraction, get_rate_fraction());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_rate, get_rate());
    def<long>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_time_cnt, get_time_cnt());
    //def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_rise_time_kernel__X__all_spikes, get_rise_time_kernel__X__all_spikes());
    //def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_decay_time_kernel__X__all_spikes, get_decay_time_kernel__X__all_spikes());

    def < std::vector < double >> (__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_epsp_trace, get_epsp_traces());
    def < std::vector < double >> (__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_weights, get_weights());
    def < std::vector < double >> (__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_fixed_spiketimes, get_fixed_spiketimes());
    def < std::vector < double >> (__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_presyn_ids, get_presyn_ids());
    def<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_normalization_sum, get_normalization_sum());

    ArchivingNode::get_status(__d);

    (*__d)[nest::names::recordables] = recordablesMap_.get_list();
}

inline void iaf_psc_exp_wta__with_stdp_stp::set_status(const DictionaryDatum &__d) {
    // parameters
    double tmp_tau_m = get_tau_m();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_tau_m, tmp_tau_m);
    
    double tmp_max_neuron_gid = get_max_neuron_gid();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_max_neuron_gid, tmp_max_neuron_gid);
    
    double tmp_tau_syn = get_tau_syn();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_tau_syn, tmp_tau_syn);
    double tmp_R_max = get_R_max();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_R_max, tmp_R_max);
    double tmp_use_variance_tracking = get_use_variance_tracking();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_use_variance_tracking, tmp_use_variance_tracking);
    double tmp_use_stdp = get_use_stdp();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_use_stdp, tmp_use_stdp);

    // initial values for state variables in ODE or kernel
    long tmp_r = get_r();
    updateValue<long>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_r, tmp_r);
    double tmp_V_m = get_V_m();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_V_m, tmp_V_m);
    double tmp_eta = get_eta();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_eta, tmp_eta);
    double tmp_rate_fraction = get_rate_fraction();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_rate_fraction, tmp_rate_fraction);
    double tmp_rate = get_rate();
    updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_rate, tmp_rate);


    long tmp_time_cnt = get_time_cnt();
    updateValue<long>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_time_cnt, tmp_time_cnt);
    //double tmp_rise_time_kernel__X__all_spikes = get_rise_time_kernel__X__all_spikes();
    //updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_rise_time_kernel__X__all_spikes, tmp_rise_time_kernel__X__all_spikes);
    //double tmp_decay_time_kernel__X__all_spikes = get_decay_time_kernel__X__all_spikes();
    //updateValue<double>(__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_decay_time_kernel__X__all_spikes, tmp_decay_time_kernel__X__all_spikes);

    std::vector<double> tmp_fixed_spiketimes = get_fixed_spiketimes();
    updateValue < std::vector < double >> (__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_fixed_spiketimes, tmp_fixed_spiketimes);

    std::vector<double> tmp_presyn_ids;
    updateValue < std::vector < double >> (__d, nest::iaf_psc_exp_wta__with_stdp_stp_names::_presyn_ids, tmp_presyn_ids);

    if (tmp_presyn_ids.size()) {
#ifdef DEBUG
        std::cout << "WARNING - resetting active sources!" << std::endl << std::flush;
#endif
        for (auto &it: tmp_presyn_ids) {
            V_.activeSources.insert((unsigned long) (it));
        }

    }


    // We now know that (ptmp, stmp) are consistent. We do not
    // write them back to (P_, S_) before we are also sure that
    // the properties to be set in the parent class are internally
    // consistent.
    ArchivingNode::set_status(__d);

    // if we get here, temporaries contain consistent set of properties
    set_tau_m(tmp_tau_m);
    set_max_neuron_gid(tmp_max_neuron_gid);
    set_tau_syn(tmp_tau_syn);
    set_R_max(tmp_R_max);
    set_use_variance_tracking(tmp_use_variance_tracking);
    set_use_stdp(tmp_use_stdp);
    set_r(tmp_r);
    set_V_m(tmp_V_m);
    set_eta(tmp_eta);
    set_rate_fraction(tmp_rate_fraction);
    set_time_cnt(tmp_time_cnt);
    //set_rise_time_kernel__X__all_spikes(tmp_rise_time_kernel__X__all_spikes);
    //set_decay_time_kernel__X__all_spikes(tmp_decay_time_kernel__X__all_spikes);

    set_fixed_spiketimes(tmp_fixed_spiketimes);


    // recompute internal variables in case they are dependent on parameters or state that might have been updated in this call to set_status()
    recompute_internal_variables();
};

#endif /* #ifndef IAF_PSC_EXP_WTA__WITH_STDP_STP */
