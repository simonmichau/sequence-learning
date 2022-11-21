// #define DEBUG 1
/*
 *  iaf_psc_exp_wta__with_stdp_stp.cpp
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

// C++ includes:
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

#include "iaf_psc_exp_wta__with_stdp_stp.h"
//#define DEBUG

// ---------------------------------------------------------------------------
//   Recordables map
// ---------------------------------------------------------------------------
nest::RecordablesMap<iaf_psc_exp_wta__with_stdp_stp> iaf_psc_exp_wta__with_stdp_stp::recordablesMap_;
namespace nest {

    // Override the create() method with one call to RecordablesMap::insert_()
    // for each quantity to be recorded.
    template<>
    void RecordablesMap<iaf_psc_exp_wta__with_stdp_stp>::create() {
        // add state variables to recordables map
        insert_(iaf_psc_exp_wta__with_stdp_stp_names::_V_m, &iaf_psc_exp_wta__with_stdp_stp::get_V_m);
        //insert_(iaf_psc_exp_wta__with_stdp_stp_names::_rise_time_kernel__X__all_spikes,
        //        &iaf_psc_exp_wta__with_stdp_stp::get_rise_time_kernel__X__all_spikes);
        //insert_(iaf_psc_exp_wta__with_stdp_stp_names::_decay_time_kernel__X__all_spikes,
        //        &iaf_psc_exp_wta__with_stdp_stp::get_decay_time_kernel__X__all_spikes);
        insert_(iaf_psc_exp_wta__with_stdp_stp_names::_normalization_sum,
                &iaf_psc_exp_wta__with_stdp_stp::get_normalization_sum);
        insert_(iaf_psc_exp_wta__with_stdp_stp_names::_rate_fraction,
                &iaf_psc_exp_wta__with_stdp_stp::get_rate_fraction);
        // Cant use r here because it has wrong type
        insert_(iaf_psc_exp_wta__with_stdp_stp_names::_rate,
                &iaf_psc_exp_wta__with_stdp_stp::get_rate);

        // Add vector variables
    }
}

// ---------------------------------------------------------------------------
//   Default constructors defining default parameters and state
//   Note: the implementation is empty. The initialization is of variables
//   is a part of iaf_psc_exp_wta__with_stdp_stp's constructor.
// ---------------------------------------------------------------------------

iaf_psc_exp_wta__with_stdp_stp::Parameters_::Parameters_() {
}

iaf_psc_exp_wta__with_stdp_stp::State_::State_() {
}

// ---------------------------------------------------------------------------
//   Parameter and state extractions and manipulation functions
// ---------------------------------------------------------------------------

iaf_psc_exp_wta__with_stdp_stp::Buffers_::Buffers_(iaf_psc_exp_wta__with_stdp_stp &n) :
        logger_(n) {
    // Initialization of the remaining members is deferred to init_buffers_().
}

iaf_psc_exp_wta__with_stdp_stp::Buffers_::Buffers_(const Buffers_ &, iaf_psc_exp_wta__with_stdp_stp &n) :
        logger_(n) {
    // Initialization of the remaining members is deferred to init_buffers_().
}

// ---------------------------------------------------------------------------
//   Default constructor for node
// ---------------------------------------------------------------------------

iaf_psc_exp_wta__with_stdp_stp::iaf_psc_exp_wta__with_stdp_stp() : ArchivingNode(), P_(), S_(), B_(*this) {
    const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function
    pre_run_hook();
    // initial values for parameters
    P_.max_neuron_gid = 1e3; // as ms
    P_.tau_m = 20; // as ms
    P_.tau_syn = 2; // as ms
    P_.R_max = 100; // as Hz
    // initial values for state variables
    S_.r = 0; // as integer
    S_.V_m = 0; // as mV
    S_.time_cnt = 0; // as integer
    //S_.rise_time_kernel__X__all_spikes = 0; // as real
    //S_.decay_time_kernel__X__all_spikes = 0; // as real

    V_.rate_fraction = 0.0;
    V_.rate = 0.0;
    V_.eta = 0.05;
    V_.normalization_max = -1e12;
    // Initialize local weights with 0
    std::fill(V_.localWeights_Wk.begin(), V_.localWeights_Wk.end(), 0);
    // Initialize local weights random (log(x))
    //for (auto& it : V_.localWeights_Wk)
    //{
    //    std::random_device rd;
    //    std::mt19937 e2(rd());
    //    std::uniform_real_distribution<> dist(0, 1);
    //    double tmp  = dist(e2);
    //    it = std::log(dist(e2));
        //std::cout << "Local Weight: " << it << std::endl;
    //}
    std::fill(V_.Q.begin(), V_.Q.end(), 1);
    std::fill(V_.S.begin(), V_.S.end(), 0);
    recordablesMap_.create();
    // state variables for archiving state for paired synapse
    n_incoming_ = 0;
    max_delay_ = 0;
    last_spike_ = -1.;

    // cache initial values
}

// ---------------------------------------------------------------------------
//   Copy constructor for node
// ---------------------------------------------------------------------------

iaf_psc_exp_wta__with_stdp_stp::iaf_psc_exp_wta__with_stdp_stp(const iaf_psc_exp_wta__with_stdp_stp &__n) :
        ArchivingNode(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this) {

    // copy parameter struct P_
    P_.tau_m = __n.P_.tau_m;
    P_.max_neuron_gid = __n.P_.max_neuron_gid;
    P_.tau_syn = __n.P_.tau_syn;
    P_.R_max = __n.P_.R_max;

    // copy state struct S_
    S_.r = __n.S_.r;
    S_.V_m = __n.S_.V_m;
    S_.time_cnt = __n.S_.time_cnt;
    //S_.rise_time_kernel__X__all_spikes = __n.S_.rise_time_kernel__X__all_spikes;
    //S_.decay_time_kernel__X__all_spikes = __n.S_.decay_time_kernel__X__all_spikes;


    // copy internals V_
    V_.normalization_max = __n.V_.normalization_max;
    V_.__h = __n.V_.__h;
    V_.__P__V_m__V_m = __n.V_.__P__V_m__V_m;
    //V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = __n.V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes;
    //V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = __n.V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes;
    V_.Q = __n.V_.Q;
    V_.S = __n.V_.S;
    V_.eta = __n.V_.eta;
    V_.rate_fraction = __n.V_.rate_fraction;
    V_.rate = __n.V_.rate;
    V_.localWeights_Wk = __n.V_.localWeights_Wk;
    V_.preSynWeights = __n.V_.preSynWeights;
    n_incoming_ = __n.n_incoming_;
    max_delay_ = __n.max_delay_;
    last_spike_ = __n.last_spike_;

    // cache initial values
}

// ---------------------------------------------------------------------------
//   Destructor for node
// ---------------------------------------------------------------------------

iaf_psc_exp_wta__with_stdp_stp::~iaf_psc_exp_wta__with_stdp_stp() {
}

// ---------------------------------------------------------------------------
//   Node initialization functions
// ---------------------------------------------------------------------------

void iaf_psc_exp_wta__with_stdp_stp::init_buffers_() {
    get_all_spikes().clear(); //includes resize
    B_.logger_.reset(); // includes resize
}

void iaf_psc_exp_wta__with_stdp_stp::recompute_internal_variables(bool exclude_timestep) {
    const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

    if (exclude_timestep) {
        V_.__P__V_m__V_m = 1; // as real
        //V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = std::exp(
        //        (-(V_.__h)) / P_.tau_syn); // as real
        //V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = std::exp(
        //        (-(V_.__h)) / P_.tau_m); // as real
    } else {
        // internals V_
        V_.__h = __resolution; // as ms
        V_.__P__V_m__V_m = 1; // as real
        //V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = std::exp(
        //        (-(V_.__h)) / P_.tau_syn); // as real
        //V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = std::exp(
        //        (-(V_.__h)) / P_.tau_m); // as real
    }
}

void iaf_psc_exp_wta__with_stdp_stp::pre_run_hook() {
    B_.logger_.init();

    recompute_internal_variables();

    // buffers B_
}

// ---------------------------------------------------------------------------
//   Update and spike handling functions
// ---------------------------------------------------------------------------
void iaf_psc_exp_wta__with_stdp_stp::evolve_weights(nest::Time const &origin, const long lag) // eq (5)
{
    if (P_.use_stdp){
        /**
        IMPORTANT
        **/
        //double eta = 0.05 / (origin.get_steps() + 1);  // +1 is needed for consistency with Klampfl
        double eta_;

        for (auto &it: V_.activeSources) {
            double w = V_.localWeights_Wk[it];  // this should be negative - same sign as in Klampfl
            double P = std::exp(w);

            if (P_.use_variance_tracking) {
                eta_ = V_.eta * (V_.Q[it] - V_.S[it] * V_.S[it]) / (std::exp(-V_.S[it]) + 1);
            } else {
                eta_ = V_.eta / (origin.get_steps() + 1);  // +1 is needed for consistency with Klampfl
            }

            double truncP = std::max(P, eta_);
            double dw = (V_.yt_epsp_traces[it] - P) / truncP;

            V_.localWeights_Wk[it] += eta_ * dw;

            if (P_.use_variance_tracking) {
                double w = V_.localWeights_Wk[it];
                V_.S[it] += eta_ * (w - V_.S[it]);
                V_.Q[it] += eta_ * (w * w - V_.Q[it]);
            }

#ifdef DEBUG
            std::cout << "[[[ " << get_node_id() << " ]]] " << "Evolved w_ik for " << it << " -> " << get_node_id() << " = " << V_.localWeights_Wk[it]
                      << " from y(t) = " << V_.yt_epsp_traces[it]
                      << std::endl << std::flush;
#endif
        }
    }

}

void iaf_psc_exp_wta__with_stdp_stp::evolve_epsps(nest::Time const &origin, const long lag) {
    /**
    IMPORTANT
    Computes the current voltage u(t) as per equation (1). Following this, we evolve the unweighted,
    but STP scaled EPSPs to get y_i(t) for each presynaptic neuron - this is used in the next step then.
    **/
    std::ostringstream msg;

    double u_t = 0;
    const double resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

    // TODO temporarily switched order - result: seems to match legacy behavior
    // TODO This is done for the recurrent weights
        // evolve synaptic activations s_ij, i.e., process any spikes and add corresponding scaled delta impulses
    for (auto it = V_.spikeEvents.begin(); it != V_.spikeEvents.end();) {
#ifdef DEBUG
        std::cout << "[[[ " << get_node_id() << " ]]] "  << "[update] spike from source " << it->id_ << ", scheduled @ " << it->deliveryTime
            << "; origin: " << origin.get_steps() << "; lag: " << lag << "\n" << std::flush;
#endif
        // update EPSPs if there's an incoming spike, but only for recurrent spikes
        if (it->deliveryTime + 1 == origin.get_steps() + lag && it->id_ <= P_.max_neuron_gid) {
            V_.yt_epsp_decay[it->id_] += V_.preSynWeights[it->id_];  // == rdyn*udyn, logic is in the synapse
            V_.yt_epsp_rise[it->id_] += V_.preSynWeights[it->id_];
            it = V_.spikeEvents.erase(it);
        } else {
            ++it;
        }
    }
    // TODO end

    for (auto &it: V_.activeSources) // iterated over all presynaptic nodes/sources
    {
        // iterate over each presyn index i and compute the weighted y_i(t) for the current time step
        V_.yt_epsp_traces[it] = V_.yt_epsp_decay[it] - V_.yt_epsp_rise[it];  // == self.Y // eq (2)
        u_t += V_.localWeights_Wk[it] * V_.yt_epsp_traces[it];  // add to sum after weighting with w_ki(t)  == eq.1

        // iterate over each presyn index and evolve rise and decay EPSP terms
        // TODO this are the exact solutions, but have to use lame numerics as in Klampfl for consistency reasons
//        V_.yt_epsp_decay[it] *= V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes;
//        V_.yt_epsp_rise[it] *= V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes;
        // TODO this appears to be crucial for similar results!
        V_.yt_epsp_decay[it] -= V_.yt_epsp_decay[it] * resolution / P_.tau_m; // evolve traces
        V_.yt_epsp_rise[it] -= V_.yt_epsp_rise[it] * resolution / P_.tau_syn; // evolve traces

#ifdef DEBUG
        std::cout  << "[[[ " << get_node_id() << " ]]] " << "[epsp] evolved EPSP " << V_.yt_epsp_decay[it] << "\t" << V_.yt_epsp_rise[it]
                    << " ==> Y(t) = " << V_.yt_epsp_traces[it]
                    << "\n" << std::flush;
#endif
    }
    S_.V_m = u_t;  // set V_m to current value of u(t)

    // evolve synaptic activations s_ij, i.e., process any spikes and add corresponding scaled delta impulses
    // This is done for the input weights
        // evolve synaptic activations s_ij, i.e., process any spikes and add corresponding scaled delta impulses
    for (auto it = V_.spikeEvents.begin(); it != V_.spikeEvents.end();) {
#ifdef DEBUG
        std::cout  << "[[[ " << get_node_id() << " ]]] " << "[update] spike from source " << it->id_ << ", scheduled @ " << it->deliveryTime
            << "; origin: " << origin.get_steps() << "; lag: " << lag << "\n" << std::flush;
#endif
        // update EPSPs if there's an incoming spike, but only for input spikes
        if (it->deliveryTime + 1 == origin.get_steps() + lag && it->id_ > P_.max_neuron_gid) {
            V_.yt_epsp_decay[it->id_] += V_.preSynWeights[it->id_];  // == rdyn*udyn, logic is in the synapse
            V_.yt_epsp_rise[it->id_] += V_.preSynWeights[it->id_];
            it = V_.spikeEvents.erase(it);
        } else {
            ++it;
        }
    }
}


void iaf_psc_exp_wta__with_stdp_stp::update(nest::Time const &origin, const long from, const long to) {
// IMPORTANT
    const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function
    double __t = 0;
    std::ostringstream msg;

    // allocate memory to store rates to be sent by rate events
    const size_t buffer_size = nest::kernel().connection_manager.get_min_delay();
    std::vector<double> new_voltage(buffer_size, 0.0);

    for (long lag = from; lag < to; ++lag) {
#ifdef DEBUG
        std::cout  << "[[[ " << get_node_id() << " ]]] " << "//////////////\nfrom " << from << " to " << to << ", lag = " << lag
                  << " @origin: " << origin.get_steps() << "\n" << std::flush;
#endif
        //B_.all_spikes_grid_sum_ = get_all_spikes().get_value(lag);
//        std::cout << "all_spikes_grid_sum_ = " << B_.all_spikes_grid_sum_ << "\n" << std::flush;
        S_.time_cnt += 1;
//      S_.V_m = get_y() * 1.0;
        //  std::cout << S_.V_m << std::endl;

//        V_.rate_fraction = std::exp(get_V_m() - V_.normalization_max) / get_normalization_sum();
        V_.rate_fraction = std::exp(get_V_m()) / get_normalization_sum();

//        std::cout << "[rate fraction ] " << get_node_id()
//            << "  --->>>  " << V_.rate_fraction
//            << std::endl << std::flush;

        if (V_.rate_fraction > 1.)
        {
            V_.rate = 0.;
        }
        else
        {
            V_.rate = P_.R_max * V_.rate_fraction;
        }

        evolve_epsps(origin, lag);  // modifies V_m

        double p = __resolution * V_.rate / 1000;

        bool use_fixed_spiketimes = V_.fixed_spiketimes.size() > 0;
        bool emit_spike = false;

        if (not use_fixed_spiketimes) {
            // assert(false);
            if (((0) + (1) * nest::get_vp_specific_rng(get_thread())->drand()) <= p) {
                emit_spike = true;
//                std::cout << "emitting spike " << V_.rate_fraction << std::endl << std::flush;
            }
        } else // iterate through spiketimes and fire if current step is firing step (set in python)
        {
            if (std::find(V_.fixed_spiketimes.begin(), V_.fixed_spiketimes.end(), get_time_cnt()) !=
                V_.fixed_spiketimes.end()) {
                emit_spike = true;
            }
        }

        if (emit_spike) {
            set_spiketime(nest::Time::step(origin.get_steps() + lag + 1));
            nest::SpikeEvent se;
//            se.set_sender(*this);
//            se.set_sender_node_id(this->get_node_id());
            se.set_sender_node_id(123456);
#ifdef DEBUG
            std::cout << "MODEL set SENDER NODE ID: " << this->get_node_id()
                        << "   double check: " << se.get_sender_node_id()
                        << std::endl << std::flush;
#endif

            nest::kernel().event_delivery_manager.send(*this, se, lag);

            evolve_weights(origin, lag);
        }

        // voltage logging
        B_.logger_.record_data(origin.get_steps() + lag);
        new_voltage[lag] = S_.V_m;

//        std::cout << "[setting voltage] " << get_node_id()
//            << "  --->>>  " << S_.V_m
//            << std::endl << std::flush;
    }

    // Send rate-neuron-event
    //std::cout << "Sending InstantaneousRateConnectionEvent..." << std::endl << std::flush;

    nest::InstantaneousRateConnectionEvent u_t_event;
    u_t_event.set_coeffarray(new_voltage);
    nest::kernel().event_delivery_manager.send_secondary(*this, u_t_event);

    // could reset the normalization factor to 0 here
    V_.normalization_max_prev = V_.normalization_max;  // store max if we need to sum later on
    V_.normalization_max = -1e12;  // reset to 0 as preparation for next update step

    V_.presyn_u_t.clear();
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void iaf_psc_exp_wta__with_stdp_stp::handle(nest::DataLoggingRequest &e) {
    B_.logger_.handle(e);
}

void iaf_psc_exp_wta__with_stdp_stp::handle(nest::SpikeEvent &e) // happens before update
{
// IMPORTANT
    assert(e.get_delay_steps() > 0);
    const double weight = e.get_weight();
    const double multiplicity = e.get_multiplicity();

//    get_all_spikes().
//        add_value(e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
//                       weight * multiplicity );
    long deliveryTime = e.get_rel_delivery_steps(nest::kernel().simulation_manager.get_slice_origin());

//#ifdef DEBUG
//    std::cout << e.get_sender_node_id() << std::flush;
//#endif

#ifdef DEBUG
    std::cout  << "[[[ " << get_node_id() << " ]]] " << "\n\t\tweight: " << weight
        << "\n\t\t sender id: " << e.get_sender_node_id()
        << "\n\t\t TIME: (event tstamp) " << e.get_stamp().get_steps()
        << "\n\t\t slice_origin(): " << nest::kernel().simulation_manager.get_slice_origin()
        << "\n\t\t get_delay_steps(): " << e.get_delay_steps()
        << "\n\t\t deliveryTime: " << deliveryTime
        << "\n\t\t absolut deliveryTime (! precise): " << e.get_stamp().get_steps() + e.get_delay_steps() - 2
        << "\n" << std::flush;
#endif
//    const double fixed_weight_epsp = 1;
    TraceTracker_ spikeEventStruct = {e.get_stamp().get_steps() + e.get_delay_steps() - 2,
                                      weight, e.get_sender_node_id(), 0, false,
                                      true}; // defined in header, stores spike id

    V_.spikeEvents.push_back(spikeEventStruct);
    V_.preSynWeights[e.get_sender_node_id()] = weight;  // this just takes into account the STP part
    V_.activeSources.insert(e.get_sender_node_id());
    get_all_spikes().add_value(deliveryTime, weight * multiplicity);
}


void iaf_psc_exp_wta__with_stdp_stp::handle(nest::InstantaneousRateConnectionEvent &e) {
//    IMPORTANT
    // std::cout << "Received InstantaneousRateConnectionEvent " << std::endl << std::flush;

    size_t i = 0;
    std::vector<unsigned int>::iterator it = e.begin();
    // The call to get_coeffvalue( it ) in this loop also advances the iterator it
    while (it != e.end()) {
        double u_i = e.get_coeffvalue(it);
        V_.presyn_u_t.push_back(u_i);  // store raw u_i(t)s in vector for later

        // update maximum of u(t) s to be used during normalization - consistency with Klampfl
        V_.normalization_max = std::max(u_i, V_.normalization_max);

//        B_.instant_rates_ex_[i] += weight * e.get_coeffvalue(it);
//        std::cout << "[neuron: " <<  get_node_id() << "] Received RateConnectionEvent - u(t) of presyn neuron:"
//            << u_i << std::endl << std::flush;
//            << u_i << std::endl << std::flush;
    }
}


inline double
iaf_psc_exp_wta__with_stdp_stp::get_spiketime_ms() const {
    return last_spike_;
}


void
iaf_psc_exp_wta__with_stdp_stp::register_stdp_connection(double t_first_read, double delay) {
    // Mark all entries in the deque, which we will not read in future as read by
    // this input input, so that we safely increment the incoming number of
    // connections afterwards without leaving spikes in the history.
    // For details see bug #218. MH 08-04-22

    for (std::deque<histentry__iaf_psc_exp_wta__with_stdp_stp>::iterator runner = history_.begin();
         runner != history_.end() and
         (t_first_read - runner->t_ > -1.0 * nest::kernel().connection_manager.get_stdp_eps());
         ++runner) {
        (runner->access_counter_)++;
    }

    n_incoming_++;

    max_delay_ = std::max(delay, max_delay_);
}


void
iaf_psc_exp_wta__with_stdp_stp::get_history__(double t1,
                                              double t2,
                                              std::deque<histentry__iaf_psc_exp_wta__with_stdp_stp>::iterator *start,
                                              std::deque<histentry__iaf_psc_exp_wta__with_stdp_stp>::iterator *finish) {
    *finish = history_.end();
    if (history_.empty()) {
        *start = *finish;
        return;
    }
    std::deque<histentry__iaf_psc_exp_wta__with_stdp_stp>::reverse_iterator runner = history_.rbegin();
    const double t2_lim = t2 + nest::kernel().connection_manager.get_stdp_eps();
    const double t1_lim = t1 + nest::kernel().connection_manager.get_stdp_eps();
    while (runner != history_.rend() and runner->t_ >= t2_lim) {
        ++runner;
    }
    *finish = runner.base();
    while (runner != history_.rend() and runner->t_ >= t1_lim) {
        runner->access_counter_++;
        ++runner;
    }
    *start = runner.base();
}

void
iaf_psc_exp_wta__with_stdp_stp::set_spiketime(nest::Time const &t_sp, double offset) {
    ArchivingNode::set_spiketime(t_sp, offset);

    unsigned int num_transferred_variables = 0;

    const double t_sp_ms = t_sp.get_ms() - offset;

    if (n_incoming_) {
        // prune all spikes from history which are no longer needed
        // only remove a spike if:
        // - its access counter indicates it has been read out by all connected
        //     STDP synapses, and
        // - there is another, later spike, that is strictly more than
        //     (max_delay_ + eps) away from the new spike (at t_sp_ms)
        while (history_.size() > 1) {
            const double next_t_sp = history_[1].t_;
            if (history_.front().access_counter_ >= n_incoming_ * num_transferred_variables
                and t_sp_ms - next_t_sp > max_delay_ + nest::kernel().connection_manager.get_stdp_eps()) {
                history_.pop_front();
            } else {
                break;
            }
        }

        if (history_.size() > 0) {
            assert(history_.back().t_ == last_spike_);
        } else {
        }


        /**
         * update state variables transferred from synapse from `last_spike_` to `t_sp_ms`
        **/

        const double old___h = V_.__h;
        V_.__h = t_sp_ms - last_spike_;
        if (V_.__h > 1E-12) {
            recompute_internal_variables(true);
            /* generated by directives/AnalyticIntegrationStep_begin.jinja2 */
            /* replace analytically solvable variables with precisely integrated values  *//* generated by directives/AnalyticIntegrationStep_end.jinja2 */
            V_.__h = old___h;
            recompute_internal_variables(true);
        }

        /**
         * apply spike updates
        **/

        last_spike_ = t_sp_ms;
        history_.push_back(histentry__iaf_psc_exp_wta__with_stdp_stp(last_spike_, 0
        ));
    } else {
        last_spike_ = t_sp_ms;
    }
}


void
iaf_psc_exp_wta__with_stdp_stp::clear_history() {
    last_spike_ = -1.0;
    history_.clear();
}



