// #define DEBUG 1
/* generated by common/NeuronClass.jinja2 *//*
 *  iaf_psc_exp_wta.cpp
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
 *  Generated from NESTML at time: 2022-09-17 19:34:29.462010
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

#include "iaf_psc_exp_wta.h"

// ---------------------------------------------------------------------------
//   Recordables map
// ---------------------------------------------------------------------------
nest::RecordablesMap<iaf_psc_exp_wta> iaf_psc_exp_wta::recordablesMap_;
namespace nest
{

  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
template <> void RecordablesMap<iaf_psc_exp_wta>::create()
  {
    // add state variables to recordables map
   insert_(iaf_psc_exp_wta_names::_V_m, &iaf_psc_exp_wta::get_V_m);
   insert_(iaf_psc_exp_wta_names::_decay_time_kernel__X__all_spikes, &iaf_psc_exp_wta::get_decay_time_kernel__X__all_spikes);
   insert_(iaf_psc_exp_wta_names::_rise_time_kernel__X__all_spikes, &iaf_psc_exp_wta::get_rise_time_kernel__X__all_spikes);

    // Add vector variables  
  }
}

// ---------------------------------------------------------------------------
//   Default constructors defining default parameters and state
//   Note: the implementation is empty. The initialization is of variables
//   is a part of iaf_psc_exp_wta's constructor.
// ---------------------------------------------------------------------------

iaf_psc_exp_wta::Parameters_::Parameters_()
{
}

iaf_psc_exp_wta::State_::State_()
{
}

// ---------------------------------------------------------------------------
//   Parameter and state extractions and manipulation functions
// ---------------------------------------------------------------------------

iaf_psc_exp_wta::Buffers_::Buffers_(iaf_psc_exp_wta &n):
  logger_(n)
{
  // Initialization of the remaining members is deferred to init_buffers_().
}

iaf_psc_exp_wta::Buffers_::Buffers_(const Buffers_ &, iaf_psc_exp_wta &n):
  logger_(n)
{
  // Initialization of the remaining members is deferred to init_buffers_().
}

// ---------------------------------------------------------------------------
//   Default constructor for node
// ---------------------------------------------------------------------------

iaf_psc_exp_wta::iaf_psc_exp_wta():ArchivingNode(), P_(), S_(), B_(*this)
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function
  pre_run_hook();
  // initial values for parameters
    /* generated by directives/MemberInitialization.jinja2 */ 
    P_.tau_m = 20; // as ms
    /* generated by directives/MemberInitialization.jinja2 */ 
    P_.tau_syn = 2; // as ms
    /* generated by directives/MemberInitialization.jinja2 */ 
    P_.R_max = 100; // as Hz
  // initial values for state variables
    /* generated by directives/MemberInitialization.jinja2 */ 
    S_.r = 0; // as integer
    /* generated by directives/MemberInitialization.jinja2 */ 
    S_.V_m = 0; // as mV
    /* generated by directives/MemberInitialization.jinja2 */ 
    S_.time_cnt = 0; // as integer
    /* generated by directives/MemberInitialization.jinja2 */ 
    S_.decay_time_kernel__X__all_spikes = 0; // as real
    /* generated by directives/MemberInitialization.jinja2 */ 
    S_.rise_time_kernel__X__all_spikes = 0; // as real
  recordablesMap_.create();
}

// ---------------------------------------------------------------------------
//   Copy constructor for node
// ---------------------------------------------------------------------------

iaf_psc_exp_wta::iaf_psc_exp_wta(const iaf_psc_exp_wta& __n):
  ArchivingNode(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this) {

  // copy parameter struct P_
  P_.tau_m = __n.P_.tau_m;
  P_.tau_syn = __n.P_.tau_syn;
  P_.R_max = __n.P_.R_max;

  // copy state struct S_
  S_.r = __n.S_.r;
  S_.V_m = __n.S_.V_m;
  S_.time_cnt = __n.S_.time_cnt;
  S_.decay_time_kernel__X__all_spikes = __n.S_.decay_time_kernel__X__all_spikes;
  S_.rise_time_kernel__X__all_spikes = __n.S_.rise_time_kernel__X__all_spikes;


  // copy internals V_
  V_.__h = __n.V_.__h;
  V_.__P__V_m__V_m = __n.V_.__P__V_m__V_m;
  V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = __n.V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes;
  V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = __n.V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes;
}

// ---------------------------------------------------------------------------
//   Destructor for node
// ---------------------------------------------------------------------------

iaf_psc_exp_wta::~iaf_psc_exp_wta()
{
}

// ---------------------------------------------------------------------------
//   Node initialization functions
// ---------------------------------------------------------------------------

void iaf_psc_exp_wta::init_buffers_()
{
  get_all_spikes().clear(); //includes resize
  B_.logger_.reset(); // includes resize
}

void iaf_psc_exp_wta::recompute_internal_variables(bool exclude_timestep) {
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

  if (exclude_timestep) {    
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__P__V_m__V_m = 1; // as real
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = std::exp((-(V_.__h)) / P_.tau_m); // as real
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = std::exp((-(V_.__h)) / P_.tau_syn); // as real
  }
  else {
    // internals V_
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__h = __resolution; // as ms
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__P__V_m__V_m = 1; // as real
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes = std::exp((-(V_.__h)) / P_.tau_m); // as real
      /* generated by directives/MemberInitialization.jinja2 */ 
      V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes = std::exp((-(V_.__h)) / P_.tau_syn); // as real
  }
}
void iaf_psc_exp_wta::pre_run_hook() {
  B_.logger_.init();

  recompute_internal_variables();

  // buffers B_
}

// ---------------------------------------------------------------------------
//   Update and spike handling functions
// ---------------------------------------------------------------------------


void iaf_psc_exp_wta::update(nest::Time const & origin,const long from, const long to)
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function



  for ( long lag = from ; lag < to ; ++lag )
  {
    B_.all_spikes_grid_sum_ = get_all_spikes().get_value(lag);

    // NESTML generated code for the update block:/* generated by directives/Block.jinja2 */ /* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/FunctionCall.jinja2 */ /* generated by directives/AnalyticIntegrationStep_begin.jinja2 */ 
  double V_m__tmp = get_V_m() * V_.__P__V_m__V_m;
  double decay_time_kernel__X__all_spikes__tmp = V_.__P__decay_time_kernel__X__all_spikes__decay_time_kernel__X__all_spikes * get_decay_time_kernel__X__all_spikes();
  double rise_time_kernel__X__all_spikes__tmp = V_.__P__rise_time_kernel__X__all_spikes__rise_time_kernel__X__all_spikes * get_rise_time_kernel__X__all_spikes();
  /* replace analytically solvable variables with precisely integrated values  *//* generated by directives/AnalyticIntegrationStep_end.jinja2 */ 
  S_.V_m = V_m__tmp;
  S_.decay_time_kernel__X__all_spikes = decay_time_kernel__X__all_spikes__tmp;
  S_.rise_time_kernel__X__all_spikes = rise_time_kernel__X__all_spikes__tmp;/* generated by directives/ApplySpikesFromBuffers.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
      S_.decay_time_kernel__X__all_spikes += (B_.all_spikes_grid_sum_) / (1.0);/* generated by directives/Assignment.jinja2 */ 
      S_.rise_time_kernel__X__all_spikes += (B_.all_spikes_grid_sum_) / (1.0);/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
      S_.time_cnt += 1;/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
      S_.V_m = get_y() * 1.0;/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Declaration.jinja2 */ 
  double rate = P_.R_max * std::exp(get_V_m() / 1.0);/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Declaration.jinja2 */ 
  double p = __resolution * rate / 1000;/* generated by directives/Statement.jinja2 */ /* generated by directives/CompoundStatement.jinja2 */ /* generated by directives/IfStatement.jinja2 */ 
  if (((0) + (1) * nest::get_vp_specific_rng( get_thread() )->drand())<=p)
  {/* generated by directives/Block.jinja2 */ /* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
      p = 1;
  }/* generated by directives/Statement.jinja2 */ /* generated by directives/CompoundStatement.jinja2 */ /* generated by directives/IfStatement.jinja2 */ 
  if (get_time_cnt()==200||get_time_cnt()==500)
  {/* generated by directives/Block.jinja2 */ /* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/FunctionCall.jinja2 */ 
  set_spiketime(nest::Time::step(origin.get_steps()+lag+1));
  nest::SpikeEvent se;
  nest::kernel().event_delivery_manager.send(*this, se, lag);
  }

    // voltage logging
    B_.logger_.record_data(origin.get_steps() + lag);
  }
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void iaf_psc_exp_wta::handle(nest::DataLoggingRequest& e)
{
  B_.logger_.handle(e);
}

void iaf_psc_exp_wta::handle(nest::SpikeEvent &e)
{
  assert(e.get_delay_steps() > 0);
  const double weight = e.get_weight();
  const double multiplicity = e.get_multiplicity();
  // this port receives excitatory spikes
  if ( weight >= 0.0 )
  {
    get_all_spikes().
        add_value(e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
                       weight * multiplicity );
  }
}
