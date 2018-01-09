extern crate rand;
extern crate rsmcmc;

use rand::thread_rng;
use rsmcmc::graph::graph::Graph;
use rsmcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use rsmcmc::graph::graph_var::GraphVar;
use rsmcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use rsmcmc::ensemble_sample::sample;
use rsmcmc::utils::HasLength;
use rsmcmc::init_ensemble::get_one_init_realization;

fn main() {}
