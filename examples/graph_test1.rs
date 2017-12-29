extern crate rsmcmc;
extern crate rand;

//use std;
use rand::thread_rng;
use rsmcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use rand::Rand;
use std::vec::Vec;
use rsmcmc::graph::graph::Graph;
use rsmcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use rsmcmc::ensemble_sample::sample_st;


fn main(){
    let mut g=Graph::new();
    g.add_node("s1", const_node(0.1)).done();
    g.add_node("s2", const_node(2.0)).done();
    g.add_node("s", uniform_node()).with_parent("s1", 0).with_parent("s2", 0).with_all_values(&[UnObserved(1.0)]).done();
    g.add_node("m", const_node(0.5)).done();


    g.add_node("y", normal_node()).with_parent("m",0).with_parent("s",0).with_all_values(&[UnObserved(1.0)]).done();

    let mut gv=g.init_gv();

    println!("{}", &gv);

    let mut ensemble=Vec::new();


    ensemble.push(1);

    //sample_st(|x|{g.logpost_all(x)}, )

}