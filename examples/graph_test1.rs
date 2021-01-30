#![allow(unused_imports)]
extern crate rand;
extern crate scorus;
extern crate rand_distr;
//use std;
use rand::{
    distributions::{
        Distribution
    }
    , Rng
};
use rand::thread_rng;
use rand_distr::{
    Normal
};
use scorus::mcmc::ensemble_sample::{sample, UpdateFlagSpec};
use scorus::mcmc::graph::graph::Graph;
use scorus::mcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use scorus::mcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use scorus::utils::HasLen;
use std::vec::Vec;

fn main() {
    let mut g = Graph::new();

    let m_lower = const_node(-1.0).add_to(&mut g, &"m_lower".to_string());
    let m_upper = const_node(1.0).add_to(&mut g, &"m_upper".to_string());

    let s_lower = const_node(0.001).add_to(&mut g, &"s_lower".to_string());
    let s_upper = const_node(10.0).add_to(&mut g, &"s_upper".to_string());

    let m = uniform_node((m_lower, 0), (m_upper, 0))
        .with_all_values(&[UnObserved(0.0)])
        .add_to(&mut g, &"m".to_string());
    let s = uniform_node((s_lower, 0), (s_upper, 0))
        .with_all_values(&[UnObserved(2.0)])
        .add_to(&mut g, &"s".to_string());

    let mut rng = thread_rng();

    let norm = Normal::new(0.0, 2.0).unwrap();

    for i in 0..10000 {
        let k = format!("n{}", i);
        normal_node((m, 0), (s, 0))
            .with_all_values(&[Observed(rng.sample(norm))])
            .add_to(&mut g, &k);
    }

    //normal_node((m,0),(s,0)).with_all_values(&[UnObserved(1.0)]).add_to(&mut g, &"x1".to_string());
    //normal_node((m,0),(s,0)).with_all_values(&[UnObserved(1.0)]).add_to(&mut g, &"x2".to_string());

    g.seal();
    let mut gv = g.init_gv();

    //println!("{}", g.logpost_all(&gv));

    let mut ensemble = Vec::new();
    let mut nchanged = 0;
    g.sample_all(&mut gv, &mut rng, 100, &mut nchanged);
    ensemble.push(gv.clone());
    g.sample_all(&mut gv, &mut rng, 100, &mut nchanged);
    ensemble.push(gv.clone());
    g.sample_all(&mut gv, &mut rng, 100, &mut nchanged);
    ensemble.push(gv.clone());
    g.sample_all(&mut gv, &mut rng, 100, &mut nchanged);
    ensemble.push(gv.clone());

    let mut lp = Vec::new();
    for _i in 0..10000 {
        //let aa = sample_st(&|x| g.logpost_all(x), &(ensemble, lp), &mut rng, 2.0).unwrap();
        //ensemble = aa.0;
        //lp = aa.1;
        sample(
            &|x| g.logpost_all(x),
            &mut ensemble,
            &mut lp,
            &mut rng,
            2.0,
            &mut UpdateFlagSpec::Prob(0.01),
        );
        println!("{} {}", ensemble[0][0], ensemble[0][1]);
    }
}
