extern crate rand;
extern crate rsmcmc;

use rand::thread_rng;
use rsmcmc::graph::graph::Graph;
use rsmcmc::graph::graph_var::GraphVar;
use rsmcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use rsmcmc::ensemble_sample::sample;
use rsmcmc::utils::HasLength;
use rsmcmc::init_ensemble::get_one_init_realization;

fn main() {
    let mut g = Graph::new();
    g.add_node("mean", const_node(1.0)).done();
    g.add_node("sigma", const_node(1.0)).done();
    g.add_node("x", normal_node())
        .with_parent("mean", 0)
        .with_parent("sigma", 0)
        .with_value(0, 1.0)
        .done();

    g.seal();

    let gv = g.init_gv();
    let mut gv1 = gv.clone();
    let mut gv2 = gv.clone();
    gv1[0] = -1.0;
    gv2[0] = 1.0;
    //println!("{}" ,g.logpost_all(&gv));
    //println!("{}", gv.length());

    //gv[0]=2;
    let mut rng = thread_rng();

    let mut ensemble = Vec::new();

    for i in 0..4 {
        ensemble.push(get_one_init_realization(&gv1, &gv2, &mut rng));
    }

    let mut logprob = Vec::<f64>::new();
    let f = move |x: &GraphVar<f64>| g.logpost_all(x);
    for i in 0..10000 {
        let aaa = sample(&f, (ensemble, logprob), &mut rng, 2.0, 1).unwrap();
        ensemble = aaa.0;
        logprob = aaa.1;
        println!("{}", ensemble[0][0]);
    }
}
