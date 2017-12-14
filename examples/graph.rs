extern crate rsmcmc;

use rsmcmc::graph::graph::Graph;
use rsmcmc::graph::nodes::{const_node, normal_node, add_node};

fn main(){
    let mut g=Graph::<String, f64>::new();
    g.add_node("a".to_string(), const_node(5.0)).done();


    g.add_node("b".to_string(), const_node(6.0)).done();

    g.add_node("c".to_string(), normal_node()).with_parent("a".to_string(), 0)
        .with_parent("b".to_string(), 0).done();

    g.add_node("d".to_string(), const_node(3.0)).done();

    g.add_node("e".to_string(), add_node()).with_parent("d".to_string(), 0).with_parent("c".to_string(),0).done();

    g.add_node("f".to_string(), normal_node()).with_parent("e".to_string(), 0).with_parent("c".to_string(),0).done();

    g.seal();
    //println!("{:?}", g.enumerate_stochastic_children(2));
    println!("{}", g);
}