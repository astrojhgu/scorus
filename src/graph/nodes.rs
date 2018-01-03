use std;
use super::node::Node;
use super::node::NodeContent;
use super::node::BasicNode;
use num_traits::float::Float;
use std::boxed::Box;
use num_traits::identities::one;
use num_traits::identities::zero;
use super::graph::NodeAdder;
use super::graph::NodeHandle;

pub fn const_node<T>(v: T) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 0,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![v]),
        },
    };
    NodeAdder::new(n, &[])
}

pub fn add_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![x[0] + x[1]]),
        },
    };
    NodeAdder::new(n,&[a, b])
}

pub fn cos_node<T>(a: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 1,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x: &[T]| vec![x[0].cos()]),
        },
    };
    NodeAdder::new(n,&[a])
}

pub fn normal_node<T>(m: (NodeHandle, usize), s: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::StochasticNode {
            all_stochastic_children: Vec::new(),
            all_deterministic_children: Vec::new(),
            is_observed: vec![false],
            values: vec![zero()],
            logprob: Box::new(move |x, p| {
                let x = x[0];
                let m = p[0];
                let s = p[1];
                return -((x - m) * (x - m) / ((one::<T>() + one::<T>()) * s * s)) - s;
            }),
        },
    };
    NodeAdder::new(n,&[m, s])
}

pub fn uniform_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::StochasticNode {
            all_stochastic_children: Vec::new(),
            all_deterministic_children: Vec::new(),
            is_observed: vec![false],
            values: vec![zero()],
            logprob: Box::new(move |x, p| one()),
        },
    };

    NodeAdder::new(n,&[a, b])
}
