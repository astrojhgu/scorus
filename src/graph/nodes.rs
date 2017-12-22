use std;
use super::node::Node;
use super::node::NodeContent;
use super::node::BasicNode;
use num_traits::float::Float;
use std::boxed::Box;
use num_traits::identities::one;
use num_traits::identities::zero;

pub fn const_node<T>(v: T) -> Node<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    Node {
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
    }
}

pub fn add_node<T>() -> Node<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    Node {
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
    }
}

pub fn cos_node<T>() -> Node<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 1,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![x[0].cos()]),
        },
    }
}

pub fn normal_node<T>() -> Node<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    Node {
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
            is_observed: Vec::new(),
            values: vec![zero()],
            logprob: Box::new(move |x, p| {
                let x = x[0];
                let m = p[0];
                let s = p[1];
                return -((x - m) * (x - m) / ((one::<T>() + one::<T>()) * s * s)) - s;
            }),
        },
    }
}

pub fn uniform_node<T>() -> Node<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    Node {
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
            is_observed: Vec::new(),
            values: vec![zero()],
            logprob: Box::new(move |x, p| one()),
        },
    }
}
