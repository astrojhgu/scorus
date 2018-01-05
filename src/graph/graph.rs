use std;

use super::node::Node;
use super::node::NodeContent;
use super::node::BasicNode;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use super::node::ValueType;
use std::rc::Rc;
use num_traits::float::Float;
use std::fmt::{Display, Error, Formatter};
use std::iter::FromIterator;
use super::graph_var::GraphVar;
use num_traits::identities::zero;
use std::boxed::Box;
use std::option::Option;
use std::ops::Index;
use std::convert::From;

#[derive(Debug)]
pub struct NodeHandle(usize);

impl Clone for NodeHandle {
    fn clone(&self) -> NodeHandle {
        NodeHandle(self.0)
    }
}

impl Copy for NodeHandle {}

pub struct Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + Display,
{
    key_node_map: HashMap<K, usize>,
    node_key_map: HashMap<usize, K>,
    nodes: Vec<Node<T>>,
    num_of_fixed_vars: usize,
    num_of_deterministic_vars: usize,
    num_of_sampleable_vars: usize,
}

impl<K, T> Display for Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone + Display,
    T: Float + Sync + Send + Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "nodes:\n")?;
        for (i, n) in self.nodes.iter().enumerate() {
            write!(f, "Node: {}, {}\n", i, self.node_key_map.get(&i).unwrap());
            write!(f, "{} \n", &n)?;
            write!(f, "========\n")?;
        }

        for (k, id) in &self.key_node_map {
            write!(f, "{}:{} ", k, id)?;
        }

        write!(f, "\nNum of fixed: {}\n", self.num_of_fixed_vars)?;
        write!(
            f,
            "Num of deterministic: {}\n",
            self.num_of_deterministic_vars
        )?;
        write!(f, "Num of sampleable: {}\n", self.num_of_sampleable_vars)?;
        Ok(())
    }
}

pub enum ParamObservability<T> {
    Observed(T),
    UnObserved(T),
}

impl<T> std::clone::Clone for ParamObservability<T>
where
    T: std::marker::Copy,
{
    fn clone(&self) -> Self {
        match self {
            &ParamObservability::Observed(x) => ParamObservability::Observed(x),
            &ParamObservability::UnObserved(x) => ParamObservability::UnObserved(x),
        }
    }
}

impl<T> std::marker::Copy for ParamObservability<T>
where
    T: std::marker::Copy,
{
}

pub struct NodeAdder<T>
where
    T: Float + Sync + Send + Display,
{
    n: Node<T>,
    parents: Vec<(NodeHandle, usize)>,
}

impl<T> NodeAdder<T>
where
    T: Float + Sync + Send + Display,
{
    pub fn new(n: Node<T>, p:&[(NodeHandle, usize)]) -> Self {
        if p.len()!=n.info.ndim_input{
            panic!("Number of parents mismatch");
        }

        NodeAdder {
            n: n,
            parents: Vec::<(NodeHandle, usize)>::from(p)
        }
    }

    pub fn with_all_values(mut self, x: &[ParamObservability<T>]) -> Self {
        if let NodeContent::StochasticNode {
            ref mut is_observed,
            ref mut values,
            ..
        } = self.n.content
        {
            if x.len() != self.n.info.ndim_output {
                panic!("Error, number of values not same as noutput");
            }
            is_observed.clear();
            values.clear();
            for i in 0..x.len() {
                match x[i] {
                    ParamObservability::Observed(ref x1) => {
                        values.push(*x1);
                        is_observed.push(true);
                    }
                    ParamObservability::UnObserved(ref x1) => {
                        values.push(*x1);
                        is_observed.push(false);
                    }
                }
            }
        } else {
            panic!("It is not a stochastic node");
        }
        self
    }

    pub fn add_to<K>(mut self, g: &mut Graph<K, T>, k: &K) -> NodeHandle
    where
        K: std::hash::Hash + Eq + Clone,
    {
        let nid = g.nodes.len();

        if g.key_node_map.contains_key(k){
            panic!("Error, key has already existed");
        }

        if self.parents.len() != self.n.info.ndim_input {
            panic!("parents mismatch");
        }

        match self.n {
            Node {
                info:
                    BasicNode {
                        ref parents,
                        ref mut idx_in_var,
                        ref mut value_type,
                        ..
                    },
                content:
                    NodeContent::StochasticNode {
                        ref is_observed, ..
                    },
            } => {
                idx_in_var.clear();
                value_type.clear();
                for &ob in is_observed {
                    if ob {
                        idx_in_var.push(g.num_of_fixed_vars);
                        value_type.push(ValueType::FIXED);
                        g.num_of_fixed_vars += 1;
                    } else {
                        idx_in_var.push(g.num_of_sampleable_vars);
                        value_type.push(ValueType::SAMPLEABLE);
                        g.num_of_sampleable_vars += 1;
                    }
                }
            }
            Node {
                info:
                    BasicNode {
                        ref parents,
                        ref mut idx_in_var,
                        ndim_input,
                        ndim_output,
                        ref mut value_type,
                        ..
                    },
                content: NodeContent::DeterministicNode { .. },
            } => {
                idx_in_var.clear();
                value_type.clear();
                if ndim_input == 0 {
                    for i in 0..ndim_output {
                        idx_in_var.push(g.num_of_fixed_vars);
                        value_type.push(ValueType::FIXED);
                        g.num_of_fixed_vars += 1;
                    }
                } else {
                    for i in 0..ndim_output {
                        idx_in_var.push(g.num_of_deterministic_vars);
                        value_type.push(ValueType::DETERMINISTIC);
                        g.num_of_deterministic_vars += 1;
                    }
                }
            }
        }

        for &(k, i) in &self.parents {
            let pid = k.0;
            self.n.info.parents.push((pid, i));
            g.nodes[pid].info.children.push(nid);
        }

        g.key_node_map.insert(k.clone(), nid);
        g.node_key_map.insert(nid, k.clone());

        g.nodes.push(self.n);

        NodeHandle(nid)
    }
}

impl<K, T> Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + Display,
{
    pub fn new() -> Graph<K, T> {
        Graph {
            key_node_map: HashMap::new(),
            node_key_map: HashMap::new(),
            nodes: Vec::new(),
            num_of_fixed_vars: 0,
            num_of_deterministic_vars: 0,
            num_of_sampleable_vars: 0,
        }
    }

    pub fn enumerate_children_by_kind(&self, nid: usize) -> (BTreeSet<usize>, BTreeSet<usize>) {
        type Stack = Vec<usize>;
        let mut result0 = BTreeSet::<usize>::new();
        let mut result1 = BTreeSet::<usize>::new();
        let mut stack = Stack::new();
        stack.push(nid);
        while !stack.is_empty() {
            let top = stack.pop().unwrap();

            for i in self.nodes[top].get_children() {
                match self.nodes[*i].content {
                    NodeContent::StochasticNode { .. } => {
                        result0.insert(*i);
                    }
                    NodeContent::DeterministicNode { .. } => {
                        result1.insert(*i);
                        stack.push(*i);
                    }
                }
            }
        }
        (result0, result1)
    }

    pub fn seal(&mut self) {
        for i in 0..self.nodes.len() {
            let (s, d) = self.enumerate_children_by_kind(i);
            match &mut self.nodes[i].content {
                &mut NodeContent::StochasticNode {
                    ref mut all_stochastic_children,
                    ref mut all_deterministic_children,
                    ..
                } => {
                    *all_stochastic_children = Vec::from_iter(s);
                    *all_deterministic_children = Vec::from_iter(d);
                }
                _ => {}
            }
        }
    }

    pub fn init_gv(&self) -> GraphVar<T> {
        let mut gv = GraphVar {
            fixed_values: RefCell::new(Vec::new()),
            deterministic_values: RefCell::new(Vec::new()),
            sampleable_values: Vec::new(),
        };

        gv.fixed_values
            .borrow_mut()
            .resize(self.num_of_fixed_vars, zero());
        gv.deterministic_values
            .borrow_mut()
            .resize(self.num_of_deterministic_vars, zero());
        gv.sampleable_values
            .resize(self.num_of_sampleable_vars, zero());

        for (i, n) in self.nodes.iter().enumerate() {
            match n.content {
                NodeContent::DeterministicNode { .. } => {
                    self.update_deterministic_value_of(i, &mut gv);
                }
                NodeContent::StochasticNode {
                    ref is_observed,
                    ref values,
                    ..
                } => for (j, p) in n.info.idx_in_var.iter().enumerate() {
                    if is_observed[j] {
                        gv.fixed_values.borrow_mut()[*p] = values[j];
                    } else {
                        gv[*p] = values[j];
                    }
                },
            }
        }
        gv
    }

    pub fn cached_value_of(&self, i: usize, j: usize, gv: &GraphVar<T>) -> T {
        match self.nodes[i].info.value_type[j] {
            ValueType::DETERMINISTIC => {
                gv.deterministic_values.borrow_mut()[self.nodes[i].info.idx_in_var[j]]
            }
            ValueType::FIXED => gv.fixed_values.borrow_mut()[self.nodes[i].info.idx_in_var[j]],
            ValueType::SAMPLEABLE => gv.sampleable_values[self.nodes[i].info.idx_in_var[j]],
        }
    }

    pub fn cached_values_of(&self, i: usize, gv: &GraphVar<T>) -> Vec<T> {
        let mut result = Vec::<T>::new();
        result.reserve(self.nodes[i].info.ndim_output);
        for j in 0..self.nodes[i].info.ndim_output {
            result.push(self.cached_value_of(i, j, gv));
        }
        result
    }

    pub fn parent_values_of(&self, i: usize, gv: &GraphVar<T>) -> Vec<T> {
        let mut result = Vec::new();
        let node = &self.nodes[i];
        result.reserve(node.info.ndim_input);
        for j in 0..node.info.ndim_input {
            let (p, k) = node.info.parents[j];
            result.push(self.cached_value_of(p, k, gv));
        }
        result
    }

    pub fn update_deterministic_value_of(&self, i: usize, gv: &GraphVar<T>) {
        let pv = self.parent_values_of(i, gv);
        let node = &self.nodes[i];
        match node.content {
            NodeContent::DeterministicNode { ref calc, .. } => {
                let v = calc(&pv);
                for (m, k) in node.info.idx_in_var.iter().enumerate() {
                    match node.info.value_type[m] {
                        ValueType::FIXED => {
                            gv.fixed_values.borrow_mut()[*k] = v[m];
                            //panic!("Impossible")
                        }
                        ValueType::DETERMINISTIC => {
                            gv.deterministic_values.borrow_mut()[*k] = v[m];
                        }
                        ValueType::SAMPLEABLE => panic!("Impossible"),
                    }
                }
            }
            NodeContent::StochasticNode {
                ref logprob,
                ref values,
                ..
            } => for (m, k) in node.info.idx_in_var.iter().enumerate() {
                match node.info.value_type[m] {
                    ValueType::FIXED => {
                        //panic!("Impossible")
                        gv.fixed_values.borrow_mut()[*k] = values[m];
                    }
                    ValueType::DETERMINISTIC => {
                        panic!("Impossible");
                    }
                    ValueType::SAMPLEABLE => {}
                }
            },
        }
    }

    pub fn logprob(&self, i: usize, gv: &GraphVar<T>) -> T {
        if let NodeContent::StochasticNode { ref logprob, .. } = self.nodes[i].content {
            let x = self.cached_values_of(i, gv);
            let p = self.parent_values_of(i, gv);
            logprob(x.as_slice(), p.as_slice())
        } else {
            panic!("not a stochastic node");
        }
    }

    pub fn range(&self, i:usize, gv:&GraphVar<T>) -> Vec<(T, T)>{
        if let NodeContent::StochasticNode { ref range, .. } = self.nodes[i].content {
            let x = self.cached_values_of(i, gv);
            let p = self.parent_values_of(i, gv);
            range(p.as_slice())
        } else {
            panic!("not a stochastic node");
        }
    }

    pub fn likelihood(&self, i: usize, gv: &GraphVar<T>) -> T {
        let mut result = zero();
        if let NodeContent::StochasticNode {
            ref all_stochastic_children,
            ..
        } = self.nodes[i].content
        {
            for n in all_stochastic_children {
                result = result + self.logprob(*n, gv);
            }
        }
        result
    }

    pub fn logpost(&self, i: usize, gv: &GraphVar<T>) -> T {
        let mut result = zero();
        if let NodeContent::StochasticNode {
            ref all_stochastic_children,
            ref logprob,
            ..
        } = self.nodes[i].content
        {
            let x = self.cached_values_of(i, gv);
            let p = self.parent_values_of(i, gv);
            for n in all_stochastic_children {
                result = result + self.logprob(*n, gv);
            }
            result = result + logprob(x.as_slice(), p.as_slice());
        }
        result
    }

    pub fn update_deterministic_children(&self, i: usize, gv: &GraphVar<T>) {
        if let Node {
            content:
                NodeContent::StochasticNode {
                    ref all_deterministic_children,
                    ..
                },
            ..
        } = self.nodes[i]
        {
            for j in all_deterministic_children {
                if let Node {
                    content: NodeContent::DeterministicNode { .. },
                    ..
                } = self.nodes[*j]
                {
                    self.update_deterministic_value_of(*j, gv);
                }
            }
        }
    }

    pub fn update_all_deterministic_nodes(&self, gv: &GraphVar<T>) {
        for i in 0..self.nodes.len() {
            self.update_deterministic_value_of(i, gv);
        }
    }

    pub fn logpost_all(&self, gv: &GraphVar<T>) -> T {
        let mut result = zero();
        self.update_all_deterministic_nodes(gv);
        for n in 0..self.nodes.len() {
            if let NodeContent::StochasticNode { .. } = self.nodes[n].content {
                result = result + self.logprob(n, gv);
            }
        }
        result
    }

    pub fn set_value_then_update(&self, i: usize, j: usize, x: T, gv: &mut GraphVar<T>) {
        if let Node {
            info:
                BasicNode {
                    ref idx_in_var,
                    ref value_type,
                    ..
                },
            content: NodeContent::StochasticNode { .. },
            ..
        } = self.nodes[i]
        {
            match value_type[j] {
                ValueType::DETERMINISTIC => gv.deterministic_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::FIXED => gv.fixed_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::SAMPLEABLE => gv.sampleable_values[idx_in_var[j]] = x,
            }
            self.update_deterministic_children(i, gv);
        }
    }

    pub fn set_value_no_update(&self, i: usize, j: usize, x: T, gv: &mut GraphVar<T>) {
        if let Node {
            info:
                BasicNode {
                    ref idx_in_var,
                    ref value_type,
                    ..
                },
            content: NodeContent::StochasticNode { .. },
            ..
        } = self.nodes[i]
        {
            match value_type[j] {
                ValueType::DETERMINISTIC => gv.deterministic_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::FIXED => gv.fixed_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::SAMPLEABLE => gv.sampleable_values[idx_in_var[j]] = x,
            }
        }
    }

    //pub fn lambda_adapter<'a>(&'a self)-> Box<Fn(&GraphVar<T>)->T>{
    //    Box::new(|x:&GraphVar<T>|->T{self.logpost_all(x)})
    //}
}

unsafe impl<K, T> std::marker::Sync for Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + std::fmt::Display,
{
}

unsafe impl<K, T> std::marker::Send for Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + std::fmt::Display,
{
}
