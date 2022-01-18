use super::graph::Graph;
use super::node::NodeContent::{DeterministicNode, StochasticNode};
use num::traits::Float;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};

use std::fmt::Debug;
use std::fmt::Display;
impl<K, T> Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone + Display + Debug,
    T: Float + Sync + SampleUniform + Send + Display + Debug,
    Standard: Distribution<T>,
{
    pub fn dump_topology(&self, w: &mut dyn std::io::Write) {
        let topology = self.topology();
        let mut edges = std::collections::HashMap::<String, Vec<String>>::new();
        let mut nodes = std::collections::HashSet::<String>::new();
        for (k, v) in topology {
            let kname = format!("\"{}\"", k);
            let plist: Vec<_> = v.into_iter().map(|t| format!("\"{}\"", t.0)).collect();

            let mut node_decl = kname.clone();

            match self.get_node(&k).content {
                DeterministicNode { .. } => {
                    node_decl += "[shape=diamond, style=rounded]";
                }
                StochasticNode {
                    ref is_observed, ..
                } => {
                    if is_observed.iter().any(|&x| x) {
                        node_decl += "[shape=box]";
                    } else {
                        node_decl += "[shape=circle]";
                    }
                }
            }

            nodes.insert(node_decl);
            edges.insert(kname, plist);
        }

        writeln!(w, "digraph G").unwrap();
        writeln!(w, "{{").unwrap();
        for n in nodes {
            writeln!(w, "{}", n).unwrap();
        }
        for (k, v) in edges {
            if !v.is_empty() {
                for j in v {
                    writeln!(w, "{} -> {}", j, k).unwrap();
                }
            }
        }

        writeln!(w, "}}").unwrap();
    }
}
