//extern crate std;
use num_traits::float::Float;
use std;
use std::rc::Rc;
use super::graph_var::GraphVar;

pub enum ValueType {
    FIXED,
    DETERMINISTIC,
    SAMPLEABLE,
}

pub struct BasicNode {
    pub parents: Vec<(usize, usize)>,
    pub children: Vec<usize>,
    pub idx_in_var: Vec<usize>,
    pub value_type: Vec<ValueType>,
    pub ndim_input: usize,
    pub ndim_output: usize,
}

pub enum NodeContent<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    StochasticNode {
        all_stochastic_children: Vec<usize>,
        all_deterministic_children: Vec<usize>,
        is_observed: Vec<bool>,
        values: Vec<T>,
        logprob: Box<Fn(&[T], &[T]) -> T>,
    },

    DeterministicNode {
        calc: Box<Fn(&[T]) -> Vec<T>>,
    },
}

pub struct Node<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    pub info: BasicNode,
    pub content: NodeContent<T>,
}

impl<T> Node<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    pub fn get_children(&self) -> &Vec<usize> {
        &self.info.children
    }
}

impl<T> std::fmt::Display for Node<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Parents:\n");

        for &(i, j) in &self.info.parents {
            write!(f, "{} - {} ", i, j)?;
        }
        write!(f, "\nChildren:\n")?;
        for &i in &self.info.children {
            write!(f, "{} ", i)?
        }
        write!(f, "\nidx_in_var:\n")?;
        for &i in &self.info.idx_in_var {
            write!(f, "{} ", i);
        }
        write!(f, "\nninput:{}\n", self.info.ndim_input)?;
        write!(f, "\nnoutput:{}\n", self.info.ndim_output)?;

        match self.content {
            NodeContent::StochasticNode {
                ref all_stochastic_children,
                ref all_deterministic_children,
                ref is_observed,
                ref values,
                ..
            } => {
                write!(f, "\nall stochastic children\n")?;
                for i in all_stochastic_children {
                    write!(f, "{} ", i)?
                }
                write!(f, "\nalldeterministic_children\n");
                for i in all_deterministic_children {
                    write!(f, "{} ", i)?
                }
                write!(f, "\nobserved:\n")?;
                for i in is_observed {
                    write!(f, "{} ", i)?;
                }
                write!(f, "\ninitial values:\n")?;
                for v in values {
                    write!(f, "{} ", v)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
