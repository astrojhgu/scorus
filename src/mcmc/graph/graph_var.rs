use super::super::super::utils::HasLen;
use num_traits::float::Float;
use std;
use std::cell::RefCell;
use std::fmt::{Display, Error, Formatter};

pub struct GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    pub fixed_values: RefCell<Vec<T>>,
    pub deterministic_values: RefCell<Vec<T>>,
    pub sampleable_values: Vec<T>,
}

impl<T> Clone for GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    fn clone(&self) -> GraphVar<T> {
        GraphVar {
            fixed_values: self.fixed_values.clone(),
            deterministic_values: self.deterministic_values.clone(),
            sampleable_values: self.sampleable_values.clone(),
        }
    }
}

unsafe impl<T> std::marker::Sync for GraphVar<T> where T: Float + Sync + Send + std::fmt::Display {}

impl<T> Display for GraphVar<T>
where
    T: Float + Send + Sync + Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "Fixed values:\n")?;
        for v in &*self.fixed_values.borrow() {
            write!(f, "{} ", *v)?;
        }
        write!(f, "\nDeterministic values:\n")?;
        for v in &*(self.deterministic_values.borrow()) {
            write!(f, "{} ", v)?;
        }
        write!(f, "\n Sampleable values:\n")?;
        for v in &self.sampleable_values {
            write!(f, "{} ", *v)?;
        }
        Ok(())
    }
}

impl<T> HasLen for GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    fn len(&self) -> usize {
        Vec::len(&self.sampleable_values)
    }
}

impl<T> std::ops::Index<usize> for GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.sampleable_values[index]
    }
}

impl<T> std::ops::IndexMut<usize> for GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.sampleable_values[index]
    }
}
