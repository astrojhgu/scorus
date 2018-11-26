use crate::linear_space::LinearSpace;
use crate::utils::HasLen;
use num_traits::float::Float;
use std;
use std::cell::RefCell;
use std::fmt::{Display, Error, Formatter};
use std::ops::{Add, Mul, Sub};

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

impl<'a, T> Add<&'a GraphVar<T>> for &'a GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    type Output = GraphVar<T>;

    fn add(self, rhs: Self) -> GraphVar<T> {
        let mut result = self.clone();
        for i in 0..result.len() {
            result[i] = result[i] + rhs[i];
        }
        result
    }
}

impl<'a, T> Sub<&'a GraphVar<T>> for &'a GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    type Output = GraphVar<T>;

    fn sub(self, rhs: Self) -> GraphVar<T> {
        let mut result = self.clone();
        for i in 0..result.len() {
            result[i] = result[i] - rhs[i];
        }
        result
    }
}

impl<'a, T> Mul<T> for &'a GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    type Output = GraphVar<T>;

    fn mul(self, rhs: T) -> GraphVar<T> {
        let mut result = self.clone();
        for i in 0..result.len() {
            result[i] = result[i] * rhs;
        }
        result
    }
}

impl<T> LinearSpace<T> for GraphVar<T>
where
    T: Float + Sync + Send + std::fmt::Display,
{
    fn dimension(&self) -> usize {
        self.len()
    }
}
