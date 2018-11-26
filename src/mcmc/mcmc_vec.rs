use crate::linear_space::LinearSpace;
use crate::utils::HasLen;
use num_traits::Float;
use std::ops::{Add, Mul, Sub};
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct McmcVec<T, V>(pub V)
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized;

impl<'a, T, V> Add<&'a McmcVec<T, V>> for &'a McmcVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = McmcVec<T, V>;
    fn add(self, rhs: Self) -> McmcVec<T, V> {
        let mut result = self.0.clone();
        for i in 0..result.len() {
            result[i] = result[i] + rhs.0[i];
        }
        McmcVec(result)
    }
}

impl<'a, T, V> Sub<&'a McmcVec<T, V>> for &'a McmcVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = McmcVec<T, V>;
    fn sub(self, rhs: Self) -> McmcVec<T, V> {
        let mut result = self.0.clone();
        for i in 0..result.len() {
            result[i] = result[i] - rhs.0[i];
        }
        McmcVec(result)
    }
}

impl<'a, T, V> Mul<T> for &'a McmcVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = McmcVec<T, V>;
    fn mul(self, rhs: T) -> McmcVec<T, V> {
        let mut result = self.0.clone();
        for i in 0..result.len() {
            result[i] = result[i] * rhs;
        }
        McmcVec(result)
    }
}

impl<T, V> Index<usize> for McmcVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.0.index(index)
    }
}

impl<T, V> IndexMut<usize> for McmcVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.0.index_mut(index)
    }
}

impl<T, V> LinearSpace<T> for McmcVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    fn dimension(&self) -> usize {
        self.0.len()
    }
}
