#![allow(clippy::needless_range_loop)]
use super::traits::{IndexableLinearSpace, InnerProdSpace, LinearSpace, PDInnerProdSpace};
use crate::utils::HasLen;
use num_traits::Float;
use std::convert::{AsMut, AsRef};
use std::ops::{Add, Mul, Sub};
use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct LsVec<T, V>(pub V)
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized;

impl<T, V> LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T, V> Add<&'a LsVec<T, V>> for &'a LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = LsVec<T, V>;
    fn add(self, rhs: Self) -> LsVec<T, V> {
        let mut result = self.0.clone();
        for i in 0..result.len() {
            result[i] = result[i] + rhs.0[i];
        }
        LsVec(result)
    }
}

impl<'a, T, V> Sub<&'a LsVec<T, V>> for &'a LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = LsVec<T, V>;
    fn sub(self, rhs: Self) -> LsVec<T, V> {
        let mut result = self.0.clone();
        for i in 0..result.len() {
            result[i] = result[i] - rhs.0[i];
        }
        LsVec(result)
    }
}

impl<'a, T, V> Mul<T> for &'a LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = LsVec<T, V>;
    fn mul(self, rhs: T) -> LsVec<T, V> {
        let mut result = self.0.clone();
        for i in 0..result.len() {
            result[i] = result[i] * rhs;
        }
        LsVec(result)
    }
}

impl<T, V> Index<usize> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.0.index(index)
    }
}

impl<T, V> IndexMut<usize> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.0.index_mut(index)
    }
}

impl<T, V> LinearSpace<T> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    fn dimension(&self) -> usize {
        self.0.len()
    }
}

impl<T, V> IndexableLinearSpace<T> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
}

impl<T, V> InnerProdSpace<T> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
    fn dot(&self, rhs: &Self) -> T {
        let ndim = self.dimension();
        (0..ndim)
            .map(|i| self.0[i] * rhs.0[i])
            .fold(T::zero(), |a, b| a + b)
    }
}

impl<T, V> PDInnerProdSpace<T> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
{
}

impl<T, V> AsRef<[T]> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized + AsRef<[T]>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T, V> AsMut<[T]> for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized + AsMut<[T]>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T, V> Deref for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized + Deref<Target = [T]>,
{
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T, V> DerefMut for LsVec<T, V>
where
    T: Float,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized + Deref<Target = [T]> + DerefMut,
{
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}
