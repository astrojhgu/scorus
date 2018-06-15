//! 2-D Euclid vectors

use num_traits::float::Float;
use std::clone::Clone;
use std::marker::Copy;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// 2-D vector
#[derive(Debug)]
pub struct Vec2d<T>
where
    T: Float + Copy,
{
    /// x component
    pub x: T,
    /// y component
    pub y: T,
}

impl<T> Clone for Vec2d<T>
where
    T: Float + Copy,
{
    fn clone(&self) -> Self {
        Vec2d::<T> {
            x: self.x,
            y: self.y,
        }
    }
}

impl<T> Copy for Vec2d<T>
where
    T: Float + Copy,
{
}

impl<T> Vec2d<T>
where
    T: Float + Copy,
{
    /// create a vector from its components
    pub fn new(x: T, y: T) -> Vec2d<T> {
        Vec2d::<T> { x: x, y: y }
    }

    /// calculate the normalized vector, i.e., normalize the length to 1
    pub fn normalized(&self) -> Vec2d<T> {
        *self / (self.length())
    }

    /// inner product
    pub fn dot(&self, rhs: Vec2d<T>) -> T {
        self.x * rhs.x + self.y * rhs.y
    }

    /// calculate the length
    pub fn length(&self) -> T {
        self.norm2().sqrt()
    }

    /// calculate the square of length
    pub fn norm2(&self) -> T {
        self.dot(*self)
    }
}

impl<T> Index<usize> for Vec2d<T>
where
    T: Float + Copy,
{
    type Output = T;

    /// if index=0, the return x, if index=1, return y
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Vec2d<T>
where
    T: Float + Copy,
{
    /// if index=0, the return x, if index=1, return y
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> Add for Vec2d<T>
where
    T: Float + Copy,
{
    type Output = Vec2d<T>;

    fn add(self, rhs: Vec2d<T>) -> Vec2d<T> {
        Vec2d {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T> Sub for Vec2d<T>
where
    T: Float + Copy,
{
    type Output = Vec2d<T>;

    fn sub(self, rhs: Vec2d<T>) -> Vec2d<T> {
        Vec2d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T> Mul<T> for Vec2d<T>
where
    T: Float + Copy,
{
    type Output = Vec2d<T>;

    fn mul(self, rhs: T) -> Vec2d<T> {
        Vec2d {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T> Div<T> for Vec2d<T>
where
    T: Float + Copy,
{
    type Output = Vec2d<T>;

    fn div(self, rhs: T) -> Vec2d<T> {
        Vec2d {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
