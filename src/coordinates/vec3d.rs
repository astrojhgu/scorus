use num_traits::float::Float;
use std::clone::Clone;
//use std::cmp::{Eq, PartialEq};
use std::convert::From;
use std::marker::Copy;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use super::sphcoord::SphCoord;

#[derive(Debug, PartialEq, Eq)]
pub struct Vec3d<T>
where
    T: Float + Copy,
{
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Clone for Vec3d<T>
where
    T: Float + Copy,
{
    fn clone(&self) -> Self {
        Vec3d::<T> {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl<T> Copy for Vec3d<T>
where
    T: Float + Copy,
{
}

impl<T> Vec3d<T>
where
    T: Float + Copy,
{
    pub fn new(x: T, y: T, z: T) -> Vec3d<T> {
        Vec3d::<T> { x: x, y: y, z: z }
    }

    pub fn normalized(&self) -> Vec3d<T> {
        (*self) / self.length()
    }

    pub fn from_sph_coord(sc: SphCoord<T>) -> Vec3d<T> {
        Vec3d::<T>::from_angle(sc.pol, sc.az)
    }

    pub fn from_angle(pol: T, az: T) -> Vec3d<T> {
        let sp = pol.sin();
        let cp = pol.cos();
        let sa = az.sin();
        let ca = az.cos();
        let z = cp;
        let x = sp * ca;
        let y = sp * sa;
        Vec3d { x: x, y: y, z: z }
    }

    pub fn dot(&self, rhs: Vec3d<T>) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(&self, rhs:Vec3d<T>) ->Vec3d<T>{
        let u1=self.x;
        let u2=self.y;
        let u3=self.z;
        let v1=rhs.x;
        let v2=rhs.y;
        let v3=rhs.z;

        let s1=u2*v3-u3*v2;
        let s2=u3*v1-u1*v3;
        let s3=u1*v2-u2*v1;

        Vec3d::new(s1, s2, s3)
    }

    pub fn length(&self) -> T {
        self.norm2().sqrt()
    }

    pub fn norm2(&self) -> T {
        self.dot(*self)
    }

    pub fn angle_between(&self, rhs: Vec3d<T>) -> T {
        (self.dot(rhs) / (self.length() * rhs.length())).acos()
    }

    pub fn vdrdpol(&self) -> Vec3d<T> {
        SphCoord::from(*self).vdrdpol()
    }

    pub fn vdrdaz(&self) -> Vec3d<T> {
        SphCoord::from(*self).vdrdaz()
    }

    pub fn rotate_about(&self, axis: Vec3d<T>, angle: T) -> Vec3d<T> {
        let axis = axis / axis.length();
        let ux: T = axis.x;
        let uy: T = axis.y;
        let uz: T = axis.z;

        let ca = angle.cos();
        let sa = angle.sin();

        let one = T::one();

        let rm = vec![
            vec![
                ca + ux * ux * (one - ca),
                ux * uy * (one - ca) - uz * sa,
                ux * uz * (one - ca) + uy * sa,
            ],
            vec![
                uy * ux * (one - ca) + uz * sa,
                ca + uy * uy * (one - ca),
                uy * uz * (one - ca) - ux * sa,
            ],
            vec![
                uz * ux * (one - ca) - uy * sa,
                uz * uy * (one - ca) + ux * sa,
                ca + uz * uz * (one - ca),
            ],
        ];

        let mut result = Vec3d::<T>::new(T::zero(), T::zero(), T::zero());
        for i in 0..3 {
            for j in 0..3 {
                result[i] = result[i] + rm[i][j] * self[j];
            }
        }
        result
    }
}

impl<T> From<SphCoord<T>> for Vec3d<T>
where
    T: Float + Copy,
{
    fn from(sc: SphCoord<T>) -> Vec3d<T> {
        Vec3d::from_angle(sc.pol, sc.az)
    }
}

impl<T> Index<usize> for Vec3d<T>
where
    T: Float + Copy,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Vec3d<T>
where
    T: Float + Copy,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> Add for Vec3d<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn add(self, rhs: Vec3d<T>) -> Vec3d<T> {
        Vec3d {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> Sub for Vec3d<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn sub(self, rhs: Vec3d<T>) -> Vec3d<T> {
        Vec3d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Mul<T> for Vec3d<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn mul(self, rhs: T) -> Vec3d<T> {
        Vec3d {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T> Div<T> for Vec3d<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn div(self, rhs: T) -> Vec3d<T> {
        Vec3d {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}
