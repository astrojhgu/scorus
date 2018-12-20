//! Spherical coordinates

use num_traits::float::Float;
use std::clone::Clone;
//use std::cmp::Eq;
use super::vec3d::Vec3d;
use std::convert::From;
use std::marker::Copy;

/// A point on a S2 sphere
#[derive(Debug, Eq, PartialEq)]
pub struct SphCoord<T>
where
    T: Float + Copy,
{
    /// polar angle
    pub pol: T,
    /// azimuth angle
    pub az: T,
}

impl<T> Clone for SphCoord<T>
where
    T: Float + Copy,
{
    fn clone(&self) -> Self {
        SphCoord::<T> {
            pol: self.pol,
            az: self.az,
        }
    }
}

impl<T> Copy for SphCoord<T> where T: Float + Copy {}

impl<T> SphCoord<T>
where
    T: Float + Copy,
{
    /// Create a SphCoord from pol and az
    pub fn new(pol: T, az: T) -> SphCoord<T> {
        SphCoord::<T> { pol, az }
    }

    /// Create a SphCoord from a 3D vector by extracting its angular component
    pub fn from_vec3d(p: Vec3d<T>) -> SphCoord<T> {
        SphCoord::from_xyz(p.x, p.y, p.z)
    }

    /// Create a SphCoord from a vector with its components of x, y, z.
    pub fn from_xyz(x: T, y: T, z: T) -> SphCoord<T> {
        let r = (x * x + y * y + z * z).sqrt();
        let az = y.atan2(x);
        let pol = (z / r).acos();
        SphCoord::new(pol, az)
    }

    /// polar angle unit vec
    pub fn vdpol(&self) -> Vec3d<T> {
        let theta = self.pol;
        let phi = self.az;
        let stheta = theta.sin();
        let ctheta = theta.cos();
        let sphi = phi.sin();
        let cphi = phi.cos();
        Vec3d::new(ctheta * cphi, ctheta * sphi, -stheta)
    }

    /// azimuth angle unit vec
    pub fn vdaz(&self) -> Vec3d<T> {
        Vec3d::new(-self.az.sin(), self.az.cos(), T::zero())
    }

    /// vector between two S2 points, i.e., the length of the arc connecting two points
    pub fn angle_between(&self, another: SphCoord<T>) -> T {
        let vec1 = Vec3d::from_sph_coord(*self);
        let vec2 = Vec3d::from_sph_coord(another);
        vec1.dot(vec2).acos()
    }
}

impl<T> From<Vec3d<T>> for SphCoord<T>
where
    T: Float + Copy,
{
    fn from(p: Vec3d<T>) -> SphCoord<T> {
        SphCoord::from_xyz(p.x, p.y, p.z)
    }
}
