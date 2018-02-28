use num_traits::float::Float;
use std::clone::Clone;
use std::marker::Copy;
use std::convert::From;
use super::vec3d::Vec3d;

pub struct SphCoord<T>
where
    T: Float + Copy,
{
    pub pol: T,
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

impl<T> Copy for SphCoord<T>
where
    T: Float + Copy,
{
}

impl<T> SphCoord<T>
where
    T: Float + Copy,
{
    pub fn new(pol: T, az: T) -> SphCoord<T> {
        SphCoord::<T> { pol: pol, az: az }
    }

    pub fn from_vec3d(p: Vec3d<T>) -> SphCoord<T> {
        SphCoord::from_xyz(p.x, p.y, p.z)
    }

    pub fn from_xyz(x: T, y: T, z: T) -> SphCoord<T> {
        let r = (x * x + y * y + z * z).sqrt();
        let az = y.atan2(x);
        let pol = (z / r).acos();
        SphCoord::new(pol, az)
    }

    pub fn vdrdpol(&self)->Vec3d<T>{
        let theta=self.pol;
        let phi=self.az;
        let stheta=theta.sin();
        let ctheta=theta.cos();
        let sphi=phi.sin();
        let cphi=phi.cos();
        Vec3d::new(ctheta*cphi, ctheta*sphi, -stheta)
    }

    pub fn vdrdaz(&self)->Vec3d<T>{
        Vec3d::new(-self.az.sin(), self.az.cos(), T::zero())
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
