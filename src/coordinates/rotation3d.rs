#![allow(clippy::needless_range_loop)]
//! Rotation of vectors in 3D Euclid space

use num::traits::float::Float;
use std::{
    fmt::{Debug, Formatter},
    ops::{Index, Mul},
};

use super::vec3d::Vec3d;

/// # Rotation matrix
/// Mearly a 3x3 matrix. Inner data should not be accessed directly

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct RotMatrix<T> {
    elements: [[T; 3]; 3],
}

impl<T> Debug for RotMatrix<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("RotMat:[\n"))?;
        for i in 0..3 {
            for j in 0..3 {
                f.write_str(&format!("{:?} ", self.elements[i][j]))?;
            }
            f.write_str("\n")?;
        }
        f.write_str("]\n")?;
        Ok(())
    }
}

impl<T> Index<[usize; 2]> for RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &T {
        &self.elements[index[0]][index[1]]
    }
}

impl<T> Index<usize> for RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = [T; 3];

    fn index(&self, index: usize) -> &[T; 3] {
        &self.elements[index]
    }
}

impl<T> Mul for RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = RotMatrix<T>;

    fn mul(self, rhs: RotMatrix<T>) -> RotMatrix<T> {
        let mut data = [[T::zero(); 3]; 3];
        let ndim = 3;
        for i in 0..ndim {
            for j in 0..ndim {
                let mut y = T::zero();
                for k in 0..ndim {
                    y = y + self.elements[i][k] * rhs.elements[k][j];
                }
                data[i][j] = y;
            }
        }
        RotMatrix { elements: data }
    }
}

impl<T> Mul for &RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = RotMatrix<T>;

    fn mul(self, rhs: &RotMatrix<T>) -> RotMatrix<T> {
        let mut data = [[T::zero(); 3]; 3];
        let ndim = 3;
        for i in 0..ndim {
            for j in 0..ndim {
                let mut y = T::zero();
                for k in 0..ndim {
                    y = y + self.elements[i][k] * rhs.elements[k][j];
                }
                data[i][j] = y;
            }
        }
        RotMatrix { elements: data }
    }
}

impl<T> Mul<&RotMatrix<T>> for RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = RotMatrix<T>;

    fn mul(self, rhs: &RotMatrix<T>) -> RotMatrix<T> {
        let mut data = [[T::zero(); 3]; 3];
        let ndim = 3;
        for i in 0..ndim {
            for j in 0..ndim {
                let mut y = T::zero();
                for k in 0..ndim {
                    y = y + self.elements[i][k] * rhs.elements[k][j];
                }
                data[i][j] = y;
            }
        }
        RotMatrix { elements: data }
    }
}

impl<T> Mul<RotMatrix<T>> for &RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = RotMatrix<T>;

    fn mul(self, rhs: RotMatrix<T>) -> RotMatrix<T> {
        let mut data = [[T::zero(); 3]; 3];
        let ndim = 3;
        for i in 0..ndim {
            for j in 0..ndim {
                let mut y = T::zero();
                for k in 0..ndim {
                    y = y + self.elements[i][k] * rhs.elements[k][j];
                }
                data[i][j] = y;
            }
        }
        RotMatrix { elements: data }
    }
}

impl<T> Mul<&Vec3d<T>> for &RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn mul(self, rhs: &Vec3d<T>) -> Vec3d<T> {
        let mut result = Vec3d::new(T::zero(), T::zero(), T::zero());
        let ndim = 3;
        for i in 0..ndim {
            for k in 0..ndim {
                result[i] = result[i] + self[i][k] * rhs[k];
            }
        }
        result
    }
}

impl<T> Mul<Vec3d<T>> for RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn mul(self, rhs: Vec3d<T>) -> Vec3d<T> {
        let mut result = Vec3d::new(T::zero(), T::zero(), T::zero());
        let ndim = 3;
        for i in 0..ndim {
            for k in 0..ndim {
                result[i] = result[i] + self[i][k] * rhs[k];
            }
        }
        result
    }
}

impl<T> Mul<Vec3d<T>> for &RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn mul(self, rhs: Vec3d<T>) -> Vec3d<T> {
        let mut result = Vec3d::new(T::zero(), T::zero(), T::zero());
        let ndim = 3;
        for i in 0..ndim {
            for k in 0..ndim {
                result[i] = result[i] + self[i][k] * rhs[k];
            }
        }
        result
    }
}

impl<T> Mul<&Vec3d<T>> for RotMatrix<T>
where
    T: Float + Copy,
{
    type Output = Vec3d<T>;

    fn mul(self, rhs: &Vec3d<T>) -> Vec3d<T> {
        let mut result = Vec3d::new(T::zero(), T::zero(), T::zero());
        let ndim = 3;
        for i in 0..ndim {
            for k in 0..ndim {
                result[i] = result[i] + self[i][k] * rhs[k];
            }
        }
        result
    }
}

impl<T> RotMatrix<T>
where
    T: Float + Copy,
{
    /// A[i,j]->A[j,i]
    pub fn transpose(&self) -> RotMatrix<T> {
        let mut data = [[T::zero(); 3]; 3];
        let ndim = 3;
        for i in 0..ndim {
            for j in 0..ndim {
                data[i][j] = self[j][i];
            }
        }
        RotMatrix { elements: data }
    }

    pub fn new(elements: [[T; 3]; 3]) -> RotMatrix<T> {
        RotMatrix { elements }
    }

    /// Because of the property of rotation matrix, inv is simply obtained through transpose.
    pub fn inv(&self) -> RotMatrix<T> {
        self.transpose()
    }

    /// Compose a rotation matrix by denoting the rotation about an arbitrary vector as the axis and
    /// by certain angle
    pub fn about_axis_by_angle(axis: &Vec3d<T>, theta: T) -> RotMatrix<T> {
        let naxis = axis.normalized();
        let ux = naxis.x;
        let uy = naxis.y;
        let uz = naxis.z;
        let mut data = [[T::zero(); 3]; 3];
        let one = T::one();
        let costheta = theta.cos();
        let sintheta = theta.sin();
        data[0][0] = costheta + ux * ux * (one - costheta);
        data[1][1] = costheta + uy * uy * (one - costheta);
        data[2][2] = costheta + uz * uz * (one - costheta);

        data[0][1] = ux * uy * (one - costheta) - uz * sintheta;
        data[1][0] = uy * ux * (one - costheta) + uz * sintheta;

        data[0][2] = ux * uz * (one - costheta) + uy * sintheta;
        data[2][0] = uz * ux * (one - costheta) - uy * sintheta;

        data[1][2] = uy * uz * (one - costheta) - ux * sintheta;
        data[2][1] = uz * uy * (one - costheta) + ux * sintheta;

        RotMatrix { elements: data }
    }
}
