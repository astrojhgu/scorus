use num::{
    traits::{
        Float
        , FloatConst
    }
    , complex::{
        Complex
    }
};

use std::{
    fmt::{
        Debug
    }
    , ops::{
        Mul
        , Div
    }
};

use crate::{
    coordinates::{
        rotation3d::{
            RotMatrix
        }, SphCoord, Vec3d
    }
    , healpix::{
        interp::{
            natural_interp_ring
        }
        , pix::{
            pix2vec_ring
        }
        , utils::{
            npix2nside
        }
    }
};

pub fn rotate_ring<T, W>(input: &[T], mat: &RotMatrix<W>)->Vec<T>
where 
T: Float + FloatConst + Debug + Mul<W, Output=T> + Div<W, Output=T>,
W: Float + FloatConst + Debug
{
    let mat=mat.transpose();
    let npix=input.len();
    let nside=npix2nside(npix);

    let pix_center:Vec<_>=(0..npix).map(|i| pix2vec_ring(nside, i)).collect();

    let pix_center_rot:Vec<_>=pix_center.iter().map(|&x|{mat*x}).collect();
    
    let dir:Vec<_>=pix_center_rot.iter().map(|&x| SphCoord::from_vec3d(x)).collect();

    dir.iter().map(|&d| natural_interp_ring(nside, input, d)).collect()
    //(0..npix).map(|i| natural_interp_ring(nside, input, SphCoord::from_vec3d(mat*pix2vec_ring(nside, i)))).collect()
}

pub fn angle_ref<T>(v: &Vec3d<T>, mat: &RotMatrix<T>)->T
where T: Float{
    let vp=mat*v;
    let north_pole=mat*Vec3d::new(T::zero(), T::zero(), T::one());
    let sinalpha = north_pole[0] * vp[1] - north_pole[1] * vp[0];
    let cosalpha = north_pole[2] - vp[2] * north_pole.dot(vp);
    T::atan2(sinalpha, cosalpha)
}

pub fn rotate_ring_pol<T, W>(t:&[T], q:&[T], u:&[T], mat: &RotMatrix<W>)->(Vec<T>, Vec<T>, Vec<T>)
where 
    T: Float + FloatConst + Debug + Mul<W, Output=T> + Div<W, Output=T>,
    W: Float + FloatConst + Debug
{
    let mat_inv=mat.transpose();
    let two=T::one()+T::one();
    let npix=t.len();
    assert_eq!(npix, q.len());
    assert_eq!(npix, u.len());
    let nside=npix2nside(npix);
    let rotated_center_dir:Vec<_>=(0..npix).map(|i| mat_inv*pix2vec_ring(nside, i)).collect();

    eprintln!("{:?}", rotated_center_dir[0]);


    let rotated_angle:Vec<_>=rotated_center_dir.iter().map(|d| SphCoord::from_vec3d(*d)).collect();
    let t_rotated:Vec<_>=rotated_angle.iter().map(|a|{natural_interp_ring(nside, t, *a)}).collect();
    let mut q_rotated:Vec<_>=rotated_angle.iter().map(|a|{natural_interp_ring(nside, q, *a)}).collect();
    let mut u_rotated:Vec<_>=rotated_angle.iter().map(|a|{natural_interp_ring(nside, u, *a)}).collect();

    let l_map:Vec<_>=rotated_center_dir.iter().zip(q_rotated.iter().zip(u_rotated.iter())).map(|(dir1, (&q1, &u1))|{
        Complex::<T>::new(q1, u1)*Complex::<T>::from_polar(T::one(), two*angle_ref(dir1, mat))
    }).collect();
    q_rotated.iter_mut().zip(u_rotated.iter_mut().zip(l_map.iter())).for_each(|(q, (u, &l))|{
        *q=l.re;
        *u=l.im;
    });
    (t_rotated, q_rotated, u_rotated)
}

pub fn get_euler_matrix<T>(a1: T, a2: T, a3: T)->RotMatrix<T>
where T: Float+FloatConst{
    let c1=a1.cos();
    let s1=a1.sin();
    let c2=a2.cos();
    let s2=a2.sin();
    let c3=a3.cos();
    let s3=a3.sin();
    let m1=RotMatrix::<T>::new([
        [c1, -s1, T::zero()],
        [s1, c1, T::zero()],
        [T::zero(),T::zero(),T::one()]
        ]);
    let m2=RotMatrix::<T>::new([
        [T::one(), T::zero(), T::zero()],
        [T::zero(), c2, -s2],
        [T::zero(), s2,  c2]
        ]);
    let m3=RotMatrix::<T>::new([
        [c3, -s3, T::zero()],
        [s3, c3, T::zero()],
        [T::zero(),T::zero(),T::one()]
        ]);

    m3.transpose()*(m2.transpose()*m1.transpose())
}

pub fn get_euler_matrix_deg<T>(a1: T, a2: T, a3: T)->RotMatrix<T>
where T:Float+FloatConst
{
    get_euler_matrix(a1.to_radians(), a2.to_radians(), a3.to_radians())
}

pub fn get_rotation_matrix<T>(a1: T, a2: T, a3: T)->RotMatrix<T>
where T:Float+FloatConst{
    get_euler_matrix(a1, T::zero()-a2, a3)
}

pub fn get_rotation_matrix_deg<T>(a1: T, a2: T, a3: T)->RotMatrix<T>
where T:Float+FloatConst{
    get_euler_matrix_deg(a1, T::zero()-a2, a3)
}
