//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use rand::thread_rng;
use num_traits::Bounded;
use std::fs::File;
use scorus::map_proj::hammer::{iproj, proj};
use scorus::coordinates::{SphCoord, Vec3d};

use scorus::interpolation::sph_nn::Interpolator;
use num_traits::float::{Float, FloatConst};
use std::io::Write;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;

fn main() {
    let mut aa = Interpolator::new();

    /*
    let interp=aa.add_point(SphCoord::from(Vec3d::new(1.0, 0.0, 0.0)), 1.5/2.0)
        .add_point(SphCoord::from(Vec3d::new(-1.0, 0.0, 0.0)), 1.5/2.0)
        .add_point(SphCoord::from(Vec3d::new(0.0, 1.0, 0.0)), 1.5/2.0)
        .add_point(SphCoord::from(Vec3d::new(0.0, -1.0, 0.0)), 1.5/2.0)
        .add_point(SphCoord::from(Vec3d::new(0.0, 0.0, 1.0)), 1.0/2.0)
        .add_point(SphCoord::from(Vec3d::new(0.0, 0.0, -1.0)), 1.0/2.0)
        .done();

    println!("{:?}", SphCoord::from(Vec3d::new(1.0, 0.0, 0.0)));
    println!("{:?}", SphCoord::new(90_f64.to_radians(), 0_f64.to_radians()));

    println!("{:?}", SphCoord::from(Vec3d::new(-1.0, 0.0, 0.0)));
    println!("{:?}", SphCoord::new(90_f64.to_radians(), 180_f64.to_radians()));

    println!("{:?}", SphCoord::from(Vec3d::new(0.0, 1.0, 0.0)));
    println!("{:?}", SphCoord::new(90_f64.to_radians(), 90_f64.to_radians()));

    println!("{:?}", SphCoord::from(Vec3d::new(0.0, -1.0, 0.0)));
    println!("{:?}", SphCoord::new(90_f64.to_radians(), 270_f64.to_radians()));

    println!("{:?}", SphCoord::from(Vec3d::new(0.0, 0.0, 1.0)));
    println!("{:?}", SphCoord::new(0_f64.to_radians(), 0_f64.to_radians()));

    println!("{:?}", SphCoord::from(Vec3d::new(0.0, 0.0, -1.0)));
    println!("{:?}", SphCoord::new(180_f64.to_radians(), 0_f64.to_radians()));
*/

    /*
    let interp=aa.add_point(SphCoord::new(90_f64.to_radians(), 0_f64.to_radians()), 1.5/2.0)
        .add_point(SphCoord::new(90_f64.to_radians(), 180_f64.to_radians()), 1.5/2.0)
        .add_point(SphCoord::new(90_f64.to_radians(), 90_f64.to_radians()), 1.5/2.0)
        .add_point(SphCoord::new(90_f64.to_radians(), -90_f64.to_radians()), 1.5/2.0)
        .add_point(SphCoord::new(0_f64.to_radians(), 0_f64.to_radians()), 1.0/2.0)
        .add_point(SphCoord::new(180_f64.to_radians(), 0_f64.to_radians()), 1.0/2.0)
        .done();
    */
    let mut rng = thread_rng();
    for _i in 0..10000000 {
        let p: SphCoord<f64> = rand_sph(&mut rng);
        aa = aa.add_point(p, p.pol.sin());
    }

    let interp = aa.shuffle(&mut rng).done();
    println!("point inserted");
    let pol_min = 0.0;
    let pol_max = f64::PI();
    let az_min = -f64::PI();
    let az_max = f64::PI();

    let npol = 100;
    let naz = 100;

    //for j in 0..naz
    let j = 54;
    {
        let pol = 90_f64.to_radians();
        let az = (az_max - az_min) / (naz as f64) * (j as f64) + az_min;

        let sph = SphCoord::new(pol, az);

        let ap = interp(&sph, 3);

        //println!("{} {}", az, ap);
    }
    //let n = Vec3d::<f64>::from(sph);
    //let n = n * ap;
    //println!("{} {} {} {} {}", i, j, n.x, n.y, n.z).unwrap();

    //return;

    let mut beam_file = File::create("beam.txt").unwrap();

    writeln!(beam_file, "{}", npol).unwrap();
    writeln!(beam_file, "{}", naz).unwrap();
    for i in 0..npol {
        let pol = (pol_max - pol_min) / (npol as f64) * (i as f64) + pol_min;
        write!(beam_file, "{} ", pol).unwrap();
    }
    writeln!(beam_file, "").unwrap();
    for j in 0..naz {
        let az = (az_max - az_min) / (naz as f64) * (j as f64) + az_min;
        write!(beam_file, "{} ", az).unwrap();
    }
    writeln!(beam_file, "").unwrap();
    for i in 0..npol {
        println!("{}", i);
        let pol = (pol_max - pol_min) / (npol as f64) * (i as f64) + pol_min;
        for j in 0..naz {
            let az = (az_max - az_min) / (naz as f64) * (j as f64) + az_min;

            let sph = SphCoord::new(pol, az);

            let ap = interp(&sph, 3);

            let n = Vec3d::<f64>::from(sph);
            let n = n * ap;
            writeln!(beam_file, "{} {} {} {} {}", i, j, n.x, n.y, n.z).unwrap();
        }
    }
}
