//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use scorus::map_proj::hammer::{iproj, proj};
use scorus::coordinates::SphCoord;
fn main() {
    let aa = proj(SphCoord::new(86_f64.to_radians(), 85_f64.to_radians()));
    println!("{:?}", aa);

    let bb = iproj(aa).unwrap();
    println!("{} {}", bb.pol.to_degrees(), bb.az.to_degrees());
}
