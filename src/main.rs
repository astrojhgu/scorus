//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use scorus::healpix::{pix2ang_ring, pix2vec_ring, nside2npix, get_interpol_ring};
use scorus::sphcoord::SphCoord;
fn main() {
    let nside=64_usize;
    let npix=nside2npix(nside);
    let (a,b)=get_interpol_ring(nside, SphCoord::new(1.0, 0.5));
    for i in &a{
        let x=pix2ang_ring::<f64>(nside, *i);
        println!("{} {}", x.pol, x.az);
    }
}
