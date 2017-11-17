extern crate rand;
extern crate scoped_threadpool;
extern crate std;
use scoped_threadpool::Pool;


use utils::draw_z;
use utils::HasLength;
use utils::Resizeable;
use utils::ItemSwapable;
use std::sync::Mutex;
use utils::scale_vec;
use utils::swap_walkers;
use std::ops::IndexMut;
use num_traits::float::Float;
use num_traits::NumCast;
use num_traits::identities::one;
use num_traits::identities::zero;


pub fn sample<T, U, V, W, X>(
    flogprob: fn(&V) -> T,
    ensemble_logprob: &(W, X),
    mut rng: &mut U,
    beta_list: &X,
    perform_swap: bool,
    a: T,
    nthread: usize,
) -> (W, X)
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
    U: rand::Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLength + std::marker::Sync + std::marker::Send,
    W: Clone
        + IndexMut<usize, Output = V>
        + HasLength
        + std::marker::Sync
        + std::marker::Send
        + Drop
        + ItemSwapable,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLength
        + std::marker::Sync
        + Resizeable
        + std::marker::Send
        + Drop
        + ItemSwapable,
{
    let (mut new_ensemble, mut new_logprob) =
        swap_walkers(ensemble_logprob, &mut rng, &beta_list, perform_swap);

    let nbeta = beta_list.length();
    let nwalkers = new_ensemble.length() / nbeta;

    if nwalkers % 2 != 0 {
        panic!(format!("nwalker={} is not even!", nwalkers));
    }
    let ndims: T = NumCast::from(new_ensemble[0].length()).unwrap();
    let half_nwalkers = nwalkers / 2;
    let unit: T = one();

    let mut walker_group: Vec<Vec<Vec<usize>>> = Vec::new();
    walker_group.resize(nbeta, vec![Vec::new(), Vec::new()]);

    let mut walker_group_id: Vec<Vec<usize>> = Vec::new();
    walker_group_id.resize(nbeta, Vec::new());
    let mut rvec: Vec<Vec<T>> = Vec::new();
    rvec.resize(nbeta, Vec::new());
    let mut jvec: Vec<Vec<usize>> = Vec::new();
    jvec.resize(nbeta, Vec::new());
    let mut zvec: Vec<Vec<T>> = Vec::new();
    zvec.resize(nbeta, Vec::new());


    for i in 0..nbeta {
        //println!("ibeta={}", i);
        walker_group[i][0].reserve(half_nwalkers);
        walker_group[i][1].reserve(half_nwalkers);
        walker_group_id[i].reserve(nwalkers);
        rvec[i].reserve(nwalkers);
        jvec[i].reserve(nwalkers);
        zvec[i].reserve(nwalkers);

        for j in 0..nwalkers {
            let mut gid: usize = rng.gen_range(0, 2);

            if walker_group[i][gid].len() == half_nwalkers {
                gid = 1 - gid;
            }
            walker_group[i][gid].push(j);
            walker_group_id[i].push(gid);
            rvec[i].push(rng.gen_range(zero(), one()));
            jvec[i].push(rng.gen_range(0, half_nwalkers));
            zvec[i].push(draw_z(rng, a));
        }
        for j in 0..half_nwalkers {
            //println!("{} {}", walker_group[i][0][j], walker_group[i][1][j]);
        }
        for j in 0..nwalkers {
            //println!("{}", walker_group_id[i][j]);
        }
    }

    let atomic_k = Mutex::new(0);
    let lp_cached = new_logprob.length() == nwalkers * nbeta;
    //println!("cached:{}", lp_cached);
    if !lp_cached {
        new_logprob.resize(nwalkers * nbeta);
    }
    let new_ensemble = Mutex::new(new_ensemble);
    let new_logprob = Mutex::new(new_logprob);

    let create_task = || {
        let atomic_k = &atomic_k;
        let new_ensemble = &new_ensemble;
        let new_logprob = &new_logprob;
        let ensemble = &ensemble_logprob.0;
        let cached_logprob = &ensemble_logprob.1;
        let zvec = &zvec;
        let walker_group = &walker_group;
        let walker_group_id = &walker_group_id;
        let jvec = &jvec;
        let rvec = &rvec;
        let beta_list = &beta_list;
        let task = move || loop {
            let n: usize;
            {
                let mut k1 = atomic_k.lock().unwrap();
                n = *k1;
                *k1 += 1;
            }
            if n >= nwalkers * nbeta {
                break;
            }

            let ibeta = n / nwalkers; //i in beta list
            let k = n - ibeta * nwalkers; //i in each beta
                                          //println!("{} {} {} {}", n, ibeta, k, ibeta*nwalkers+k);

            let lp_last_y = match lp_cached {
                false => flogprob(&ensemble[ibeta * nwalkers + k]),
                _ => cached_logprob[ibeta * nwalkers + k],
            };
            if lp_last_y.is_infinite() || lp_last_y.is_nan() {
                panic!("Error, old lpy cannot be inf or nan")
            }

            let i = walker_group_id[ibeta][k];
            let j = jvec[ibeta][k];
            let ni = 1 - i;
            let z = zvec[ibeta][k];
            let r = rvec[ibeta][k];
            let new_y = scale_vec(
                &ensemble[ibeta * nwalkers + k],
                &ensemble[ibeta * nwalkers + walker_group[ibeta][ni][j]],
                z,
            );
            let lp_y = flogprob(&new_y);
            let delta_lp = lp_y - lp_last_y;
            let beta = beta_list[ibeta];

            let q = ((ndims - unit) * (z.ln()) + delta_lp * beta).exp();
            //println!("{} {} {} {} {} {} ",ibeta, k, lp_y, lp_last_y, z, j);
            {
                let mut yy = new_ensemble.lock().unwrap();
                let mut lpyy = new_logprob.lock().unwrap();
                if r <= q || !lp_cached {
                    //println!("{} {} {} {}", n, ibeta, k, ibeta*nwalkers+k);
                    yy[ibeta * nwalkers + k] = new_y;
                    lpyy[ibeta * nwalkers + k] = lp_y;
                }
            }
        };
        task
    };

    if nthread > 1 {
        let mut pool = Pool::new(nthread as u32);
        pool.scoped(|scope| {
            for _ in 0..nthread {
                scope.execute(create_task());
            }
        });
    } else {
        let task = create_task();
        task();
    }



    let new_ensemble = (&new_ensemble).lock().unwrap();
    let new_logprob = (&new_logprob).lock().unwrap();

    (new_ensemble.clone(), new_logprob.clone())
}
