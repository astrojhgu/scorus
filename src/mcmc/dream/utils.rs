#![allow(clippy::comparison_chain)]
use num_traits::Float;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

pub fn multinomial<T, U>(logp: &[T], rng: &mut U) -> usize
where
    T: Float + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    U: Rng,
{
    let mut max_lp = -T::infinity();
    for &lp in logp {
        max_lp = max_lp.max(lp);
    }

    let mut p_idx_pair: Vec<_> = logp
        .iter()
        .enumerate()
        .map(|(i, &lp)| ((lp - max_lp).exp(), i))
        .collect();

    p_idx_pair.sort_by(|a, b| {
        if a.0 > b.0 {
            std::cmp::Ordering::Less
        } else if a.0 < b.0 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    let cum: Vec<_> = p_idx_pair
        .iter()
        .scan(T::zero(), |st, &x| {
            *st = *st + x.0;
            Some((*st, x.1))
        })
        .collect();

    let u = rng.gen_range(T::zero(), cum[cum.len() - 1].0);
    for i in &cum {
        if u < i.0 {
            return i.1;
        }
    }
    cum[cum.len() - 1].1
}

pub fn calc_gamma<T>(delta: usize, dim: usize) -> T
where
    T: Float + std::fmt::Debug,
{
    let two = T::one() + T::one();
    T::from(2.38).unwrap() / (two * T::from(delta * dim).unwrap()).sqrt()
}

pub fn replace_flag<T, R>(ndims: usize, cr: T, rng: &mut R) -> (Vec<bool>, usize)
where
    T: Float + std::fmt::Debug + SampleUniform,
    R: Rng,
{
    let result: Vec<_> = (0..ndims)
        .map(|_| rng.gen_range(T::zero(), T::one()) > (T::one() - cr))
        .collect();
    let dprime = result.iter().filter(|&&x| x).count();
    (result, dprime)
}
