extern crate rand;
extern crate std;

use std::ops::IndexMut;
use rand::Rand;
use std::collections::LinkedList;
use num_traits::float::Float;
use num_traits::NumCast;
use num_traits::identities::one;
use num_traits::identities::zero;
use utils::HasLength;
use utils::Resizeable;
use utils::ItemSwapable;

#[derive(Debug)]
pub enum ArmsErrs {
    ResultIsInf,
    ResultIsNan,
    VarOutOfRange,
}


fn eval_log<T>(pd: fn(T) -> T, x: T, scale: T) -> Result<T, ArmsErrs>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    match pd(x) {
        y if y.is_infinite() => Err(ArmsErrs::ResultIsInf),
        y if y.is_nan() => Err(ArmsErrs::ResultIsNan),
        y => Ok(y),
    }
}

pub fn int_exp_y<T>(x: T, p1: (T, T), p2: (T, T)) -> Result<T, ArmsErrs>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    match if x1 == x2 {
        zero()
    } else if y1.is_finite() && y2.is_finite() {
        let k: T = (y2 - y1) / (x2 - x1);
        if k.is_zero() {
            y1.exp() * (x - x1)
        } else {
            match ((k * (x - x1)).exp() - one()) * y1.exp() / k {
                x if x.is_nan() => ((k * (x - x1) + y1).exp() - y1.exp()) / k,
                x => x,
            }
        }
    } else {
        zero()
    } {
        x if x.is_nan() => Err(ArmsErrs::ResultIsNan),
        x => Ok(x),
    }
}


pub fn inv_int_exp_y<T>(z: T, p1: (T, T), p2: (T, T)) -> Result<T, ArmsErrs>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    let k: T = (y2 - y1) / (x2 - x1);
    match if (x1 == x2) && z.is_zero() {
        x1
    } else {
        if k.is_zero() {
            match x1 + z * (-y1).exp() {
                r if r.is_infinite() => x1 + (z.ln() - y1).exp(),
                r => r,
            }
        } else {
            let u: T = one::<T>() + k * z * (-y1).exp();
            if u.is_infinite() {
                x1 + ((k * z).ln() - y1) / k
            } else {
                x1 + (if u <= zero() {
                    (y1.exp() + k * z).ln() - y1
                } else {
                    u.ln()
                }) / k
            }
        }
    } {
        r if r.is_nan() => Err(ArmsErrs::ResultIsNan),
        r => Ok(r),
    }
}


pub struct Section<T>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    _x_l: T,
    _x_u: T,
    _x_i: T,

    _y_l: T,
    _y_u: T,
    _y_i: T,

    _int_exp_y_l: T,
    _int_exp_y_u: T,
    _cum_int_exp_y_l: T,
    _cum_int_exp_y_u: T,
}

impl<T> Section<T>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    fn new() -> Self {
        let nan = Float::nan();
        Section {
            _x_l: nan,
            _x_u: nan,
            _x_i: nan,
            _y_l: nan,
            _y_u: nan,
            _y_i: nan,
            _int_exp_y_l: nan,
            _int_exp_y_u: nan,
            _cum_int_exp_y_l: nan,
            _cum_int_exp_y_u: nan,
        }
    }

    fn x_l(&self) -> T {
        self._x_l
    }

    fn x_u(&self) -> T {
        self._x_u
    }

    fn x_i(&self) -> T {
        self._x_i
    }

    fn y_l(&self) -> T {
        self._y_l
    }

    fn y_u(&self) -> T {
        self._y_u
    }

    fn y_i(&self) -> T {
        self._y_i
    }
    fn int_exp_y_l(&self) -> T {
        self._int_exp_y_l
    }
    fn int_exp_y_u(&self) -> T {
        self._int_exp_y_u
    }
    fn cum_int_exp_y_l(&self) -> T {
        self._cum_int_exp_y_l
    }
    fn cum_int_exp_y_u(&self) -> T {
        self._cum_int_exp_y_u
    }
    fn set_x_l(&mut self, x: T) {
        self._x_l = x;
    }
    fn set_x_u(&mut self, x: T) {
        self._x_u = x;
    }
    fn set_x_i(&mut self, x: T) {
        self._x_i = x;
        if self._x_i < self._x_l {
            self._x_i = self._x_l;
        }
        if self._x_i > self._x_u {
            self._x_i = self._x_u;
        }
    }

    fn set_y_l(&mut self, y: T) {
        self._y_l = y;
    }
    fn set_y_u(&mut self, y: T) {
        self._y_u = y;
    }
    fn set_y_i(&mut self, y: T) {
        self._y_i = y;
    }

    fn set_int_exp_y_l(&mut self, y: T) {
        self._int_exp_y_l = y;
    }
    fn set_int_exp_y_u(&mut self, y: T) {
        self._int_exp_y_u = y;
    }
    fn set_cum_int_exp_y_l(&mut self, y: T) {
        self._cum_int_exp_y_l = y;
    }
    fn set_cum_int_exp_y_u(&mut self, y: T) {
        self._cum_int_exp_y_u = y;
    }
    fn encloses(&self, x: T) -> bool {
        (x >= self._x_l) && (x <= self._x_u)
    }
    fn calc_int_exp_y(self) -> Result<Self, ArmsErrs> {
        let _int_exp_y_l: T = if self._x_i == self._x_l {
            zero()
        } else {
            int_exp_y(self._x_i, (self._x_l, self._y_l), (self._x_i, self._y_i))?
        };
        let _int_exp_y_u: T = if self._x_i == self._x_u {
            zero()
        } else {
            int_exp_y(self._x_u, (self._x_i, self._y_i), (self._x_u, self._y_u))?
        };
        Ok(Section::<T> {
            _int_exp_y_l: _int_exp_y_l,
            _int_exp_y_u: _int_exp_y_u,
            ..self
        })
    }

    fn eval_y(&self, x: T) -> T {
        if x == self._x_l {
            self._y_l
        } else if x == self._x_u {
            self._y_u
        } else if x == self._x_i {
            self._y_i
        } else if x < self._x_i {
            self._y_l + (x - self._x_l) * (self._y_i - self._y_l) / (self._x_i - self._x_l)
        } else {
            self._y_u + (x - self._x_u) * (self._y_i - self._y_u) / (self._x_i - self._x_u)
        }
    }
}


pub fn eval<T>(x: T, section_list: &LinkedList<Section<T>>) -> Result<T, ArmsErrs>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    for i in section_list.iter() {
        if i.encloses(x) {
            return Ok(i.eval_y(x));
        }
    }
    Err(ArmsErrs::VarOutOfRange)
}

pub fn eval_ey<T>(x: T, section_list: &LinkedList<Section<T>>) -> Result<T, ArmsErrs>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    let y = eval(x, section_list)?;
    Ok(y.exp())
}

pub fn solve_intersection<T>(
    s: &Section<T>,
    p1: (T, T),
    p2: (T, T),
    p3: (T, T),
    p4: (T, T),
) -> Result<(T, T), ArmsErrs>
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let (x3, y3) = p3;
    let (x4, y4) = p4;

    let two = one::<T>() + one::<T>();
    match if y2.is_infinite() {
        match (x1, y4 + (x1 - x4) / (x3 - x4) * (y3 - y4)) {
            (x_i, y_i) if y_i < y1 => ((s.x_l() + s.x_u()) / two, (s.y_l() + s.y_u()) / two),
            (x_i, y_i) => (x_i, y_i),
        }
    } else if (y4.is_infinite()) {
        match (x3, y1 + (x3 - x1) / (x2 - x1) * (y2 - y1)) {
            (x_i, y_i) if y_i < y3 => ((s.x_l() + s.x_u()) / two, (s.y_l() + s.y_u()) / two),
            (x_i, y_i) => (x_i, y_i),
        }
    } else {
        let x_i = -((-(x3 - x4) * (x2 * y1 - x1 * y2) + (x1 - x2) * (x4 * y3 - x3 * y4))
            / (-(x3 - x4) * (-y1 + y2) + (x1 - x2) * (-y3 + y4)));
        let y_i = -(x2 * y1 * y3 - x4 * y1 * y3 - x1 * y2 * y3 + x4 * y2 * y3 - x2 * y1 * y4
            + x3 * y1 * y4 + x1 * y2 * y4 - x3 * y2 * y4)
            / (-x3 * y1 + x4 * y1 + x3 * y2 - x4 * y2 + x1 * y3 - x2 * y3 - x1 * y4 + x2 * y4);

        match (x_i, y_i) {
            (x_i, y_i)
                if ((y3 - y1) * (x2 - x1) == (y2 - y1) * (x3 - x1)
                    && (y4 - y1) * (x2 - x1) == (y2 - y1) * (x4 - x1))
                    || (y2 - y1) * (x4 - x3) == (x2 - x1) * (y4 - y3)
                    || y_i.is_infinite() =>
            {
                ((s.x_l() + s.x_u()) / two, (s.y_l() + s.y_u()) / two)
            }
            (x_i, y_i)
                if (x_i <= s.x_l()) || (x_i >= s.x_u())
                    || y_i < (y2 + (x_i - x2) / (x3 - x2) * (y3 - y2)) =>
            {
                ((s.x_l() + s.x_u()) / two, (s.y_l() + s.y_u()) / two)
            }
            (x_i, y_i) => (x_i, y_i),
        }
    } {
        (x_i, y_i) if x_i.is_nan() || y_i.is_nan() => Err(ArmsErrs::ResultIsNan),
        (x_i, y_i) => Ok((x_i, y_i)),
    }
}
