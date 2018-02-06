use std;
use std::ops::IndexMut;
//use rand::Rand;
use std::collections::VecDeque;
use num_traits::float::Float;
use num_traits::NumCast;
use num_traits::identities::one;
use num_traits::identities::zero;

use rand::{Rng,Rand};
use rand::distributions::range::SampleRange;


use super::super::utils::HasLength;
//use utils::Resizeable;
//use utils::ItemSwapable;

#[derive(Debug)]
pub struct Section<T>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    pub _x_l: T,
    pub _y_l: T,

    pub _x_u: T,
    pub _y_u: T,

    pub _x_i: Option<T>,
    pub _y_i: Option<T>,

    pub _int_exp_y_l: Option<T>,
    pub _int_exp_y_u: Option<T>,

    pub _cum_int_exp_y_l: Option<T>,
    pub _cum_int_exp_y_u: Option<T>,
}

#[derive(Debug)]
pub enum ArmsErr<T>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    LogProbIsInf(String, T),
    LogProbIsNan(String, T),
    SectionIntIsInf(String, T, (T, T), (T, T)),
    SectionIntIsNan(String, T, (T, T), (T, T)),
    InvIntErr(String, T, (T, T), (T, T)),
    VarOutOfRange(String, T),
    SolveInterSectionErr(String, Section<T>, (T, T), (T, T), (T, T), (T, T)),
    InvalidNeighbor(String),
    IllConditionedDistribution(String, VecDeque<Section<T>>),
    TooFewInitPoints(String),
    DataNotInOrder(String),
}

fn fmin<T>(x: T, y: T) -> T
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    if x > y {
        y
    } else {
        x
    }
}

fn fmax<T>(x: T, y: T) -> T
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    if x < y {
        y
    } else {
        x
    }
}

fn eval_log<T, F>(pd: &F, x: T, scale: T) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
{
    match pd(x) - scale {
        y if y.is_infinite() => Err(ArmsErr::LogProbIsInf(format!("Error@{}", line!()), x)),
        y if y.is_nan() => {
            //panic!("a");
            Err(ArmsErr::LogProbIsNan(format!("Error@{}", line!()), x))
        }
        y => Ok(y),
    }
}

pub fn int_exp_y<T>(x: T, p1: (T, T), p2: (T, T)) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
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
                inty if inty.is_nan() => ((k * (x - x1) + y1).exp() - y1.exp()) / k,
                inty => inty,
            }
        }
    } else {
        zero()
    } {
        inty if inty.is_nan() => {
            println!("{} {} {} {} {}", x, x1, y1, x2, y2);
            //panic!("b");
            Err(ArmsErr::SectionIntIsNan(
                format!("Error@{}", line!()),
                x,
                p1,
                p2,
            ))
        }
        inty if inty.is_infinite() => {
            //panic!("cc");
            Err(ArmsErr::SectionIntIsInf(
                format!("Error@{}", line!()),
                x,
                p1,
                p2,
            ))
        }
        inty => Ok(inty),
    }
}

pub fn inv_int_exp_y<T>(z: T, p1: (T, T), p2: (T, T)) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    let k: T = (y2 - y1) / (x2 - x1);
    //println!("k={}", k);
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
            //println!("u={}", u);
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
        r if r.is_nan() => {
            println!("{} {} {} {} {}", z, x1, y1, x2, y2);
            //panic!("c");
            Err(ArmsErr::InvIntErr(format!("Error@{}", line!()), z, p1, p2))
        }
        r => Ok(r),
    }
}

impl<T> std::clone::Clone for Section<T>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Section { ..*self }
    }
}

/*
impl<T> std::marker::Copy for Section<T>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
}
*/
impl<T> Section<T>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    pub fn new(xl: T, yl: T, xu: T, yu: T) -> Self {
        //let nan = Float::nan();
        Section {
            _x_l: xl,
            _x_u: xu,
            _x_i: None,
            _y_l: yl,
            _y_u: yu,
            _y_i: None,
            _int_exp_y_l: None,
            _int_exp_y_u: None,
            _cum_int_exp_y_l: None,
            _cum_int_exp_y_u: None,
        }
    }

    pub fn x_l(&self) -> T {
        self._x_l
    }

    pub fn x_u(&self) -> T {
        self._x_u
    }

    pub fn x_i(&self) -> T {
        self._x_i.unwrap()
    }

    pub fn y_l(&self) -> T {
        self._y_l
    }

    pub fn y_u(&self) -> T {
        self._y_u
    }

    pub fn y_i(&self) -> T {
        self._y_i.unwrap()
    }

    pub fn int_exp_y_l(&self) -> T {
        self._int_exp_y_l.unwrap()
    }

    pub fn int_exp_y_u(&self) -> T {
        self._int_exp_y_u.unwrap()
    }

    pub fn cum_int_exp_y_l(&self) -> T {
        self._cum_int_exp_y_l.unwrap()
    }

    pub fn cum_int_exp_y_u(&self) -> T {
        self._cum_int_exp_y_u.unwrap()
    }

    pub fn set_x_l(&mut self, x: T) {
        self._x_l = x;
    }

    pub fn set_x_u(&mut self, x: T) {
        self._x_u = x;
    }

    pub fn set_x_i(&mut self, x: T) {
        if x.is_nan() {
            panic!("x cannot be nan");
        }
        self._x_i = Some(x);
        if self.x_i() < self.x_l() {
            let x = self.x_l();
            self._x_i = Some(x);
        }
        if self.x_i() > self.x_u() {
            let x = self.x_u();
            self._x_i = Some(x);
        }
    }

    pub fn set_y_l(&mut self, y: T) {
        self._y_l = y;
    }

    pub fn set_y_u(&mut self, y: T) {
        self._y_u = y;
    }

    pub fn set_y_i(&mut self, y: T) {
        if y.is_nan() {
            panic!("y cannot be nan");
        }
        self._y_i = Some(y);
    }

    pub fn set_int_exp_y_l(&mut self, y: T) {
        self._int_exp_y_l = Some(y);
    }

    pub fn set_int_exp_y_u(&mut self, y: T) {
        self._int_exp_y_u = Some(y);
    }

    pub fn set_cum_int_exp_y_l(&mut self, y: T) {
        self._cum_int_exp_y_l = Some(y);
    }

    pub fn set_cum_int_exp_y_u(&mut self, y: T) {
        self._cum_int_exp_y_u = Some(y);
    }

    pub fn encloses(&self, x: T) -> bool {
        (x >= self._x_l) && (x <= self._x_u)
    }

    pub fn calc_int_exp_y(self) -> Result<Self, ArmsErr<T>> {
        //println!("aaa");
        let _int_exp_y_l: T = if self._x_i.unwrap() == self._x_l {
            zero()
        } else {
            int_exp_y(
                self.x_i(),
                (self.x_l(), self.y_l()),
                (self.x_i(), self.y_i()),
            )?
        };
        let _int_exp_y_u: T = if self.x_i() == self._x_u {
            zero()
        } else {
            int_exp_y(
                self._x_u,
                (self.x_i(), self.y_i()),
                (self.x_u(), self.y_u()),
            )?
        };

        //println!("inty={} {}", _int_exp_y_l, _int_exp_y_u);
        Ok(Section::<T> {
            _int_exp_y_l: Some(_int_exp_y_l),
            _int_exp_y_u: Some(_int_exp_y_u),
            ..self
        })
    }

    pub fn eval_y(&self, x: T) -> T {
        if x == self._x_l {
            self.y_l()
        } else if x == self._x_u {
            self.y_u()
        } else if x == self.x_i() {
            self.y_i()
        } else if x < self._x_i.unwrap() {
            self.y_l() + (x - self.x_l()) * (self.y_i() - self.y_l()) / (self.x_i() - self.x_l())
        } else {
            self.y_u() + (x - self.x_u()) * (self.y_i() - self.y_u()) / (self.x_i() - self.x_u())
        }
    }
}

pub fn eval<T>(x: T, section_list: &VecDeque<Section<T>>) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    for i in section_list.iter() {
        if i.encloses(x) {
            return Ok(i.eval_y(x));
        }
    }
    Err(ArmsErr::VarOutOfRange(format!("Error@{}", line!()), x))
}

pub fn eval_ey<T>(x: T, section_list: &VecDeque<Section<T>>) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
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
) -> Result<(T, T), ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let (x3, y3) = p3;
    let (x4, y4) = p4;

    /*
    eprintln!("l {} {}", s.x_l(), s.y_l());
    eprintln!("u {} {}", s.x_u(), s.y_u());

    eprintln!("p1 {} {}", x1, y1);
    eprintln!("p2 {} {}", x2, y2);
    eprintln!("p3 {} {}", x3, y3);
    eprintln!("p4 {} {}", x4, y4);
    */
    let two = one::<T>() + one::<T>();
    match if y2.is_infinite() {
        match (x1, y4 + (x1 - x4) / (x3 - x4) * (y3 - y4)) {
            (_, y_i) if y_i < y1 => ((s.x_l() + s.x_u()) / two, (s.y_l() + s.y_u()) / two),
            (x_i, y_i) => (x_i, y_i),
        }
    } else if y4.is_infinite() {
        match (x3, y1 + (x3 - x1) / (x2 - x1) * (y2 - y1)) {
            (_, y_i) if y_i < y3 => ((s.x_l() + s.x_u()) / two, (s.y_l() + s.y_u()) / two),
            (x_i, y_i) => (x_i, y_i),
        }
    } else {
        let x_i = -((-(x3 - x4) * (x2 * y1 - x1 * y2) + (x1 - x2) * (x4 * y3 - x3 * y4))
            / (-(x3 - x4) * (-y1 + y2) + (x1 - x2) * (-y3 + y4)));
        let y_i = -(x2 * y1 * y3 - x4 * y1 * y3 - x1 * y2 * y3 + x4 * y2 * y3 - x2 * y1 * y4
            + x3 * y1 * y4 + x1 * y2 * y4 - x3 * y2 * y4)
            / (-x3 * y1 + x4 * y1 + x3 * y2 - x4 * y2 + x1 * y3 - x2 * y3 - x1 * y4 + x2 * y4);
        /*
        eprintln!("pi {} {}", x_i, y_i);
        eprintln!("");
        */

        match (x_i, y_i) {
            (_, y_i)
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
        (x_i, y_i) if x_i.is_nan() || y_i.is_nan() => {
            //panic!("d");
            Err(ArmsErr::SolveInterSectionErr(
                format!("Error@{}", line!()),
                s.clone(),
                p1,
                p2,
                p3,
                p4,
            ))
        }
        (x_i, y_i) => Ok((x_i, y_i)),
    }
}

pub fn calc_intersection<T>(
    s: &Section<T>,
    before: Option<&Section<T>>,
    after: Option<&Section<T>>,
) -> Result<(T, T), ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    match (before, after) {
        (Some(a), Some(b)) => solve_intersection(
            s,
            (a.x_l(), a.y_l()),
            (a.x_u(), a.y_u()),
            (b.x_l(), b.y_l()),
            (b.x_u(), b.y_u()),
        ),
        (None, Some(b)) => solve_intersection(
            s,
            (s.x_l(), s.y_l()),
            (s.x_l(), Float::infinity()),
            (b.x_l(), b.y_l()),
            (b.x_u(), b.y_u()),
        ),
        (Some(a), None) => solve_intersection(
            s,
            (a.x_l(), a.y_l()),
            (a.x_u(), a.y_u()),
            (s.x_u(), s.y_u()),
            (s.x_u(), Float::infinity()),
        ),
        _ => Err(ArmsErr::InvalidNeighbor(format!("Error@{}", line!()))),
    }
}
/*
enum InsertionResult {
    Succeed,
    SearchFailed,
    PointOverlapped,
}*/

fn calc_cum_int_exp_y<T>(section_list: &mut VecDeque<Section<T>>)
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    let mut cy = zero::<T>();
    for i in section_list {
        i._cum_int_exp_y_l = Some(cy + i._int_exp_y_l.unwrap());
        i._cum_int_exp_y_u = Some(i._cum_int_exp_y_l.unwrap() + i._int_exp_y_u.unwrap());
        cy = i._cum_int_exp_y_u.unwrap();
        //println!("cy={}", cy);
    }
}

fn calc_scale<T>(section_list: &VecDeque<Section<T>>) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    let mut scale: T = Float::neg_infinity();
    for i in section_list {
        let y = fmax(i.y_l(), i.y_u());
        //println!("y={}", y);
        scale = match y {
            x if x > scale => x,
            _ => scale,
        }
    }
    match scale {
        scale if scale.is_infinite() => {
            //panic!("#1");
            Err(ArmsErr::IllConditionedDistribution(
                format!("Error@{}", line!()),
                section_list.clone(),
            ))
        }
        _ => Ok(scale),
    }
}

pub fn update_scale<T>(
    section_list: VecDeque<Section<T>>,
    scale: &mut T,
) -> Result<VecDeque<Section<T>>, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
{
    let mut section_list = section_list;
    let new_scale = calc_scale(&section_list)?;

    for i in &mut section_list {
        i._y_l = i._y_l - new_scale;
        i._y_u = i._y_u - new_scale;
        //i._y_i = Some(i._y_i.unwrap() - new_scale);
        //i._cum_int_exp_y_u = i._cum_int_exp_y_u / new_scale.exp();
        //i._cum_int_exp_y_l = i._cum_int_exp_y_l / new_scale.exp();
    }

    for i in 0..section_list.len() {
        let (xi, yi) = calc_intersection(
            section_list.get(i).unwrap(),
            if i == 0 {
                None
            } else {
                section_list.get(i - 1)
            },
            section_list.get(i + 1),
        )?;
        section_list[i].set_x_i(xi);
        section_list[i].set_y_i(yi);
        let aa = section_list[i].clone().calc_int_exp_y()?;
        section_list[i] = aa;
    }
    calc_cum_int_exp_y(&mut section_list);
    *scale = *scale + new_scale;
    Ok(section_list)
}

pub fn insert_point<T, F>(
    pd: &F,
    section_list: VecDeque<Section<T>>,
    x: T,
    scale: T,
) -> Result<VecDeque<Section<T>>, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
{
    let mut section_list = section_list;
    let mut new_list: VecDeque<Section<T>> = VecDeque::new();
    new_list.reserve(1);
    loop {
        match section_list.pop_front() {
            Some(ref s) if s.x_l() < x && s.x_u() > x => {
                let y = eval_log(pd, x, scale)?;
                let mut news1 = Section::new(s.x_l(), s.y_l(), x, y);
                let mut news2 = Section::new(x, y, s.x_u(), s.y_u());
                {
                    let (xi, yi) = calc_intersection(
                        &news1,
                        if new_list.is_empty() {
                            None
                        } else {
                            new_list.get(new_list.len() - 1)
                        },
                        Some(&news2),
                    )?;
                    news1.set_x_i(xi);
                    news1.set_y_i(yi);
                    news1 = news1.calc_int_exp_y()?;
                }
                {
                    let (xi, yi) = calc_intersection(&news2, Some(&news1), section_list.get(0))?;
                    news2.set_x_i(xi);
                    news2.set_y_i(yi);
                    news2 = news2.calc_int_exp_y()?;
                }
                if new_list.len() > 1 {
                    let (xi, yi) = calc_intersection(
                        new_list.back().unwrap(),
                        new_list.get(new_list.len() - 2),
                        Some(&news1),
                    )?;
                    new_list.back_mut().unwrap().set_x_i(xi);
                    new_list.back_mut().unwrap().set_y_i(yi);
                    //let aa = new_list.back_mut().unwrap().calc_int_exp_y()?;
                    let aa = new_list.pop_back().unwrap().calc_int_exp_y()?;
                    new_list.push_back(aa);
                    //new_list.back_mut().unwrap()._int_exp_y_l = aa._int_exp_y_l;
                    //new_list.back_mut().unwrap()._int_exp_y_u = aa._int_exp_y_u;
                }
                if section_list.len() > 1 {
                    let (xi, yi) = calc_intersection(
                        section_list.front().unwrap(),
                        Some(&news2),
                        section_list.get(1),
                    )?;
                    section_list.front_mut().unwrap().set_x_i(xi);
                    section_list.front_mut().unwrap().set_y_i(yi);
                    //let aa = section_list.pop_front().unwrap().calc_int_exp_y()?;
                    let aa = section_list.pop_front().unwrap().calc_int_exp_y()?;
                    section_list.push_front(aa);
                    //section_list.front_mut().unwrap()._int_exp_y_l = aa._int_exp_y_l;
                    //section_list.front_mut().unwrap()._int_exp_y_u = aa._int_exp_y_u;
                }
                new_list.push_back(news1);
                new_list.push_back(news2);
            }
            Some(s) => new_list.push_back(s),
            _ => break,
        }
    }
    Ok(new_list)
}

pub fn init<T, F, V>(
    pd: &F,
    xrange: (T, T),
    init_x1: &V,
    scale: &mut T,
) -> Result<VecDeque<Section<T>>, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
    V: Clone + IndexMut<usize, Output = T> + HasLength + std::marker::Sync + std::marker::Send,
{
    if init_x1.length() < 3 {
        return Err(ArmsErr::TooFewInitPoints(format!("Error@{}", line!())));
    }
    for i in 0..(init_x1.length() - 1) {
        if init_x1[i] >= init_x1[i + 1] {
            return Err(ArmsErr::DataNotInOrder(format!("Error@{}", line!())));
        }
    }
    let mut init_x = vec![xrange.0];
    for i in 0..init_x1.length() {
        init_x.push(init_x1[i]);
    }
    init_x.push(xrange.1);

    let mut section_list = VecDeque::<Section<T>>::new();
    for i in 0..(init_x.length() - 1) {
        let mut s = Section::new(
            init_x[i],
            eval_log(pd, init_x[i], zero())?,
            init_x[i + 1],
            eval_log(pd, init_x[i + 1], zero())?,
        );
        //s.set_x_l(init_x[i]);
        //s.set_x_u(init_x[i + 1]);
        //s.set_y_l(eval_log(pd, init_x[i], zero())?);
        //s.set_y_u(eval_log(pd, init_x[i + 1], zero())?);
        section_list.push_back(s);
    }

    *scale = calc_scale(&section_list)?;
    //println!("{}", *scale);
    for s in &mut section_list {
        s._y_l = s._y_l - *scale;
        s._y_u = s._y_u - *scale;
    }

    for i in 0..section_list.len() {
        let (xi, yi) = if i == 0 {
            calc_intersection(&section_list[i], None, Some(&section_list[i + 1]))?
        } else if i == section_list.len() - 1 {
            calc_intersection(&section_list[i], Some(&section_list[i - 1]), None)?
        } else {
            calc_intersection(
                &section_list[i],
                Some(&section_list[i - 1]),
                Some(&section_list[i + 1]),
            )?
        };

        section_list[i].set_x_i(xi);
        section_list[i].set_y_i(yi);
        section_list[i] = section_list[i].clone().calc_int_exp_y()?;
        //println!("{:?}", &section_list[i]);
    }

    calc_cum_int_exp_y(&mut section_list);
    //
    assert!(
        section_list
            .back()
            .unwrap()
            ._cum_int_exp_y_u
            .unwrap()
            .is_finite()
    );
    Ok(section_list)
}

pub fn search_point<T>(
    section_list: &VecDeque<Section<T>>,
    p: T,
    //pd: &F,
    //scale: T,
) -> Result<usize, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    //F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
{
    let x = {
        match section_list.back() {
            Some(x) => p * x.cum_int_exp_y_u(),
            _ => {
                //panic!("#2");
                return Err(ArmsErr::IllConditionedDistribution(
                    format!("Error@{}", line!()),
                    section_list.clone(),
                ));
            }
        }
    };

    for i in 0..section_list.len() {
        let xx = section_list[i].cum_int_exp_y_u();
        if (p < one() && x < xx) || (p == one() && x <= xx) {
            return Ok(i);
        }
    }
    //panic!("#3");
    Err(ArmsErr::IllConditionedDistribution(
        format!("Error@{}", line!()),
        section_list.clone(),
    ))
}

pub fn check_range<T, F>(
    pd: &F,
    section_list: VecDeque<Section<T>>,
    scale: &mut T,
) -> Result<VecDeque<Section<T>>, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
{
    let mut section_list = section_list;
    let two: T = one::<T>() + one::<T>();
    loop {
        let mut has_inf = false;
        let mut n = section_list.len();
        for i in 0..section_list.len() {
            if section_list[i]._cum_int_exp_y_u.unwrap().is_infinite()
                || section_list[i]._cum_int_exp_y_l.unwrap().is_infinite()
            {
                has_inf = true;
                n = i;
                break;
            }
        }

        if has_inf {
            let x = if section_list[n]._cum_int_exp_y_l.unwrap().is_infinite() {
                (section_list[n]._x_l + section_list[n]._x_i.unwrap()) / two
            } else {
                (section_list[n]._x_i.unwrap() + section_list[n]._x_u) / two
            };
            section_list = insert_point(pd, section_list, x, *scale)?;
            section_list = update_scale(section_list, scale)?;
        } else {
            break;
        }
    }
    Ok(section_list)
}

pub fn fetch_one<T, R>(
    section_list: &VecDeque<Section<T>>,
    rng: &mut R,
    //pd: &F,
    //scale: T,
) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    //F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
    R: Rng,
{
    let p = rng.gen_range(zero(), one());
    let n = search_point(section_list, p /*, pd, scale*/)?;
    let y: T = match section_list.back() {
        Some(s) => p * s._cum_int_exp_y_u.unwrap(),
        _ => {
            return Err(ArmsErr::IllConditionedDistribution(
                format!("Error@{}", line!()),
                section_list.clone(),
            ))
        }
    };
    let (ybase, x1, y1, x2, y2) = if y >= section_list[n]._cum_int_exp_y_l.unwrap()
        && y <= section_list[n]._cum_int_exp_y_u.unwrap()
    {
        (
            section_list[n]._cum_int_exp_y_l.unwrap(),
            section_list[n].x_i(),
            section_list[n].y_i(),
            section_list[n].x_u(),
            section_list[n].y_u(),
        )
    } else if y < section_list[n]._cum_int_exp_y_l.unwrap() {
        (
            if n > 0 {
                match section_list.get(n - 1) {
                    Some(s) => {
                        assert!(
                            y - s._cum_int_exp_y_u.unwrap()
                                < section_list.get(n).unwrap()._int_exp_y_l.unwrap()
                        );
                        s._cum_int_exp_y_u.unwrap()
                    }
                    _ => zero::<T>(),
                }
            } else {
                zero::<T>()
            },
            section_list[n].x_l(),
            section_list[n].y_l(),
            section_list[n].x_i(),
            section_list[n].y_i(),
        )
    } else {
        return Err(ArmsErr::IllConditionedDistribution(
            format!("Error@{}", line!()),
            section_list.clone(),
        ));
    };

    if y == ybase {
        Ok(x1)
    } else {
        match inv_int_exp_y(y - ybase, (x1, y1), (x2, y2)) {
            Ok(x) => Ok(x),
            Err(x) => {
                println!("n={}", n);
                println!("y={} ybase={}", y, ybase);
                for i in section_list {
                    println!("{:?}", i);
                }
                //panic!("aaa");
                Err(x)
            }
        }
        //Ok(inv_int_exp_y(y - ybase, (x1, y1), (x2, y2))?)
    }
}

pub fn dump_section_list<T, F>(section_list: &VecDeque<Section<T>>, pd: Option<&F>, dx: T, scale: T)
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
{
    for s in section_list {
        println!("{} {}", s.x_l(), s.y_l());
        println!("{} {}", s.x_i(), s.y_i());
        println!("{} {}", s.x_u(), s.y_u());
        println!("");
    }

    println!("no no no");
    //return;
    match pd {
        Some(ref f) => {
            let xmin = section_list.front().unwrap().x_l();
            let xmax = section_list.back().unwrap().x_u();
            let mut x = xmin;
            loop {
                println!("{} {}", x, eval_log(f, x, scale).unwrap());
                x = x + dx;
                if x > xmax {
                    break;
                }
            }
        }
        _ => (),
    }
    //println!("{}", scale)
}

pub fn sample<T, F, R, V>(
    pd: &F,
    xrange: (T, T),
    init_x: &V,
    xcur: T,
    n: usize,
    rng: &mut R,
    xmchange_count: &mut usize,
) -> Result<T, ArmsErr<T>>
where
    T: Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display
        + std::fmt::Debug,
    R: Rng,
    F: Fn(T) -> T + std::marker::Sync + std::marker::Send,
    V: Clone + IndexMut<usize, Output = T> + HasLength + std::marker::Sync + std::marker::Send,
{
    //let two = one::<T>() + one::<T>();
    let mut xcur = xcur;
    let mut scale = zero();
    let mut section_list = init(pd, xrange, init_x, &mut scale)?;
    let mut x: T;
    let mut xm = xcur;
    let mut i = 0;
    while i < n {
        x = fetch_one(&section_list, rng /*, pd, scale*/)?;

        let xa = if rng.gen_range(zero::<T>(), one::<T>()).ln() + eval(x, &section_list)?
            > eval_log(&pd, x, scale)?
        {
            section_list = insert_point(&pd, section_list, x, scale)?;
            section_list = update_scale(section_list, &mut scale)?;

            let need_cr = if let Some(x) = section_list.back() {
                if x._cum_int_exp_y_u.unwrap().is_infinite() {
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if need_cr {
                section_list = check_range(pd, section_list, &mut scale)?;
            }
            continue;
        } else {
            x
        };

        let u: T = rng.gen_range(zero(), one());

        let ya = eval_log(&pd, xa, scale)?;
        let ycur = eval_log(&pd, xcur, scale)?;
        let eycur = eval(xcur, &section_list)?;
        let eya = eval(xa, &section_list)?;

        if u.ln() > fmin(zero(), ya - ycur + fmin(ycur, eycur) - fmin(ya, eya)) {
            xm = xcur;
        } else {
            *xmchange_count = *xmchange_count + 1;
            xm = xa;
            i = i + 1;
        }
        xcur = xm;
    }
    Ok(xm)
}
