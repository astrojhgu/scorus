use num_traits::float::Float;

fn gen_spline1<T>(x_list: &[T], y_list: &[T], d2y0: T, d2y1: T) -> Vec<T>
where
    T: Float + Copy + 'static,
{
    let n = x_list.len();
    let mut y2_list = Vec::new();
    let mut u = Vec::new();
    if d2y0.abs() < T::epsilon() {
        y2_list.push(T::zero());
        u.push(T::zero());
    } else {
        y2_list.push(-T::one() / (T::one() + T::one()));
        u.push(
            (T::from(3.0).unwrap() / (x_list[1] - x_list[0]))
                * ((y_list[1] - y_list[0]) / (x_list[1] - x_list[0]) - d2y0),
        );
    }

    for i in 1..(n - 1) {
        let sig = (x_list[i] - x_list[i - 1]) / (x_list[i + 1] - x_list[i - 1]);
        let p = sig * y2_list[i - 1] + T::from(2.0).unwrap();
        y2_list.push((sig - T::one()) / p);
        u.push(
            (y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])
                - (y_list[i] - y_list[i - 1]) / (x_list[i] - x_list[i - 1]),
        );
        u[i] =
            (T::from(6.0).unwrap() * u[i] / (x_list[i + 1] - x_list[i - 1]) - sig * u[i - 1]) / p;
    }
    let (qn, un) = if T::abs(d2y1) < T::epsilon() {
        (T::zero(), T::zero())
    } else {
        (
            T::from(0.5).unwrap(),
            (T::from(3.0).unwrap() / (x_list[n - 1] - x_list[n - 2]))
                * (d2y1 - (y_list[n - 1] - y_list[n - 2]) / (x_list[n - 1] - x_list[n - 2])),
        )
    };

    let t = y2_list[n - 2];
    y2_list.push((un - qn * u[n - 2]) / (qn * t + T::one()));
    for i in (0..n - 1).rev() {
        y2_list[i] = y2_list[i] * y2_list[i + 1] + u[i];
    }
    y2_list
}

pub fn gen_spline<T>(input_x: &[T], input_y: &[T], d2y0: T, d2y1: T) -> Box<dyn Fn(T) -> T>
where
    T: Float + Copy + 'static,
{
    assert!(input_x.len() == input_y.len());
    let mut x_list = Vec::new();
    let mut y_list = Vec::new();
    for i in 0..input_x.len() {
        if let Some(x) = x_list.last() {
            assert!(*x <= input_x[i]);
        }
        x_list.push(input_x[i]);
        y_list.push(input_y[i])
    }

    let y2_list = gen_spline1(&x_list, &y_list, d2y0, d2y1);

    Box::new(move |x: T| {
        if x < *x_list.first().unwrap() {
            return *y_list.first().unwrap();
        }
        if x >= *x_list.last().unwrap() {
            return *y_list.last().unwrap();
        }
        let mut n1 = 0;
        let mut n2 = x_list.len() - 1;
        while (n2 - n1) != 1 {
            if x_list[n1 + 1] <= x {
                n1 += 1;
            }
            if x_list[n2 - 1] > x {
                n2 -= 1;
            }
        }
        let h = x_list[n2] - x_list[n1];
        let a = (x_list[n2] - x) / h;
        let b = (x - x_list[n1]) / h;
        a * y_list[n1]
            + b * y_list[n2]
            + ((a.powi(3) - a) * y2_list[n1] + (b.powi(3) - b) * y2_list[n2]) * h.powi(2)
                / T::from(6.0).unwrap()
    })
}
