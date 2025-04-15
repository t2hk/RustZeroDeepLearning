// ライブラリを一括でインポート
#[allow(unused_imports)]
use crate::modules::math::*;
use crate::modules::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Array1, Array2, Ix0, Ix1, Ix2, IxDyn};

/// Sphere 関数
/// z = x^2 + y^2 を計算する。
///
/// Arguments:
/// * x (Variable<V>)
/// * y (Variable<V>)
/// Returns:
/// * Variable<V>: Sphere 関数の計算結果
pub fn sphere<V: MathOps>(x: Variable<V>, y: Variable<V>) -> Variable<V> {
    let z = &(&x ^ 2) + &(&y ^ 2);
    z
}

/// matyas 関数
/// z = 0.26 * (x^2 + y^2) - 0.48 * x * y を計算する。
///
/// Arguments:
/// * x (Variable<V>)
/// * y (Variable<V>)
/// Returns:
/// * Variable<V>: matyas 関数の計算結果
pub fn matyas<V: MathOps>(x: Variable<V>, y: Variable<V>) -> Variable<V> {
    let z = &(0.26 * &(&(&x ^ 2) + &(&y ^ 2))) - &(0.48 * &(&x * &y));
    z
}

/// Goldstein-Price 関数
/// z = [1 + (x * y * 1)^2 (19 - 14x + 3x^2 -14y + 6xy + 3y^2)][30 + (2x - 3y)^2 (18 - 32x + 12x^2 + 48y -36xy + 27y^2)] を計算する。
///
/// Arguments:
/// * x (Variable<V>)
/// * y (Variable<V>)
/// Returns:
/// * Variable<V>: matyas 関数の計算結果
pub fn goldstein<V: MathOps>(x: Variable<V>, y: Variable<V>) -> Variable<V> {
    // let a = &(&(&x + &y) + 1) ^ 2usize;
    // let b = 14 * &x;
    // let c = 3 * &(&x ^ 2);
    // let d = 14 * &y;
    // let e = &(6 * &x) * &y;
    // let f = 3 * &(&y ^ 2);
    // let a_f = &(1 + &(&a * &(&(&(&(&(19 - &b) + &c) - &d) + &e) + &f)));

    // let g = &(&(2 * &x) - &(3 * &y)) ^ 2usize;
    // let h = 18 - &(32 * &x);
    // let i = 12 * &(&x ^ 2usize);
    // let j = &(48 * &y) - &(36 * &(&x * &y));
    // let k = 27 * &(&y ^ 2usize);

    // let g_k = &(30 + &(&g * (&(&(&(&h + &i) + &j) + &k))));

    // let z = a_f * g_k;

    // let a_f = &(1 + &(&(&(&(&x + &y) + 1) ^ 2usize) * &(&(&(&(&(19 - &(14 * &x)) + &(3 * &(&x ^ 2))) - &(14 * &y)) + &(&(6 * &x) * &y)) + &(3 * &(&y ^ 2)))));
    // let g_k = &(30 + &(&(&(&(2 * &x) - &(3 * &y)) ^ 2usize) * (&(&(&(&(18 - &(32 * &x)) + &(12 * &(&x ^ 2usize))) + &(&(48 * &y) - &(36 * &(&x * &y)))) + &(27 * &(&y ^ 2usize))))));

    let z = &(1 + &(&(&(&(&x + &y) + 1) ^ 2)
        * &(&(&(&(&(19 - &(14 * &x)) + &(3 * &(&x ^ 2))) - &(14 * &y)) + &(&(6 * &x) * &y))
            + &(3 * &(&y ^ 2)))))
        * &(30
            + &(&(&(&(2 * &x) - &(3 * &y)) ^ 2)
                * (&(&(&(&(18 - &(32 * &x)) + &(12 * &(&x ^ 2)))
                    + &(&(48 * &y) - &(36 * &(&x * &y))))
                    + &(27 * &(&y ^ 2))))));

    z
}

/// ローゼンブロック関数
pub fn rosenblock<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    let lhs = 100 * &(&(&x1 - &(&x0 ^ 2)) ^ 2);
    let rhs = &(&x0 - 1) ^ 2;

    &lhs + &rhs
}

/// 行列積
///
/// Arguments:
/// * x0 (Array<V, IxDyn>)
/// * x1 (Array<V, IxDyn>)
///
/// Return
/// * Array<V, IxDyn>
pub fn dot<V: MathOps>(x0: Array<V, IxDyn>, x1: Array<V, IxDyn>) -> Array<V, IxDyn> {
    debug!("dot x ndim: {:?}, w ndim: {:?}", x0.ndim(), x1.ndim());

    debug!(
        "dot x0 ndim: {:?}, dim: {:?}, shape: {:?}",
        x0.ndim(),
        x0.dim(),
        x0.shape()
    );
    debug!(
        "dot x1 ndim: {:?}, dim: {:?}, shape: {:?}",
        x1.ndim(),
        x1.dim(),
        x1.shape()
    );

    match (x0.ndim(), x1.ndim()) {
        (0, 0) => {
            info!("dot for shape (0, 0)");

            let x_tmp = x0.into_dimensionality::<Ix0>().unwrap();
            let w_tmp = x1.into_dimensionality::<Ix0>().unwrap();

            let result_value = x_tmp
                .iter()
                .zip(w_tmp.iter())
                .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

            Array::from_elem(IxDyn(&[]), result_value)
        }
        (0, 1) => {
            info!("dot for shape (0, 1)");

            let x_tmp = x0.into_dimensionality::<Ix0>().unwrap();
            let w_tmp = x1.into_dimensionality::<Ix1>().unwrap();

            let result_value = w_tmp
                .iter()
                .zip(x_tmp.iter())
                .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

            Array::from_shape_vec(vec![1], vec![result_value]).unwrap()
        }
        (1, 0) => {
            info!("dot for shape (1, 0)");

            let x_tmp = x0.into_dimensionality::<Ix1>().unwrap();
            let w_tmp = x1.into_dimensionality::<Ix0>().unwrap();

            let result_value = x_tmp
                .iter()
                .zip(w_tmp.iter())
                .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

            Array::from_shape_vec(vec![1], vec![result_value]).unwrap()
        }
        (1, 1) => {
            info!("dot for shape (1, 1)");

            let x_tmp = x0.into_dimensionality::<Ix1>().unwrap();
            let w_tmp = x1.into_dimensionality::<Ix1>().unwrap();

            let result_value = x_tmp
                .iter()
                .zip(w_tmp.iter())
                .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

            Array::from_elem(IxDyn(&[]), result_value)
        }
        (1, 2) => {
            info!("dot for shape (1, 2)");

            if x0.shape()[0] != x1.shape()[0] {
                panic!(
                    "shapes {:?} and {:?} not aligned. {} != {}",
                    x0.shape(),
                    x1.shape(),
                    x0.shape()[0],
                    x1.shape()[0]
                );
            }

            let x_tmp: Array1<V> = x0.into_dimensionality::<Ix1>().unwrap();
            let w_tmp: Array2<V> = x1.into_dimensionality::<Ix2>().unwrap();

            let x_len = x_tmp.len();
            let w_cols = w_tmp.shape()[1];

            let mut result = Array::zeros((w_cols,));

            for j in 0..w_cols {
                let mut sum = V::zero();
                for i in 0..x_len {
                    sum = sum + x_tmp[i].clone() * w_tmp[[i, j]].clone();
                }
                result[j] = sum;
            }

            dbg!(&result);

            result.into_dimensionality::<IxDyn>().unwrap()
        }
        (2, 1) => {
            info!("dot for shape (2, 1)");

            if x0.shape()[0] != x1.shape()[0] {
                panic!(
                    "shapes {:?} and {:?} not aligned. {} != {}",
                    x0.shape(),
                    x1.shape(),
                    x0.shape()[0],
                    x1.shape()[0]
                );
            }
            let x_tmp = x0.into_dimensionality::<Ix2>().unwrap();
            let w_tmp = x1.into_dimensionality::<Ix1>().unwrap();

            let x_rows = x_tmp.shape()[0];
            let x_cols = x_tmp.shape()[1];

            // 結果は x_rows 長のベクトルになる
            let mut result = Array::zeros((x_rows,));

            for i in 0..x_rows {
                let mut sum = V::zero();
                for j in 0..x_cols {
                    sum = sum + x_tmp[[i, j]].clone() * w_tmp[j].clone();
                }
                result[i] = sum;
            }

            result.into_dimensionality::<IxDyn>().unwrap()
        }
        (2, 2) => {
            info!("dot for shape (2, 2)");
            let x_tmp = x0.into_dimensionality::<Ix2>().unwrap();
            let w_tmp = x1.into_dimensionality::<Ix2>().unwrap();

            let x_rows = x_tmp.shape()[0];
            let x_cols = x_tmp.shape()[1];
            let w_cols = w_tmp.shape()[1];

            // x_cols と w_rows（= x_tmp.shape()[1]とw_tmp.shape()[0]）は同じサイズであることを確認
            assert_eq!(x_cols, w_tmp.shape()[0], "行列の次元が不一致です");

            // 結果は x_rows x w_cols の行列になる
            let mut result = Array::zeros((x_rows, w_cols));

            for i in 0..x_rows {
                for j in 0..w_cols {
                    let mut sum = V::zero();
                    for k in 0..x_cols {
                        sum = sum + x_tmp[[i, k]].clone() * w_tmp[[k, j]].clone();
                    }
                    result[[i, j]] = sum;
                }
            }

            result.into_dimensionality::<IxDyn>().unwrap()
        }
        _ => {
            error!("dot for invalid shape");
            debug!("x ndim: {} w ndim: {}", x0.ndim(), x1.ndim());
            debug!(
                "x : {:?} w : {:?}",
                x0.flatten().to_vec(),
                x1.flatten().to_vec()
            );
            panic!("error: invalid dimension. x: {:?}, w: {:?}", x0, x1);
        }
    }
}

/// linear 関数 (線形変換)
/// y = w * x + b
///
/// Arguments:
/// * x (Variable<V>): 入力値
/// * w (Variable<V>): 重み
/// * b (Option<Variable<V>>): バイアス
///
/// Retrun
/// * Variable<V>: 線形変換の結果
pub fn linear_simple<V: MathOps>(
    x: Variable<V>,
    w: Variable<V>,
    b: Option<Variable<V>>,
) -> Variable<V> {
    let t = matmul(x, w);
    if let Some(b_var) = b {
        let y = &t + &b_var;
        std::mem::drop(t);
        return y;
    }

    return t;
}

/// シグモイド関数
///
/// Arguments:
/// * x (Variable<V>): 入力値
///
/// Retrun:
/// * Variable<V>
pub fn sigmoid_simple<V: MathOps>(x: Variable<V>) -> Variable<V> {
    let y = 1.0 / &(1 + &exp(-x));
    y
}

fn type_of<T>(_: T) -> String {
    let a = std::any::type_name::<T>();
    return a.to_string();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};
    use rand::prelude::*;

    use plotters::chart::ChartBuilder;
    use plotters::prelude::{BitMapBackend, Circle, EmptyElement, IntoDrawingArea, PathElement};
    use plotters::series::{LineSeries, PointSeries};
    use plotters::style::{Color, IntoFont, BLACK, BLUE, GREEN, MAGENTA, RED, WHITE};

    /// シグモイド関数のグラフ描画
    #[test]
    fn test_sigmoid_simple() {
        // グラフ描画
        // 描画先の Backend を初期化する。
        let root =
            BitMapBackend::new("graph/step43_simple_sigmoid.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // グラフの軸の設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("y=1 / (1 + exp(-x))", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-5.0..5.0, 0.0..1.0)
            .unwrap();
        chart.configure_mesh().draw().unwrap();

        // sigmoid_simple(Variable::new(RawVariable::new(x)))
        // 元データのプロット
        // 折れ線グラフ（関数グラフ）を描画
        // 折れ線グラフ（関数グラフ）を描画
        chart
            .draw_series(LineSeries::new(
                (-500..=500).map(|i| {
                    let x = i as f64 / 100.0;
                    let y = sigmoid_simple(Variable::new(RawVariable::new(x)));
                    (x, y.clone().borrow().get_data().flatten().to_vec()[0])
                }),
                &RED,
            ))
            .unwrap();

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }

    /// ステップ29 ニュートン法による最適化の実装
    #[test]
    fn test_step29_newton_method() {
        /// y = f(x)
        fn f<V: MathOps>(x: Variable<V>) -> Variable<V> {
            let y = &(&x ^ 4) - &(2 * &(&x ^ 2));
            return y;
        }

        /// y = f(x) の２階微分
        fn gx2<V: MathOps>(x: Variable<V>) -> Variable<V> {
            let y = &(12 * &(&x ^ 2)) - 4;
            return y;
        }

        let mut x = Variable::new(RawVariable::new(2.0));
        let iters = 10;

        for i in 0..iters {
            println!("i: {} x: {:?}", i, x.borrow().get_data());
            // 書籍と同じ値になるかテストする。
            match i {
                0 => assert_eq!(2.0, x.borrow().get_data()[[]]),
                1 => assert_eq!(1.4545454545454546, x.borrow().get_data()[[]]),
                2 => assert_eq!(1.1510467893775467, x.borrow().get_data()[[]]),
                3 => assert_eq!(1.0253259289766978, x.borrow().get_data()[[]]),
                4 => assert_eq!(1.0009084519430513, x.borrow().get_data()[[]]),
                5 => assert_eq!(1.0000012353089454, x.borrow().get_data()[[]]),
                6 => assert_eq!(1.000000000002289, x.borrow().get_data()[[]]),
                7 => assert_eq!(1.0, x.borrow().get_data()[[]]),
                8 => assert_eq!(1.0, x.borrow().get_data()[[]]),
                9 => assert_eq!(1.0, x.borrow().get_data()[[]]),
                _ => {}
            }

            let y = f(x.clone());
            x.borrow_mut().clear_grad();

            y.backward();
            let x_data = Variable::new(RawVariable::new(x.borrow().get_data()));
            let x_grad = x.borrow().get_grad().unwrap().borrow().get_data()[[]];

            let new_data: Variable<f64> = &x_data - &(x_grad / &gx2(x_data.clone()));

            x.set_data(new_data.borrow().get_data());
        }
    }

    #[test]
    /// ローゼンブロック関数のテスト
    fn test_rosenblock() {
        let x0 = Variable::new(RawVariable::new(0.0));
        let x1 = Variable::new(RawVariable::new(2.0));

        let y = rosenblock(x0.clone(), x1.clone());
        y.backward();

        let expected_x0_grad = Array::from_elem(IxDyn(&[]), -2.0);
        let expected_x1_grad = Array::from_elem(IxDyn(&[]), 400.0);

        assert_eq!(
            expected_x0_grad,
            x0.borrow().get_grad().unwrap().borrow().get_data()
        );
        assert_eq!(
            expected_x1_grad,
            x1.borrow().get_grad().unwrap().borrow().get_data()
        );
    }

    /// ローゼンブロック関数の勾配降下法
    #[test]
    fn test_step28() {
        let mut x0 = Variable::new(RawVariable::new(0.0));
        let mut x1 = Variable::new(RawVariable::new(2.0));
        let lr = 0.001;
        let iters = 10000;

        for _i in 0..iters {
            println!(
                "x0: {:?} x1: {:?}",
                x0.borrow().get_data(),
                x1.borrow().get_data()
            );

            let y = rosenblock(x0.clone(), x1.clone());
            x0.borrow_mut().clear_grad();
            x1.borrow_mut().clear_grad();
            y.backward();

            let x0_data = x0.borrow().get_data();
            let x1_data = x1.borrow().get_data();
            let x0_grad = x0.borrow().get_grad().unwrap().borrow().get_data()[[]];
            let x1_grad = x1.borrow().get_grad().unwrap().borrow().get_data()[[]];

            x0.set_data(x0_data - lr * x0_grad);
            x1.set_data(x1_data - lr * x1_grad);
        }
    }

    /// Sphere 関数のテスト
    #[test]
    fn test_sphere_1() {
        let x = Variable::new(RawVariable::new(1));
        let y = Variable::new(RawVariable::new(1));
        let z = sphere(x.clone(), y.clone());

        z.backward();

        let expected = Array::from_elem(IxDyn(&[]), 2);
        let expect_x_grad = Array::from_elem(IxDyn(&[]), 2);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 2);
        assert_eq!(expected, z.borrow().get_data());
        assert_eq!(
            expect_x_grad,
            x.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
        assert_eq!(
            expect_y_grad,
            y.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
    }

    /// matyas 関数のテスト
    #[test]
    fn test_matyas_1() {
        let x = Variable::new(RawVariable::new(1.0));
        let y = Variable::new(RawVariable::new(1.0));
        let z = matyas(x.clone(), y.clone());

        z.backward();

        let expected = Array::from_elem(IxDyn(&[]), 0.040000000000000036);
        let expect_x_grad = Array::from_elem(IxDyn(&[]), 0.040000000000000036);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 0.040000000000000036);
        assert_eq!(expected, z.borrow().get_data());
        assert_eq!(
            expect_x_grad,
            x.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
        assert_eq!(
            expect_y_grad,
            y.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
    }

    /// goldstein 関数のテスト
    #[test]
    fn test_goldstein_1() {
        let x = Variable::new(RawVariable::new(1));
        let y = Variable::new(RawVariable::new(1));
        let z = goldstein(x.clone(), y.clone());

        z.backward();

        let expected = Array::from_elem(IxDyn(&[]), 1876);
        let expect_x_grad = Array::from_elem(IxDyn(&[]), -5376);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 8064);

        assert_eq!(expected, z.borrow().get_data());
        assert_eq!(
            expect_x_grad,
            x.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
        assert_eq!(
            expect_y_grad,
            y.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
    }

    /// linear_simple のテスト
    #[test]
    fn test_linear_simple() {
        let w = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![1., 2., 3., 4.],
        ));
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![2., 2., 2., 2.],
        ));
        let b = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![10., 10., 10., 10.],
        ));

        let y = linear_simple(x, w, Some(b));
        assert_eq!(vec![2, 2], y.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![18., 22., 18., 22.,],
            y.borrow().get_data().flatten().to_vec()
        );
    }
}
