// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};

use std::cell::RefCell;
use std::rc::Rc;

/// Sigmoid 関数
#[derive(Debug, Clone)]
pub struct SigmoidFunction {}
impl<V: MathOps> Function<V> for SigmoidFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Sigmoid".to_string()
    }

    // Sigmoid の順伝播
    fn forward(&self, inputs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("sigmoid(forward)");

        let x = inputs[0].clone();

        debug!("sigmoid(forward) x: {:?}", x.flatten().to_vec());

        let result = vec![x.mapv(|tmp| {
            let y = (V::to_f64(&tmp).unwrap() * 0.5).tanh() * 0.5 + 0.5;
            V::from(y).unwrap()
        })];
        result
    }

    /// 逆伝播
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("sigmoid(backward)");

        let x = inputs[0].clone();

        let y = sigmoid(x);
        let gx = &(&gys[0].clone() * &y.clone())
            * &(&Variable::new(RawVariable::new(V::from(1).unwrap())) - &y.clone());

        vec![gx]
    }
}

/// Sigmoid 関数
///
/// Arguments
/// * x (Variable<V>): 変数
/// * w (Variable<V>): 重み
/// * b (Option<Variable<V>>): バイアス
///
/// Return
/// * Variable<V>: 結果
pub fn sigmoid<V: MathOps>(x: Variable<V>) -> Variable<V> {
    let mut sigmoid = FunctionExecutor::new(Rc::new(RefCell::new(SigmoidFunction {})));

    let inputs = vec![x];

    // 順伝播
    sigmoid
        //.forward(vec![x.clone(), w.clone()])
        .forward(inputs)
        .get(0)
        .unwrap()
        .clone()
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    // use rand::prelude::*;
    use plotters::chart::ChartBuilder;
    use plotters::prelude::{BitMapBackend, IntoDrawingArea};
    use plotters::series::LineSeries;
    use plotters::style::{Color, IntoFont, BLACK, RED, WHITE};
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_0() {
        let rand_x = rand::random::<f64>();

        let x = Variable::new(RawVariable::new(rand_x));

        let mut sigmoid: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(SigmoidFunction {})));

        utils::gradient_check(&mut sigmoid, vec![x.clone()]);
    }

    #[test]
    fn test_num_grad_check_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((3, 2), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![3, 2],
            x_var.flatten().to_vec(),
        ));

        let mut sigmoid: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(SigmoidFunction {})));

        utils::gradient_check(&mut sigmoid, vec![x.clone()]);
    }

    #[test]
    fn test_num_grad_check_2() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((10, 10, 10), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![10, 10, 10],
            x_var.flatten().to_vec(),
        ));

        let mut sigmoid: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(SigmoidFunction {})));

        utils::gradient_check(&mut sigmoid, vec![x.clone()]);
    }

    #[test]
    fn test_sigmoid_graph() {
        // グラフ描画
        // 描画先の Backend を初期化する。
        let root = BitMapBackend::new("graph/step43_sigmoid.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // グラフの軸の設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("Sigmoid", ("sans-serif", 50).into_font())
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
                    let y = sigmoid(Variable::new(RawVariable::new(x)));
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
}
