// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;

/// AdaGrad 構造体
#[derive(Debug, Clone)]
pub struct AdaGrad<V> {
    lr: f64,  // 学習係数
    eps: f64, // イプシロン
    hs: HashMap<String, Array<V, IxDyn>>,
}

impl<V> AdaGrad<V> {
    /// オプティマイザ AdaGrad を初期化する。
    ///
    /// Arguments
    /// * lr (f64): 学習率
    /// * eps (f64): イプシロン
    ///
    /// Return
    /// * AdaGrad: AdaGrad 構造体のインスタンス
    pub fn new(lr: f64, eps: f64) -> Self {
        AdaGrad {
            lr: lr,
            eps: eps,
            hs: HashMap::new(),
        }
    }
}

/// AdaGrad の Optimizer トレイト実装
impl<V: MathOps> Optimizer for AdaGrad<V> {
    type Val = V;

    /// 勾配を更新する。
    ///
    /// Argumesnts
    /// * param (Variable<V>): 更新対象の変数
    fn update_one(&mut self, param: &mut Variable<V>) {
        let param_id = format!("{:?}", param.as_ref().as_ptr());

        if self.hs.get(&param_id).is_none() {
            let zero_array = Array::zeros(param.get_data().shape());
            self.hs.insert(param_id.clone(), zero_array);
        }

        let h = self.hs.get(&param_id).unwrap();
        let grad = param.get_grad().unwrap().get_data();
        let grad_2 = grad.clone() * grad.clone();

        let h_plus_grad_2 = h + grad_2;
        self.hs.insert(param_id.to_string(), h_plus_grad_2.clone());

        // param.data -= lr * grad / (xp.sqrt(h) + eps)

        // lr / (sqrt(h_plus_grad_2) + eps)
        let lr_div_sqrt_h_plus_grad_2_plus_eps = h_plus_grad_2
            .mapv(|x| V::from(self.lr / (x.to_f64().unwrap().sqrt() + self.eps)).unwrap());
        // (lr * grad) / (sqrt(h + grad^2) + eps)
        let update_value = lr_div_sqrt_h_plus_grad_2_plus_eps * grad;

        param.set_data(param.get_data() - update_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rand::prelude::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use std::cell::RefCell;
    use std::f64::consts::PI;
    use std::rc::Rc;

    /// update_one のテスト
    #[test]
    fn test_ada_grad_update_one_01() {
        let lr = rand::random::<f64>();
        let eps = rand::random::<f64>();
        let value = rand::random::<f64>();
        let grad = rand::random::<f64>();
        let expect = value - (lr * grad) / ((grad * grad).sqrt() + eps);

        let mut ada_grad = AdaGrad::new(lr, eps);

        let var = Variable::new(RawData::new(value));
        var.set_grad(Variable::new(RawData::new(grad)));
        ada_grad.update_one(&mut var.clone());

        assert!((expect - var.get_data().flatten().to_vec()[0]).abs() < 1e-10);

        for value in ada_grad.hs.values().enumerate() {
            let diff = ((grad * grad) - value.1.flatten().to_vec()[0]).abs();
            assert!(diff < 1e-10);
        }
    }

    #[test]
    fn test_mlp_ada_grad_sin_10_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // 学習用の入力値(X)
        let x_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![100, 1],
            x_var.flatten().to_vec(),
        ));
        x.set_name("x".to_string());

        // 学習用の出力値(Y = sin(2 * Pi * x) + random)
        let y_array = (2.0 * PI * x_var.clone()).sin()
            + Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let y = Variable::new(RawData::from_shape_vec(
            y_array.shape().to_vec(),
            y_array.flatten().to_vec(),
        ));
        y.set_name("y".to_string());

        let lr = 0.1;
        let eps = 1e-8;
        let iters = 10000;

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let ada_grad = AdaGrad::new(lr, eps);
        let mut mlp = Mlp::new(vec![10, 1], sigmoid, ada_grad);

        // 学習
        for i in 0..iters {
            let y_pred = mlp.forward(vec![x.clone()]);
            let loss = mean_squared_error(y.clone(), y_pred[0].clone());
            mlp.cleargrads();
            loss.backward();

            mlp.update_parameters();

            // 学習過程の確認
            if i % 1000 == 0 {
                println!("[{}] loss: {:?}", i, loss.get_data());

                let plot_x = x_var.flatten().to_vec();
                let plot_y = y_array.flatten().to_vec();

                // 推論
                let test_x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
                let test_x_var =
                    Variable::new(RawData::from_shape_vec(vec![100, 1], test_x.clone()));

                let test_y_pred = mlp.forward(vec![test_x_var.clone()])[0]
                    .get_data()
                    .flatten()
                    .to_vec();

                let mut test_xy = vec![];
                for (i, tmp_x) in test_x.iter().enumerate() {
                    test_xy.push((*tmp_x, test_y_pred[i]));
                }
                utils::draw_graph(
                    "y=sin(2 * pi * x) + b",
                    &format!("graph/step45_mlp_ada_grad_sin_pred_10_1_{}.png", i),
                    (0.0, 1.0),
                    (-1.0, 2.0),
                    plot_x,
                    plot_y,
                    test_xy,
                    &format!("loss: {}", loss.get_data().flatten().to_vec()[0]),
                );
            }
        }
    }
}
