// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
/// MomentumSgd 構造体
#[derive(Debug, Clone)]
pub struct MomentumSgd<V> {
    lr: f64, // 学習係数
    momentum: f64,
    vs: HashMap<String, Array<V, IxDyn>>,
}

impl<V> MomentumSgd<V> {
    /// オプティマイザ SGD を初期化する。
    ///
    /// Arguments
    /// * lr (f64): 学習率
    ///
    /// Return
    /// * MomentumSgd: MomentumSgd 構造体のインスタンス
    pub fn new(lr: f64, momentum: f64) -> Self {
        MomentumSgd {
            lr: lr,
            momentum: momentum,
            vs: HashMap::new(),
        }
    }
}

/// MomentumSgd の Optimizer トレイト実装
impl<V: MathOps> Optimizer for MomentumSgd<V> {
    type Val = V;

    /// 勾配を更新する。
    ///
    /// Argumesnts
    /// * param (Variable<V>): 更新対象の変数
    fn update_one(&mut self, param: &mut Variable<V>) {
        let param_id = format!("{:?}", param.as_ref().as_ptr());

        if self.vs.get(&param_id).is_none() {
            let zero_array = Array::zeros(param.get_data().shape());
            self.vs.insert(param_id.clone(), zero_array);
        }

        let v = self.vs.get(&param_id).unwrap();
        let v_momentum = &v.mapv(|x| V::from(x.to_f64().unwrap() * self.momentum).unwrap());

        let lr_grad = param
            .get_grad()
            .unwrap()
            .get_data()
            .mapv(|x| V::from(x.to_f64().unwrap() * self.lr).unwrap());

        let update_value = &(v_momentum - lr_grad);

        param.set_data(param.get_data() + update_value);
        self.vs.insert(param_id.clone(), update_value.clone());
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
    fn test_memoentum_sgd_update_one_01() {
        let lr = rand::random::<f64>();
        let value = rand::random::<f64>();
        let grad = rand::random::<f64>();
        let expect = value - lr * grad;

        let mut mementum_sgd = MomentumSgd::new(lr, 0.9);
        let var = Variable::new(RawData::new(value));
        var.set_grad(Variable::new(RawData::new(grad)));
        mementum_sgd.update_one(&mut var.clone());

        assert_eq!(expect, var.get_data().flatten().to_vec()[0]);
    }

    #[test]
    fn test_mlp_momentum_sgd_sin_10_1() {
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

        let lr = 0.2;
        let momentum = 0.9;
        let iters = 10000;

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let momentum_sgd = MomentumSgd::new(lr, momentum);
        let mut mlp = Mlp::new(vec![10, 1], sigmoid, momentum_sgd);

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
                    &format!("graph/step45_mlp_momentum_sgd_sin_pred_10_1_{}.png", i),
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
