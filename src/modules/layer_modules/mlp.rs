// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Two Layer Net 関数
#[derive(Debug, Clone)]
pub struct Mlp<O> {
    layer_model: LayerModel<f64, O>,
    activation: Rc<RefCell<dyn Function<f64>>>,
    parameters: HashMap<String, Variable<f64>>,
}

impl<O: Optimizer + std::fmt::Debug> Layer<f64> for Mlp<O> {
    /// 順伝播
    ///
    /// Arguments
    /// * inputs (Vec<Variable<f64>>): 入力値
    ///
    /// Retrun
    /// * Vec<Variable<f64>>: 結果
    fn forward(&mut self, inputs: Vec<Variable<f64>>) -> Vec<Variable<f64>> {
        let layers = self.layer_model.get_layers();

        let mut exec_activate = FunctionExecutor::new(self.activation.clone());

        let mut x = inputs.clone();

        for (idx, key) in layers.iter().enumerate() {
            let layer = layers.get(key);

            if idx < layers.len() - 1 {
                x = self.layer_model.forward(key, x);
                x = exec_activate.forward(x);
            } else {
                x = self.layer_model.forward(key, x);
            }
        }

        x
    }

    /// パラメータを追加する。
    fn add_parameter(&mut self, name: &str, parameter: Variable<f64>) {
        self.parameters.insert(name.to_string(), parameter);
    }

    /// パラメータを取得する。
    fn get_parameter(&self, name: &str) -> Variable<f64> {
        self.parameters.get(&name.to_string()).unwrap().clone()
    }

    /// 全てのパラメータを取得する。
    fn get_parameters(&self) -> HashMap<String, Variable<f64>> {
        self.parameters.clone()
    }

    /// パラメータの勾配をクリアする。
    fn cleargrads(&mut self) {
        for (name, parameter) in self.parameters.iter_mut() {
            parameter.clear_grad();
        }
        self.layer_model.cleargrads();
    }
}

impl<O: Optimizer> Mlp<O> {
    pub fn new(
        fc_output_sizes: Vec<usize>,
        activation: Rc<RefCell<dyn Function<f64>>>,
        optimizer: O,
    ) -> Mlp<O> {
        // let sgd = Sgd::new(0.2);
        let mut layer_model: LayerModel<f64, O> = LayerModel::new();
        layer_model.set_optimizer(optimizer);

        for (idx, out_size) in fc_output_sizes.iter().enumerate() {
            let ll = LinearLayer::new(None, *out_size, false);
            let mut layer = LayerExecutor::new(Rc::new(RefCell::new(ll)));
            layer_model.add_layer(&format!("l{}", idx), layer);
        }

        Mlp {
            layer_model: layer_model,
            activation: activation,
            parameters: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use std::f64::consts::PI;

    use crate::plot_dot_graph;

    /// MLP による計算グラフの描画テスト
    #[test]
    fn test_mlp_10_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((5, 10), Uniform::new(0., 1.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![5, 10],
            x_var.flatten().to_vec(),
        ));
        x.set_name("x".to_string());

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(0.2);
        let mut mlp = Mlp::new(vec![10, 1], sigmoid, sgd);
        mlp.plot(vec![x], "test_step45_mlp_10-1.png", true);
    }

    /// MLP による計算グラフの描画テスト
    #[test]
    fn test_mlp_10_20_30_40_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((5, 10), Uniform::new(0., 1.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![5, 10],
            x_var.flatten().to_vec(),
        ));
        x.set_name("x".to_string());

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(0.2);
        let mut mlp = Mlp::new(vec![10, 20, 30, 40, 1], sigmoid, sgd);
        mlp.plot(vec![x], "test_step45_mlp_10-20-30-40-1.png", true);
    }

    /// MLP のモデルによる Sin の学習と推論
    #[test]
    fn test_mlp_sin_10_1() {
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
        let iters = 10000;
        let hidden_size = 10;

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(0.2);
        let mut mlp = Mlp::new(vec![10, 1], sigmoid, sgd);

        // 学習
        for i in 0..iters {
            let y_pred = mlp.forward(vec![x.clone()]);
            let loss = mean_squared_error(y.clone(), y_pred[0].clone());
            mlp.cleargrads();
            loss.backward();

            mlp.layer_model.update_parameters();

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
                    &format!("graph/step45_mlp_sin_pred_10_1_{}.png", i),
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

    /// MLP のモデルによる Sin の学習と推論
    #[test]
    fn test_mlp_sin_20_10_1() {
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
        let iters = 10000;

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(0.2);
        let mut mlp = Mlp::new(vec![20, 10, 1], sigmoid, sgd);

        // 学習
        for i in 0..iters {
            let y_pred = mlp.forward(vec![x.clone()]);
            let loss = mean_squared_error(y.clone(), y_pred[0].clone());
            mlp.cleargrads();
            loss.backward();

            mlp.layer_model.update_parameters();

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
                    &format!("graph/step45_mlp_sin_pred_20_10_1_{}.png", i),
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
