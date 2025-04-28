// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Two Layer Net 関数
#[derive(Debug, Clone)]
pub struct TwoLayerNet<V, O>
where
    V: MathOps + 'static,
{
    layer_model: LayerModel<V, O>,
    parameters: HashMap<String, Variable<V>>,
}

impl<V: MathOps, O: Optimizer<Val = V> + std::fmt::Debug> Layer<V> for TwoLayerNet<V, O> {
    fn forward(&mut self, inputs: Vec<Variable<V>>) -> Vec<Variable<V>> {
        let y0 = self.layer_model.forward("l1", inputs);
        let y1 = sigmoid(y0[0].clone());
        let y = self.layer_model.forward("l2", vec![y1.clone()]);

        y
    }

    /// パラメータを追加する。
    fn add_parameter(&mut self, name: &str, parameter: Variable<V>) {
        self.parameters.insert(name.to_string(), parameter);
    }

    /// パラメータを取得する。
    fn get_parameter(&self, name: &str) -> Variable<V> {
        self.parameters.get(&name.to_string()).unwrap().clone()
    }

    /// 全てのパラメータを取得する。
    fn get_parameters(&self) -> HashMap<String, Variable<V>> {
        self.parameters.clone()
    }

    /// パラメータの勾配をクリアする。
    fn cleargrads(&mut self) {
        for (_name, parameter) in self.parameters.iter_mut() {
            parameter.clear_grad();
        }
        self.layer_model.cleargrads();
    }

    /// パラメータを更新する
    fn update_parameters(&mut self) {
        self.layer_model.update_parameters();
    }
}

impl<V: MathOps, O: Optimizer<Val = V>> TwoLayerNet<V, O> {
    pub fn new(hidden_size: usize, out_size: usize, optimizer: O) -> TwoLayerNet<V, O> {
        let mut layer_model: LayerModel<V, O> = LayerModel::new();
        layer_model.set_optimizer(optimizer);

        // 1層目
        let ll1: LinearLayer<V> = LinearLayer::new(None, hidden_size, false);
        let l1 = LayerExecutor::new(Rc::new(RefCell::new(ll1)));

        // 2層目
        let ll2: LinearLayer<V> = LinearLayer::new(None, out_size, false);
        let l2 = LayerExecutor::new(Rc::new(RefCell::new(ll2)));

        layer_model.add_layer("l1", l1.clone());
        layer_model.add_layer("l2", l2.clone());

        TwoLayerNet {
            layer_model: layer_model,
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

    /// Two Layer Net による計算グラフの描画テスト
    #[test]
    fn test_two_layer_net() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((5, 10), Uniform::new(0., 1.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![5, 10],
            x_var.flatten().to_vec(),
        ));
        x.set_name("x".to_string());

        let sgd = Sgd::new(0.2);
        let mut tln = TwoLayerNet::new(100, 10, sgd);
        tln.plot(vec![x], "test_step45_two_layer_net.png", true);
    }

    /// Two Layer Net のモデルによる Sin の学習と推論
    #[test]
    fn test_two_layer_net_sin() {
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

        let iters = 10000;
        let hidden_size = 10;

        let sgd = Sgd::new(0.2);
        let mut tln = TwoLayerNet::new(hidden_size, 1, sgd);

        // 学習
        for i in 0..iters {
            let y_pred = tln.forward(vec![x.clone()]);
            let loss = mean_squared_error(y.clone(), y_pred[0].clone());
            tln.cleargrads();
            loss.backward();

            tln.update_parameters();

            // 学習過程の確認
            if i % 1000 == 0 {
                println!("[{}] loss: {:?}", i, loss.get_data());

                let plot_x = x_var.flatten().to_vec();
                let plot_y = y_array.flatten().to_vec();

                // 推論
                let test_x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
                let test_x_var =
                    Variable::new(RawData::from_shape_vec(vec![100, 1], test_x.clone()));

                let test_y_pred = tln.forward(vec![test_x_var.clone()])[0]
                    .get_data()
                    .flatten()
                    .to_vec();

                let mut test_xy = vec![];
                for (i, tmp_x) in test_x.iter().enumerate() {
                    test_xy.push((*tmp_x, test_y_pred[i]));
                }
                utils::draw_graph(
                    "y=sin(2 * pi * x) + b",
                    &format!("graph/step45_two_layer_net_sin_pred_{}.png", i),
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
