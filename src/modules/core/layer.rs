// ライブラリを一括でインポート
use crate::{modules::*, plot_dot_graph};

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
pub struct LayerModel<V: MathOps> {
    layers: HashMap<String, LayerExecutor<V>>,
    layer_models: HashMap<String, LayerModel<V>>,
}

impl<V: MathOps> LayerModel<V> {
    /// コンストラクタ
    pub fn new() -> LayerModel<V> {
        LayerModel {
            layers: HashMap::new(),
            layer_models: HashMap::new(),
        }
    }

    /// レイヤーを追加する。
    pub fn add_layer(&mut self, name: &str, layer: LayerExecutor<V>) {
        self.layers.insert(name.to_string(), layer);
    }

    /// レイヤーを取得する。
    pub fn get_layer(&self, name: &str) -> &LayerExecutor<V> {
        self.layers.get(&name.to_string()).unwrap()
    }

    /// レイヤーモデルを追加する。
    pub fn add_layer_model(&mut self, name: &str, layer_model: LayerModel<V>) {
        self.layer_models.insert(name.to_string(), layer_model);
    }

    /// レイヤーモデルを取得する。
    pub fn get_layer_model(&self, name: &str) -> &LayerModel<V> {
        self.layer_models.get(&name.to_string()).unwrap()
    }

    /// 内包するレイヤー、及び レイヤーモデルの全ての勾配をクリアする。
    pub fn cleargrads(&mut self) {
        for (_layer_name, layer) in self.layers.iter() {
            for (_param_name, layer_param) in layer.get_parameters().iter() {
                layer_param.clear_grad();
            }
        }
        for (_layer_model_name, layer_model) in self.layer_models.iter_mut() {
            layer_model.cleargrads();
        }
    }

    /// 順伝播
    pub fn forward(&mut self, layer_name: &str, x: Vec<Variable<V>>) -> Vec<Variable<V>> {
        let mut layer = self.layers.get(&layer_name.to_string()).unwrap().clone();
        layer.forward(x)
    }

    /// パラメータの勾配を更新する。
    pub fn update_parameters(&mut self, lr: f64) {
        for (_layer_name, layer) in self.layers.iter() {
            for (_param_name, layer_param) in layer.get_parameters().iter_mut() {
                let new_data = layer_param.get_data().mapv(|x| x.to_f64().unwrap())
                    - layer_param
                        .get_grad()
                        .unwrap()
                        .get_data()
                        .mapv(|x| x.to_f64().unwrap())
                        * lr;
                layer_param.set_data(new_data.mapv(|x| V::from(x).unwrap()));
            }
        }
    }
}

/// Layer トレイト
pub trait Layer<V>: std::fmt::Debug
where
    V: MathOps,
{
    /// 順伝播
    fn forward(&mut self, inputs: Vec<Variable<V>>) -> Vec<Variable<V>>;

    /// パラメータを追加する。
    fn add_parameter(&mut self, name: &str, parameter: Variable<V>);

    /// パラメータを取得する。
    fn get_parameter(&self, name: &str) -> Variable<V>;

    /// 全てのパラメータを取得する。
    fn get_parameters(&self) -> HashMap<String, Variable<V>>;

    /// パラメータの勾配をクリアする。
    fn cleargrads(&mut self);

    /// 計算グラフを描画する。
    ///
    /// Arguments
    /// * x (Vec<Variable<V>): 入力値
    /// * to_file (&str): ファイル名(png)
    /// * detail (bool): 詳細を出力するかどうか
    fn plot(&mut self, x: Vec<Variable<V>>, to_file: &str, detail: bool) {
        let y = self.forward(x.clone());
        let y_tmp = y[0].clone();
        if detail {
            plot_dot_graph!(y_tmp, to_file, true);
        } else {
            plot_dot_graph!(y_tmp, to_file, false);
        }
    }
}

/// レイヤ用ラッパー
/// レイヤの入出力やパラメータなどの値を保持する。
#[derive(Debug, Clone)]
pub struct LayerExecutor<V: MathOps> {
    inputs: Vec<Weak<RefCell<Variable<V>>>>,   // 関数の入力値
    outputs: Vec<Weak<RefCell<Variable<V>>>>,  //関数の出力値
    layer_function: Rc<RefCell<dyn Layer<V>>>, // レイヤー関数のトレイトオブジェクト
}

/// レイヤラッパーの実装
impl<V: MathOps> LayerExecutor<V> {
    /// コンストラクタ
    ///
    /// Arguments
    /// * layer_function (Rc<RefCell<dyn Layer<V>>>): 実行するレイヤー関数
    ///
    /// Return
    /// * LayerExecutor: 関数のラッパー
    pub fn new(layer_function: Rc<RefCell<dyn Layer<V>>>) -> LayerExecutor<V> {
        LayerExecutor {
            inputs: vec![],
            outputs: vec![],
            layer_function: layer_function,
        }
    }

    /// 引数をレイヤー構造体に設定する。
    ///
    /// Arguments
    /// * inputs: Vec<Variable<V>>
    pub fn set_inputs(&mut self, inputs: Vec<Variable<V>>) {
        self.inputs = inputs
            .clone()
            .iter()
            .map(|input| Rc::downgrade(&Rc::new(RefCell::new(input.clone()))))
            .collect();
    }

    /// 処理結果をレイヤー構造体に設定する。
    ///
    /// Arguments
    /// * outputs: Vec<Variable<V>>
    pub fn set_outputs(&mut self, outputs: Vec<Variable<V>>) {
        self.outputs = outputs
            .clone()
            .iter()
            .map(|output| Rc::downgrade(&Rc::new(RefCell::new(output.clone()))))
            .collect();
    }
    /// 入力値を取得する。
    ///
    /// Return
    /// * Vec<Weak<RefCell<Variable<V>>>>: 関数に対する入力値のベクタ
    pub fn get_inputs(&self) -> Vec<Weak<RefCell<Variable<V>>>> {
        self.inputs.clone()
    }

    /// 出力値を取得する。
    ///
    /// Return
    /// * Vec<Weak<RefCell<Variable<V>>>>: 関数の出力値のベクタ
    pub fn get_outputs(&self) -> Vec<Weak<RefCell<Variable<V>>>> {
        self.outputs.clone()
    }

    /// パラメータを追加する。
    ///
    /// Arguments
    /// * name (String): パラメータの名前
    /// * parameter (Variable<V>): パラメータ
    pub fn add_parameter(&mut self, name: &str, parameter: Variable<V>) {
        //self.parameters.insert(name, parameter);
        self.layer_function
            .borrow_mut()
            .add_parameter(name, parameter);
    }

    /// パラメータを取得する。
    ///
    /// Arguments
    /// * name (&str): 取得するパラメータの名前
    ///
    /// Return
    /// * &Variable<V>: パラメータ
    pub fn get_parameter(&self, name: &str) -> Variable<V> {
        self.layer_function.borrow().get_parameter(name)
    }

    /// 内包するレイヤのパラメータを全て取得する。
    pub fn get_parameters(&self) -> HashMap<String, Variable<V>> {
        self.layer_function.borrow().get_parameters()
    }

    /// パラメータの勾配をクリアする。
    pub fn cleargrads(&mut self) {
        self.layer_function.borrow_mut().cleargrads();
    }

    /// 順伝播
    ///
    /// Arguments
    /// * inputs (Vec<Variable<V>>): 入力値
    ///
    /// Returns
    /// * Vec<Variable<V>>: 出力値
    fn forward(&mut self, inputs: Vec<Variable<V>>) -> Vec<Variable<V>> {
        let outputs = self.layer_function.borrow_mut().forward(inputs.clone());
        self.set_inputs(inputs.clone());
        self.set_outputs(outputs.clone());

        outputs
    }
}

pub fn predict(model: &mut LayerModel<f64>, x: Variable<f64>) -> Vec<Variable<f64>> {
    let y1 = model.forward("l1", vec![x.clone()]);
    let y2 = sigmoid(y1[0].clone());
    let y = model.forward("l2", vec![y2.clone()]);
    y
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, IxDyn};
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use std::{env, f64::consts::PI};

    use super::*;

    #[test]
    fn test_step45() {
        let mut layer_model: LayerModel<f64> = LayerModel::<f64>::new();
        // 1層目
        let mut ll1: LinearLayer<f64> = LinearLayer::new(None, 10, false);
        let mut l1 = LayerExecutor::new(Rc::new(RefCell::new(ll1)));

        // 2層目
        let mut ll2: LinearLayer<f64> = LinearLayer::new(None, 1, false);
        let mut l2 = LayerExecutor::new(Rc::new(RefCell::new(ll2)));

        layer_model.add_layer("l1", l1.clone());
        layer_model.add_layer("l2", l2.clone());

        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // 学習用の入力値(X)
        let x_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![100, 1],
            x_var.flatten().to_vec(),
        ));

        // 学習用の出力値(Y = sin(2 * Pi * x) + random)
        let y_array = (2.0 * PI * x_var.clone()).sin()
            + Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let y = Variable::new(RawData::from_shape_vec(
            y_array.shape().to_vec(),
            y_array.flatten().to_vec(),
        ));

        let lr = 0.2;
        let iters = 10000;

        // 学習
        for i in 0..iters {
            let y_pred = predict(&mut layer_model, x.clone());
            let loss = mean_squared_error(y.clone(), y_pred[0].clone());

            layer_model.cleargrads();

            loss.backward();

            layer_model.update_parameters(lr);

            // 学習過程の確認
            if i % 1000 == 0 {
                println!("[{}] loss: {:?}", i, loss.get_data());

                let plot_x = x_var.flatten().to_vec();
                let plot_y = y_array.flatten().to_vec();

                // 推論
                let test_x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
                let test_x_var =
                    Variable::new(RawData::from_shape_vec(vec![100, 1], test_x.clone()));

                let test_y_pred = predict(&mut layer_model, test_x_var.clone())[0]
                    .get_data()
                    .flatten()
                    .to_vec();

                let mut test_xy = vec![];
                for (i, tmp_x) in test_x.iter().enumerate() {
                    test_xy.push((*tmp_x, test_y_pred[i]));
                }
                utils::draw_graph(
                    "y=sin(2 * pi * x) + b",
                    &format!("graph/step45_neural_network_pred_{}.png", i),
                    plot_x,
                    plot_y,
                    test_xy,
                    &format!("loss: {}", loss.get_data().flatten().to_vec()[0]),
                );
            }
        }
    }

    #[test]
    fn test_step44() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // 学習用の入力値(X)
        let x_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![100, 1],
            x_var.flatten().to_vec(),
        ));

        // 学習用の出力値(Y = sin(2 * Pi * x) + random)
        let y_array = (2.0 * PI * x_var.clone()).sin()
            + Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let y = Variable::new(RawData::from_shape_vec(
            y_array.shape().to_vec(),
            y_array.flatten().to_vec(),
        ));

        // 1層目
        let mut ll1: LinearLayer<f64> = LinearLayer::new(None, 10, false);
        let mut l1 = LayerExecutor::new(Rc::new(RefCell::new(ll1)));

        // 2層目
        let mut ll2: LinearLayer<f64> = LinearLayer::new(None, 1, false);
        let mut l2 = LayerExecutor::new(Rc::new(RefCell::new(ll2)));

        let lr = 0.2;
        let iters = 10000;

        // 学習
        for i in 0..iters {
            let y1 = l1.forward(vec![x.clone()]);
            let y2 = sigmoid(y1[0].clone());
            let y_pred = l2.forward(vec![y2.clone()]);
            let loss = mean_squared_error(y.clone(), y_pred[0].clone());

            l1.cleargrads();
            l2.cleargrads();

            loss.backward();

            for l in [l1.clone(), l2.clone()] {
                let params = l.get_parameters();
                for (_name, param) in params {
                    let new_data = param.get_data() - param.get_grad().unwrap().get_data() * lr;
                    param.set_data(new_data);
                }
            }

            // 学習過程の確認
            if i % 1000 == 0 {
                println!("[{}] loss: {:?}", i, loss.get_data());

                let plot_x = x_var.flatten().to_vec();
                let plot_y = y_array.flatten().to_vec();

                // 推論
                let test_x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
                let test_x_var =
                    Variable::new(RawData::from_shape_vec(vec![100, 1], test_x.clone()));

                let test_y1 = l1.forward(vec![test_x_var.clone()]);
                let test_y2 = sigmoid(test_y1[0].clone());
                let test_y_pred = l2.forward(vec![test_y2.clone()])[0]
                    .get_data()
                    .flatten()
                    .to_vec();

                let mut test_xy = vec![];
                for (i, tmp_x) in test_x.iter().enumerate() {
                    test_xy.push((*tmp_x, test_y_pred[i]));
                }
                utils::draw_graph(
                    "y=sin(2 * pi * x) + b",
                    &format!("graph/step44_neural_network_pred_{}.png", i),
                    plot_x,
                    plot_y,
                    test_xy,
                    &format!("loss: {}", loss.get_data().flatten().to_vec()[0]),
                );
            }
        }
    }
}
