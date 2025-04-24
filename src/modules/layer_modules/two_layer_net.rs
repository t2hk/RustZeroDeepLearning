// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Two Layer Net 関数
#[derive(Debug, Clone)]
pub struct TwoLayerNet {
    layer_model: LayerModel<f64>,
    parameters: HashMap<String, Variable<f64>>,
}

impl Layer<f64> for TwoLayerNet {
    fn forward(&mut self, inputs: Vec<Variable<f64>>) -> Vec<Variable<f64>> {
        let y0 = self.layer_model.forward("l1", inputs);
        let y1 = sigmoid(y0[0].clone());
        let y = self.layer_model.forward("l2", vec![y1.clone()]);

        y
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
    }
}

impl TwoLayerNet {
    pub fn new(hidden_size: usize, out_size: usize) -> TwoLayerNet {
        let mut layer_model: LayerModel<f64> = LayerModel::new();

        // 1層目
        let mut ll1: LinearLayer<f64> = LinearLayer::new(None, hidden_size, false);
        let mut l1 = LayerExecutor::new(Rc::new(RefCell::new(ll1)));

        // 2層目
        let mut ll2: LinearLayer<f64> = LinearLayer::new(None, out_size, false);
        let mut l2 = LayerExecutor::new(Rc::new(RefCell::new(ll2)));

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
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use crate::plot_dot_graph;

    use super::*;

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

        let mut tln = TwoLayerNet::new(100, 10);
        tln.plot(vec![x], "test_step45_two_layer_net.png", true);
        // let y0 = tln.forward(vec![x]);
        // let y = y0[0].clone();

        // let to_file = "test_step45_two_layer_net.png";
        // plot_dot_graph!(y, to_file, false);
    }
}
