// ライブラリを一括でインポート
use crate::modules::core::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::Array;
use rand_isaac::Isaac64Rng;
use std::collections::HashMap;

use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, SeedableRng};

/// linear 関数
#[derive(Debug, Clone)]
pub struct LinearLayer<V: MathOps> {
    in_size: Option<usize>,
    out_size: usize,
    nobias: bool,
    parameters: HashMap<String, Variable<V>>,
}

// レイヤの実装
impl<V: MathOps> Layer<V> for LinearLayer<V> {
    /// 順伝播
    ///
    /// Arguments
    /// * inputs (Vec<Variable<V>>): 入力値
    ///
    /// Outputs
    /// * Vec<Variable<V>>: 出力値
    fn forward(&mut self, inputs: Vec<Variable<V>>) -> Vec<Variable<V>> {
        // パラメータとして重みが設定されていない場合、ランダムに生成する。
        let param_w = self.parameters.get("w");
        if param_w.is_none() {
            let seed = 0;
            let mut rng = Isaac64Rng::seed_from_u64(seed);
            let in_size = inputs[0].get_data().shape().to_vec()[1];
            self.in_size = Some(in_size);
            let w_data_array =
                (Array::random_using((in_size, self.out_size), Uniform::new(0., 1.), &mut rng)
                    * (1.0 / in_size as f64).sqrt())
                .mapv(|x| V::from(x).unwrap());

            let w_data = Variable::new(RawData::from_shape_vec(
                w_data_array.shape().to_vec(),
                w_data_array.flatten().to_vec(),
            ));

            self.parameters.insert("w".to_string(), w_data.clone());
        }

        // バイアスの有無に応じて順伝播を実行する。
        if let Some(b) = self.parameters.get("b") {
            let output = linear(
                inputs[0].clone(),
                self.parameters.get("w").unwrap().clone(),
                Some(b.clone()),
            );
            vec![output.clone()]
        } else {
            let output = linear(
                inputs[0].clone(),
                self.parameters.get("w").unwrap().clone(),
                None,
            );
            vec![output.clone()]
        }
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
        for (name, parameter) in self.parameters.iter_mut() {
            parameter.clear_grad();
        }
    }
}

/// レイヤ用線形変換
impl<V: MathOps> LinearLayer<V> {
    /// レイヤ用線形変換のコンストラクタ。
    ///
    /// Arguments
    /// * in_size (Option<usize>): 入力サイズ。設定しない場合、入力値のサイズを順伝播時に設定する。
    /// * out_size (usize): 出力サイズ。
    /// * nobias (bool): バイアスを使用するかどうか。
    pub fn new(in_size: Option<usize>, out_size: usize, nobias: bool) -> LinearLayer<V> {
        let mut parameters_map = HashMap::new();
        // 入力サイズが指定されている場合、そのサイズで重みを初期化する。
        if let Some(in_size) = in_size {
            let seed = 0;
            let mut rng = Isaac64Rng::seed_from_u64(seed);
            let w_data_array =
                (Array::random_using((in_size, out_size), Uniform::new(0., 1.), &mut rng)
                    * (1.0 / in_size as f64).sqrt())
                .mapv(|x| V::from(x).unwrap());
            let w_data = Variable::new(RawData::from_shape_vec(
                w_data_array.shape().to_vec(),
                w_data_array.flatten().to_vec(),
            ));
            parameters_map.insert("w".to_string(), w_data);
        }

        // バイアスを使用する場合、初期化する。
        if nobias == false {
            let b = Variable::new(RawData::from_vec(
                std::iter::repeat(V::zero()).take(out_size).collect(),
            ));
            parameters_map.insert("b".to_string(), b);
        }

        LinearLayer {
            in_size: in_size,
            out_size: out_size,
            nobias: nobias,
            parameters: parameters_map,
        }
    }
}
