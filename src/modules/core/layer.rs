// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Layer トレイト
pub trait Layer<V>: std::fmt::Debug
where
    V: MathOps,
{
    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * inputs (Vec<Variable<V>>): 入力値
    ///
    /// Returns
    /// * Vec<Variable<V>>: 出力値
    fn forward(&self, inputs: Vec<Variable<V>>) -> Vec<Variable<V>>;
}

/// レイヤ用ラッパー
/// レイヤの入出力やパラメータなどの値を保持する。
#[derive(Debug, Clone)]
pub struct LayerExecutor<V: MathOps> {
    inputs: Vec<Weak<RefCell<Variable<V>>>>,  // 関数の入力値
    outputs: Vec<Weak<RefCell<Variable<V>>>>, //関数の出力値
    parameters: HashMap<String, Parameter<V>>,
}

/// レイヤラッパーの実装
impl<V: MathOps> LayerExecutor<V> {
    /// コンストラクタ
    ///
    ///
    /// Return
    /// * LayerExecutor: 関数のラッパー
    pub fn new() -> LayerExecutor<V> {
        LayerExecutor {
            inputs: vec![],
            outputs: vec![],
            parameters: HashMap::new(),
        }
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
    /// * parameter (Parameter<V>): パラメータ
    pub fn add_parameter(&mut self, name: String, parameter: Parameter<V>) {
        self.parameters.insert(name, parameter);
    }

    /// パラメータを取得する。
    ///
    /// Arguments
    /// * name (&str): 取得するパラメータの名前
    ///
    /// Return
    /// * &Parameter<V>: パラメータ
    pub fn get_parameter(&self, name: &str) -> &Parameter<V> {
        self.parameters.get(name).unwrap()
    }

    /// パラメータの勾配をクリアする。
    pub fn cleargrads(&mut self) {
        for (name, parameter) in self.parameters.iter_mut() {
            parameter.clear_grad();
        }
    }
}
