// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// SGD 構造体
#[derive(Debug, Clone)]
pub struct Sgd {
    lr: f64, // 学習係数
}

impl Sgd {
    /// オプティマイザ SGD を初期化する。
    ///
    /// Arguments
    /// * lr (f64): 学習率
    ///
    /// Return
    /// * Sgd: Sgd 構造体のインスタンス
    pub fn new(lr: f64) -> Self {
        Sgd { lr: lr }
    }
}

/// SGD の Optimizer トレイト実装
impl Optimizer for Sgd {
    /// 勾配を更新する。
    ///
    /// Argumesnts
    /// * param (Variable<V>): 更新対象の変数
    fn update_one<V: MathOps>(&self, param: &mut Variable<V>) {
        let new_data = param.get_data().mapv(|x| x.to_f64().unwrap())
            - param
                .get_grad()
                .unwrap()
                .get_data()
                .mapv(|x| x.to_f64().unwrap())
                * self.lr;
        param.set_data(new_data.mapv(|x| V::from(x).unwrap()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// update_one のテスト
    #[test]
    fn test_sgd_update_one_01() {
        let lr = rand::random::<f64>();
        let value = rand::random::<f64>();
        let grad = rand::random::<f64>();

        let expect = value - lr * grad;

        let sgd = Sgd::new(lr);
        let mut var = Variable::new(RawData::new(value));
        var.set_grad(Variable::new(RawData::new(grad)));
        sgd.update_one(&mut var.clone());

        assert_eq!(expect, var.get_data().flatten().to_vec()[0]);
    }
}
