// ライブラリを一括でインポート
use crate::modules::math::*;

#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::Neg;
use std::rc::{Rc, Weak};

/// 負数 Neg 関数
#[derive(Debug, Clone)]
pub struct NegFunction;
impl<V: MathOps> Function<V> for NegFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Neg".to_string()
    }

    // Neg (y=-x) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("neg(forward)");
        debug!("neg(forward): -{:?}", xs[0].flatten().to_vec());
        let result = vec![xs[0].mapv(|x| V::from(-1).unwrap() * V::from(x).unwrap())];

        result
    }

    /// 逆伝播
    /// y=-x の微分 dy/dx=-1 である。
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        _outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        info!("neg(backward)");
        let x = inputs[0].get_data();
        let gys_val = gys[0].get_data();

        let x_exp = vec![x.mapv(|x| V::from(x).unwrap())];
        let gxs: Vec<Variable<V>> = x_exp.iter().map(|_x_exp| &gys[0] * -1).collect();
        debug!("neg(backward): -1 * {:?}", &gys_val.flatten().to_vec());

        gxs
    }
}

/// 負数 Neg 関数
///
/// Arguments
/// * input (Rc<RefCell<RawData>>): 入力値
///
/// Return
/// * Rc<RefCell<RawData>>: 結果
pub fn neg<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut neg = FunctionExecutor::new(Rc::new(RefCell::new(NegFunction)));
    // NEG の順伝播
    neg.forward(vec![input.clone()]).get(0).unwrap().clone()
}

/// 負数 Neg のオーバーロード (-Variable<V>)
///
/// Arguments
/// * self (Variable<V>): 左オペランド
/// * rhs (Variable<V>): 右オペランド
///
/// Returns
/// * Variable<V>: 乗算結果
impl<V: MathOps> Neg for Variable<V> {
    type Output = Variable<V>;
    fn neg(self) -> Variable<V> {
        // 順伝播
        let mut neg = FunctionExecutor::new(Rc::new(RefCell::new(NegFunction)));
        let result = neg.forward(vec![self.clone()]).get(0).unwrap().clone();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 負数 Neg に関するテスト
    #[test]
    fn test_neg_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let pos_val_i32_1 = Variable::new(RawData::new(2i32));
        let pos_val_i32_2 = Variable::new(RawData::new(3i32));
        let pos_val_i32_3 = Variable::new(RawData::new(4i32));
        let neg_val_i32 = &(&pos_val_i32_1 + &-pos_val_i32_2.clone()) + &-pos_val_i32_3.clone();

        assert_eq!(RawData::new(-5).get_data(), &neg_val_i32.get_data());

        let pos_val_f64_1 = Variable::new(RawData::new(2f64));
        let pos_val_f64_2 = Variable::new(RawData::new(3f64));
        let pos_val_f64_3 = Variable::new(RawData::new(4f64));
        let neg_val_f64 = &(&pos_val_f64_1 + &-pos_val_f64_2) + &-pos_val_f64_3;

        assert_eq!(RawData::new(-5f64).get_data(), &neg_val_f64.get_data());
    }
}
