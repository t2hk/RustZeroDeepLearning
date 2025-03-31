// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::Mul;
use std::rc::Rc;

/// 乗算関数
#[derive(Debug, Clone)]
pub struct MulFunction;
impl<V: MathOps> Function<V> for MulFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Mul".to_string()
    }

    // Mul (乗算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] * &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x1 * x2 の微分であるため、dy/dx1=x2 * gy, dy/dx2= x1 * gy である。
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let x0 = inputs[0].borrow().get_data();
        let x1 = inputs[1].borrow().get_data();
        let gx_x0 = &gys[0].clone() * &x1;
        let gx_x1 = &gys[0].clone() * &x0;

        let gxs = vec![gx_x0, gx_x1];
        gxs
    }
}

/// 乗算関数
///
/// Arguments
/// * x0 (Rc<RefCell<Variable>>): 乗算する変数
/// * x1 (Rc<RefCell<Variable>>): 乗算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 乗算結果
pub fn mul<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    let mut mul = FunctionExecutor::new(Rc::new(RefCell::new(MulFunction)));
    // 乗算の順伝播
    mul.forward(vec![x0.clone(), x1.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// 乗算のオーバーロード (Variable<V> * Variable<V>)
///
/// Arguments
/// * self (Variable<V>): 左オペランド
/// * rhs (Variable<V>): 右オペランド
///
/// Returns
/// * Variable<V>: 乗算結果
impl<V: MathOps> Mul<&Variable<V>> for &Variable<V> {
    type Output = Variable<V>;
    fn mul(self, rhs: &Variable<V>) -> Variable<V> {
        // 順伝播
        let mut mul = FunctionExecutor::new(Rc::new(RefCell::new(MulFunction)));
        let result = mul
            .forward(vec![self.clone(), rhs.clone()])
            .get(0)
            .unwrap()
            .clone();
        result
    }
}

/// 乗算のオーバーロード (Variable<V> * Array)
impl<V: MathOps> Mul<&Array<V, IxDyn>> for &Variable<V> {
    type Output = Variable<V>;
    fn mul(self, rhs: &Array<V, IxDyn>) -> Variable<V> {
        // 順伝播
        let rhs_val = Variable::new(RawVariable::new(rhs.clone()));
        self * &rhs_val
    }
}

/// 乗算のオーバーロード (Array * Variable<V>)
impl<V: MathOps> Mul<&Variable<V>> for &Array<V, IxDyn> {
    type Output = Variable<V>;
    fn mul(self, rhs: &Variable<V>) -> Variable<V> {
        // 順伝播
        let lhs_val = Variable::new(RawVariable::new(self.clone()));
        &lhs_val * rhs
    }
}

/// Variable と様々な数値とのオーバーロード用のマクロ
macro_rules! impl_variable_mul {
    ($scalar:ty) => {
        // Variable<V> * $scalar
        impl<V: MathOps> Mul<$scalar> for &Variable<V> {
            type Output = Variable<V>;

            fn mul(self, rhs: $scalar) -> Variable<V> {
                // 順伝播
                let rhs_val = Variable::new(RawVariable::new(V::from(rhs).unwrap()));
                self * &rhs_val
            }
        }

        // $scalar * Variable<V>
        impl<V: MathOps> Mul<&Variable<V>> for $scalar {
            type Output = Variable<V>;

            fn mul(self, rhs: &Variable<V>) -> Variable<V> {
                // 順伝播
                let lhs_val = Variable::new(RawVariable::new(V::from(self).unwrap()));
                &lhs_val * rhs
            }
        }
    };
}

// 複数の数値型に対して一括実装
impl_variable_mul!(i32);
impl_variable_mul!(i64);
impl_variable_mul!(f32);
impl_variable_mul!(f64);
impl_variable_mul!(u32);
impl_variable_mul!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    #[test]
    /// 乗算のテスト(f32)
    fn test_mul_2() {
        // 順伝播
        let x1 = Variable::new(RawVariable::new(5.0f32));
        let x2 = Variable::new(RawVariable::new(10.0f32));
        let expected = RawVariable::new(50.0f32);

        let result = mul(x1, x2);
        assert_eq!(expected.get_data(), result.borrow().get_data());
    }

    #[test]
    /// 乗算のテスト(i32)
    fn test_mul_1() {
        // 順伝播
        let x1 = Variable::new(RawVariable::new(5i32));
        let x2 = Variable::new(RawVariable::new(10i32));
        let expected = RawVariable::new(50);

        let result = mul(x1, x2);
        assert_eq!(expected.get_data(), result.borrow().get_data());
    }

    /// オーバーロードのテスト
    #[test]
    fn test_add_mul_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 変数を用意する。
        let mut raw_a = RawVariable::new(3.0f32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        let mut raw_b = RawVariable::new(2.0f32);
        raw_b.set_name("val_b".to_string());
        let b = Variable::new(raw_b);
        let mut raw_c = RawVariable::new(1.0f32);
        raw_c.set_name("val_c".to_string());
        let c = Variable::new(raw_c);

        // 計算する。a * b + c
        let result = &(&a * &b) + &c;

        let expected = RawVariable::new(7.0f32);

        // 逆伝播を実行する。
        result.backward();

        println!(
            "result grad: {:?}, a grad: {:?}, b grad: {:?}, c grad: {:?}",
            &result.borrow().get_grad(),
            // &a.borrow().get_grad(),
            &a.borrow().get_grad(),
            &b.borrow().get_grad(),
            &c.borrow().get_grad(),
        );

        assert_eq!(expected.get_data(), result.borrow().get_data());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1.0),
            result.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2.0),
            a.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 3.0),
            b.borrow().get_grad().expect("No grad exist.")
        );
    }

    /// オーバーロードのテスト (Variable 以外との計算)
    #[test]
    fn test_add_mul_other_than_variable_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 変数を用意する。
        let mut raw_a = RawVariable::new(3i32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        let mut raw_b = RawVariable::new(2i32);
        raw_b.set_name("val_b".to_string());
        let b = Variable::new(raw_b);

        // c は Variable ではなく i32 として計算する。
        let c = 1i32;

        // 計算する。a * b + c
        let result = &(&a * &b) + c;

        let expected = RawVariable::new(7i32);

        // 逆伝播を実行する。
        result.backward();

        println!(
            "result grad: {:?}, a grad: {:?}, b grad: {:?}",
            &result.borrow().get_grad(),
            // &a.borrow().get_grad(),
            &a.borrow().get_grad(),
            &b.borrow().get_grad(),
            // &c.borrow().get_grad(),
        );

        assert_eq!(expected.get_data(), result.borrow().get_data());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1),
            result.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2),
            a.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 3),
            b.borrow().get_grad().expect("No grad exist.")
        );
    }

    /// オーバーロードのテスト (Array との計算)
    #[test]
    fn test_add_mul_with_array_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 変数を用意する。
        let mut raw_a = RawVariable::new(3i32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        // let mut raw_b = RawVariable::new(2i32);
        // raw_b.set_name("val_b".to_string());
        // let b = Variable::new(raw_b);

        // b は Array とする。
        let b = Array::from_elem(IxDyn(&[]), 2i32);

        let mut raw_c = RawVariable::new(1i32);
        raw_c.set_name("val_c".to_string());
        let c = Variable::new(raw_c);

        // 計算する。a * b + c
        let result = &(&a * &b) + &c;

        let expected = RawVariable::new(7i32);

        // 逆伝播を実行する。
        result.backward();

        println!(
            "result grad: {:?}, a grad: {:?}, c grad: {:?}",
            &result.borrow().get_grad(),
            // &a.borrow().get_grad(),
            &a.borrow().get_grad(),
            // &b.borrow().get_grad(),
            &c.borrow().get_grad(),
        );

        assert_eq!(expected.get_data(), result.borrow().get_data());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1),
            result.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2),
            a.borrow().get_grad().expect("No grad exist.")
        );
        // assert_eq!(
        //     Array::from_elem(IxDyn(&[]), 3),
        //     b.borrow().get_grad().expect("No grad exist.")
        // );
    }

    /// 乗算オーバーロードのテスト
    /// 様々な型、および、左右オペランドを入れ替えたテスト
    #[test]
    fn test_mul_overload_macro() {
        let overload_val_i32 = Variable::new(RawVariable::new(2i32));
        let overload_val_f32 = Variable::new(RawVariable::new(2.0f32));
        let overload_val_f64 = Variable::new(RawVariable::new(2.0f64));
        let overload_val_u32 = Variable::new(RawVariable::new(2u32));
        let overload_array_f32 = Array::from_elem(IxDyn(&[]), 2.0f32);

        let result_val_i32_mul_val_i32 = &overload_val_i32 * &overload_val_i32;
        let result_val_u32_mul_scalar_u32 = &overload_val_u32 * 10u32;
        let result_scalar_f64_mul_val_f64 = 10.0f64 * &overload_val_f64;
        let result_val_f32_mul_array_f32 = &overload_val_f32 * &overload_array_f32;
        let result_array_f32_mul_val_f32 = &overload_array_f32 * &overload_val_f32;

        assert_eq!(
            RawVariable::new(4i32).get_data(),
            result_val_i32_mul_val_i32.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(20u32).get_data(),
            result_val_u32_mul_scalar_u32.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(20.0f64).get_data(),
            result_scalar_f64_mul_val_f64.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(4.0f32).get_data(),
            result_val_f32_mul_array_f32.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(4.0f32).get_data(),
            result_array_f32_mul_val_f32.borrow().get_data()
        );
    }
}
