use crate::modules::functions::*;
use crate::modules::settings::*;
use crate::modules::variable::*;
use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::{Neg, Sub};
use std::rc::Rc;

/// 減算関数
#[derive(Debug, Clone)]
pub struct SubFunction;
impl<V: MathOps> Function<V> for SubFunction {
    // Sub (減算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] - &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x0-x1 の微分であるため、dy/dx0=1, dy/dx1=-1 である。
    fn backward(
        &self,
        _inputs: Vec<Variable<V>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        vec![
            gys[0].clone(),
            gys[0].mapv(|x| V::from(-1).unwrap() * V::from(x).unwrap()),
        ]
    }
}

/// 減算関数
///
/// Arguments
/// * x0 (Variable<V>): 減算する変数
/// * x1 (Variable<V>): 減算する変数
///
/// Return
/// * Rc<RefCell<RawVariable>>: 減算結果
pub fn sub<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    let mut sub = FunctionExecutor::new(Rc::new(RefCell::new(SubFunction)));
    // 減算の順伝播
    sub.forward(vec![x0.clone(), x1.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// 減算のオーバーロード
///
/// Arguments
/// * self (Variable<V>): 左オペランド
/// * rhs (Variable<V>): 右オペランド
///
/// Returns
/// * Variable<V>: 減算結果
impl<V: MathOps> Sub<&Variable<V>> for &Variable<V> {
    type Output = Variable<V>;
    fn sub(self, rhs: &Variable<V>) -> Variable<V> {
        // 順伝播
        let mut sub = FunctionExecutor::new(Rc::new(RefCell::new(SubFunction)));
        let result = sub
            .forward(vec![self.clone(), rhs.clone()])
            .get(0)
            .unwrap()
            .clone();
        result
    }
}

/// 減算のオーバーロード (Variable<V> - Array)
impl<V: MathOps> Sub<&Array<V, IxDyn>> for &Variable<V> {
    type Output = Variable<V>;
    fn sub(self, rhs: &Array<V, IxDyn>) -> Variable<V> {
        // 順伝播
        let rhs_val = Variable::new(RawVariable::new(rhs.clone()));
        self - &rhs_val
    }
}

/// 減算のオーバーロード (Array - Variable<V>)
impl<V: MathOps> Sub<&Variable<V>> for &Array<V, IxDyn> {
    type Output = Variable<V>;
    fn sub(self, rhs: &Variable<V>) -> Variable<V> {
        // 順伝播
        let lhs_val = Variable::new(RawVariable::new(self.clone()));
        &lhs_val - rhs
    }
}

/// Variable と様々な数値とのオーバーロード用のマクロ
macro_rules! impl_variable_sub {
    ($scalar:ty) => {
        // Variable<V> - $scalar
        impl<V: MathOps> Sub<$scalar> for &Variable<V> {
            type Output = Variable<V>;

            fn sub(self, rhs: $scalar) -> Variable<V> {
                // 順伝播
                let rhs_val = Variable::new(RawVariable::new(V::from(rhs).unwrap()));
                self - &rhs_val
            }
        }

        // $scalar - Variable<V>
        impl<V: MathOps> Sub<&Variable<V>> for $scalar {
            type Output = Variable<V>;

            fn sub(self, rhs: &Variable<V>) -> Variable<V> {
                // 順伝播
                let lhs_val = Variable::new(RawVariable::new(V::from(self).unwrap()));
                &lhs_val - rhs
            }
        }
    };
}

// 複数の数値型に対して減算マクロの一括実装
impl_variable_sub!(i32);
impl_variable_sub!(i64);
impl_variable_sub!(f32);
impl_variable_sub!(f64);
impl_variable_sub!(u32);
impl_variable_sub!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// 減算のテスト
    #[test]
    fn test_sub() {
        // 減算値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x1 = rng.random::<f64>();
        let rand_x2 = rng.random::<f64>();
        let x1 = Variable::new(RawVariable::new(rand_x1));
        let x2 = Variable::new(RawVariable::new(rand_x2));

        // 減算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 - rand_x2);

        // 順伝播、逆伝播を実行する。
        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let result = sub(x1, x2);

        // 足し算の結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }

    /// オーバーロードのテスト
    #[test]
    fn test_sub_mul_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 変数を用意する。
        let mut raw_a = RawVariable::new(5.0f32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        let mut raw_b = RawVariable::new(3.0f32);
        raw_b.set_name("val_b".to_string());
        let b = Variable::new(raw_b);
        let mut raw_c = RawVariable::new(2.0f32);
        raw_c.set_name("val_c".to_string());
        let c = Variable::new(raw_c);

        // 計算する。(a - b) * c
        let result = &(&a - &b) * &c;

        let expected = RawVariable::new(4.0f32);

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
            Array::from_elem(IxDyn(&[]), -2.0),
            b.borrow().get_grad().expect("No grad exist.")
        );

        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2.0),
            c.borrow().get_grad().expect("No grad exist.")
        );
    }

    /// オーバーロードのテスト (Array との計算)
    #[test]
    fn test_add_sub_with_array_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 変数を用意する。
        let mut raw_a = RawVariable::new(5i32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        // b は Array とする。
        let b = Array::from_elem(IxDyn(&[]), 3i32);

        let mut raw_c = RawVariable::new(2i32);
        raw_c.set_name("val_c".to_string());
        let c = Variable::new(raw_c);

        // 計算する。(a - b) * c
        let result = &(&a - &b.clone()) * &c;

        let expected = RawVariable::new(4i32);

        // 逆伝播を実行する。
        result.backward();

        println!(
            "result grad: {:?}, a grad: {:?}, c grad: {:?}",
            &result.borrow().get_grad(),
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
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2),
            c.borrow().get_grad().expect("No grad exist.")
        );
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

    /// 減算オーバーロードのテスト
    /// 様々な型、および、左右オペランドを入れ替えたテスト
    #[test]
    fn test_sub_overload_macro() {
        let val_i64_5 = Variable::new(RawVariable::new(5i64));
        let val_i64_2 = Variable::new(RawVariable::new(2i64));
        let val_f32_5 = Variable::new(RawVariable::new(5.0f32));
        let val_f32_2 = Variable::new(RawVariable::new(2.0f32));
        let val_f64_2 = Variable::new(RawVariable::new(2.0f64));
        let val_u64_5 = Variable::new(RawVariable::new(5u64));
        let array_f32_5 = Array::from_elem(IxDyn(&[]), 5.0f32);
        let array_f32_2 = Array::from_elem(IxDyn(&[]), 2.0f32);

        let result_val_i64_sub = &val_i64_5 - &val_i64_2;
        let result_val_u64_sub_scalar_u64 = &val_u64_5 - 3u64;
        let result_scalar_f64_sub_val_f64 = 10.0f64 - &val_f64_2;
        let result_val_f32_sub_array_f32 = &val_f32_5 - &array_f32_2;
        let result_array_f32_sub_val_f32 = &array_f32_5 - &val_f32_2;

        assert_eq!(
            RawVariable::new(3i64).get_data(),
            result_val_i64_sub.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(2u64).get_data(),
            result_val_u64_sub_scalar_u64.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(8.0f64).get_data(),
            result_scalar_f64_sub_val_f64.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(3.0f32).get_data(),
            result_val_f32_sub_array_f32.borrow().get_data()
        );

        assert_eq!(
            RawVariable::new(3.0f32).get_data(),
            result_array_f32_sub_val_f32.borrow().get_data()
        );
    }
}
