// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

/// 加算関数
#[derive(Debug, Clone)]
pub struct AddFunction {
    x0_shape: Vec<usize>,
    x1_shape: Vec<usize>,
}

impl<V: MathOps> Function<V> for AddFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Add".to_string()
    }

    // Add (加算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("add(forward)");
        debug!(
            "add(forward) {:?} + {:?}",
            &xs[0].flatten().to_vec(),
            &xs[1].flatten().to_vec()
        );

        let result = vec![&xs[0] + &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x0+x1 の微分であるため、dy/dx0=1, dy/dx1=1 である。
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("add(backward)");

        if self.x0_shape != self.x1_shape {
            let gx0 = sum_to(gys[0].clone(), self.x0_shape.clone());
            let gx1 = sum_to(gys[0].clone(), self.x1_shape.clone());
            debug!(
                "add(backward) dy/dx0: {:?}, dy/dx1: {:?}",
                gx0.get_data().flatten().to_vec(),
                gx1.get_data().flatten().to_vec()
            );
            return vec![gx0, gx1];
        }

        let result = vec![gys[0].clone(), gys[0].clone()];
        debug!(
            "add(backward) dy/dx0: {:?}, dy/dx1: {:?}",
            result[0].get_data().flatten().to_vec(),
            result[1].get_data().flatten().to_vec()
        );
        result
    }
}

/// 加算関数
///
/// Arguments
/// * x0 (Variable<V>): 加算する変数
/// * x1 (Variable<V>): 加算する変数
///
/// Return
/// * Rc<RefCell<RawData>>: 加算結果
pub fn add<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    debug!("AddFunction::add");

    let x0_shape = x0.get_data().shape().to_vec();
    let x1_shape = x1.get_data().shape().to_vec();

    let mut add = FunctionExecutor::new(Rc::new(RefCell::new(AddFunction {
        x0_shape: x0_shape,
        x1_shape: x1_shape,
    })));
    // 加算の順伝播
    add.forward(vec![x0.clone(), x1.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// 加算のオーバーロード
///
/// Arguments
/// * self (Variable<V>): 左オペランド
/// * rhs (Variable<V>): 右オペランド
///
/// Returns
/// * Variable<V>: 加算結果
impl<V: MathOps> Add<&Variable<V>> for &Variable<V> {
    type Output = Variable<V>;
    fn add(self, rhs: &Variable<V>) -> Variable<V> {
        debug!("Add overload (Variable<V> + Variable<V>)");

        let x0_shape = self.get_data().shape().to_vec();
        let x1_shape = rhs.get_data().shape().to_vec();

        // 順伝播
        let mut add = FunctionExecutor::new(Rc::new(RefCell::new(AddFunction {
            x0_shape: x0_shape,
            x1_shape: x1_shape,
        })));
        let result = add
            .forward(vec![self.clone(), rhs.clone()])
            .get(0)
            .unwrap()
            .clone();
        result
    }
}

/// 加算のオーバーロード (Variable<V> + Array)
impl<V: MathOps> Add<&Array<V, IxDyn>> for &Variable<V> {
    type Output = Variable<V>;
    fn add(self, rhs: &Array<V, IxDyn>) -> Variable<V> {
        debug!("Add overload (Variable<V> + Array)");
        // 順伝播
        let rhs_val = Variable::new(RawData::new(rhs.clone()));
        self + &rhs_val
    }
}

/// 加算のオーバーロード (Array + Variable<V>)
impl<V: MathOps> Add<&Variable<V>> for &Array<V, IxDyn> {
    type Output = Variable<V>;
    fn add(self, rhs: &Variable<V>) -> Variable<V> {
        debug!("Add overload (Array + Variable<V>)");
        // 順伝播
        let lhs_val = Variable::new(RawData::new(self.clone()));
        &lhs_val + rhs
    }
}

/// Variable と様々な数値とのオーバーロード用のマクロ
macro_rules! impl_variable_add {
    ($scalar:ty) => {
        // Variable<V> + $scalar
        impl<V: MathOps> Add<$scalar> for &Variable<V> {
            type Output = Variable<V>;

            fn add(self, rhs: $scalar) -> Variable<V> {
                // 順伝播
                let rhs_val = Variable::new(RawData::new(V::from(rhs).unwrap()));
                self + &rhs_val
            }
        }

        // $scalar + Variable<V>
        impl<V: MathOps> Add<&Variable<V>> for $scalar {
            type Output = Variable<V>;

            fn add(self, rhs: &Variable<V>) -> Variable<V> {
                // 順伝播
                let lhs_val = Variable::new(RawData::new(V::from(self).unwrap()));
                &lhs_val + rhs
            }
        }
    };
}

// 複数の数値型に対して加算マクロの一括実装
impl_variable_add!(i32);
impl_variable_add!(i64);
impl_variable_add!(f32);
impl_variable_add!(f64);
impl_variable_add!(u32);
impl_variable_add!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, prelude::*};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1, 100), Uniform::new(0., 1.), &mut rng);
        let x1_var = Array::random_using((1, 100), Uniform::new(0., 1.), &mut rng);

        let x0 = Variable::new(RawData::from_shape_vec(
            vec![1, 100],
            x0_var.flatten().to_vec(),
        ));
        let x1 = Variable::new(RawData::from_shape_vec(
            vec![1, 100],
            x1_var.flatten().to_vec(),
        ));

        let mut add = FunctionExecutor::new(Rc::new(RefCell::new(AddFunction {
            x0_shape: vec![1, 100],
            x1_shape: vec![1, 100],
        })));

        utils::gradient_check(&mut add, vec![x0.clone(), x1.clone()]);
    }

    /// 加算のテスト
    #[test]
    fn test_add() {
        // 加算値をランダムに生成する。
        // let mut rng = rand::rng();
        // let rand_x1 = rng.random::<f64>();
        // let rand_x2 = rng.random::<f64>();
        let rand_x1 = rand::random::<f64>();
        let rand_x2 = rand::random::<f64>();
        let x1 = Variable::new(RawData::new(rand_x1));
        let x2 = Variable::new(RawData::new(rand_x2));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

        // 順伝播、逆伝播を実行する。
        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let result = add(x1, x2);

        // 足し算の結果
        assert_eq!(expected_output_data, result.get_data());
    }

    /// オーバーロードのテスト
    #[test]
    fn test_add_mul_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 変数を用意する。
        let mut raw_a = RawData::new(3.0f32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        let mut raw_b = RawData::new(2.0f32);
        raw_b.set_name("val_b".to_string());
        let b = Variable::new(raw_b);
        let mut raw_c = RawData::new(1.0f32);
        raw_c.set_name("val_c".to_string());
        let c = Variable::new(raw_c);

        // 計算する。a * b + c
        let result = &(&a * &b) + &c;

        let expected = RawData::new(7.0f32);

        // 逆伝播を実行する。
        result.backward();

        assert_eq!(expected.get_data(), result.get_data());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1.0),
            result.get_grad().expect("No grad exist.").get_data()
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2.0),
            a.get_grad().expect("No grad exist.").get_data()
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 3.0),
            b.get_grad().expect("No grad exist.").get_data()
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
        let mut raw_a = RawData::new(3i32);
        raw_a.set_name("val_a".to_string());
        let a = Variable::new(raw_a);

        // let mut raw_b = RawData::new(2i32);
        // raw_b.set_name("val_b".to_string());
        // let b = Variable::new(raw_b);

        // b は Array とする。
        let b = Array::from_elem(IxDyn(&[]), 2i32);

        let mut raw_c = RawData::new(1i32);
        raw_c.set_name("val_c".to_string());
        let c = Variable::new(raw_c);

        // 計算する。a * b + c
        let result = &(&a * &b) + &c;

        let expected = RawData::new(7i32);

        // 逆伝播を実行する。
        result.backward();

        assert_eq!(expected.get_data(), result.get_data());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1),
            result.get_grad().expect("No grad exist.").get_data()
        );
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2),
            a.get_grad().expect("No grad exist.").get_data()
        );
        // assert_eq!(
        //     Array::from_elem(IxDyn(&[]), 3),
        //     b.get_grad().expect("No grad exist.")
        // );
    }

    /// 乗算オーバーロードのテスト
    /// 様々な型、および、左右オペランドを入れ替えたテスト
    #[test]
    fn test_mul_overload_macro() {
        let overload_val_i32 = Variable::new(RawData::new(2i32));
        let overload_val_f32 = Variable::new(RawData::new(2.0f32));
        let overload_val_f64 = Variable::new(RawData::new(2.0f64));
        let overload_val_u32 = Variable::new(RawData::new(2u32));
        let overload_array_f32 = Array::from_elem(IxDyn(&[]), 2.0f32);

        let result_val_i32_mul_val_i32 = &overload_val_i32 * &overload_val_i32;
        let result_val_u32_mul_scalar_u32 = &overload_val_u32 * 10u32;
        let result_scalar_f64_mul_val_f64 = 10.0f64 * &overload_val_f64;
        let result_val_f32_mul_array_f32 = &overload_val_f32 * &overload_array_f32;
        let result_array_f32_mul_val_f32 = &overload_array_f32 * &overload_val_f32;

        assert_eq!(
            RawData::new(4i32).get_data(),
            result_val_i32_mul_val_i32.get_data()
        );

        assert_eq!(
            RawData::new(20u32).get_data(),
            result_val_u32_mul_scalar_u32.get_data()
        );

        assert_eq!(
            RawData::new(20.0f64).get_data(),
            result_scalar_f64_mul_val_f64.get_data()
        );

        assert_eq!(
            RawData::new(4.0f32).get_data(),
            result_val_f32_mul_array_f32.get_data()
        );

        assert_eq!(
            RawData::new(4.0f32).get_data(),
            result_array_f32_mul_val_f32.get_data()
        );
    }

    /// 加算オーバーロードのテスト
    /// 様々な型、および、左右オペランドを入れ替えたテスト
    #[test]
    fn test_add_overload_macro() {
        let overload_val_i64 = Variable::new(RawData::new(2i64));
        let overload_val_f32 = Variable::new(RawData::new(2.0f32));
        let overload_val_f64 = Variable::new(RawData::new(2.0f64));
        let overload_val_u64 = Variable::new(RawData::new(2u64));
        let overload_array_f32 = Array::from_elem(IxDyn(&[]), 2.0f32);

        let result_val_i64_add_val_i64 = &overload_val_i64 + &overload_val_i64;
        let result_val_u64_add_scalar_u64 = &overload_val_u64 + 10u64;
        let result_scalar_f64_add_val_f64 = 10.0f64 + &overload_val_f64;
        let result_val_f32_add_array_f32 = &overload_val_f32 + &overload_array_f32;
        let result_array_f32_add_val_f32 = &overload_array_f32 + &overload_val_f32;

        assert_eq!(
            RawData::new(4i64).get_data(),
            result_val_i64_add_val_i64.get_data()
        );

        assert_eq!(
            RawData::new(12u64).get_data(),
            result_val_u64_add_scalar_u64.get_data()
        );

        assert_eq!(
            RawData::new(12.0f64).get_data(),
            result_scalar_f64_add_val_f64.get_data()
        );

        assert_eq!(
            RawData::new(4.0f32).get_data(),
            result_val_f32_add_array_f32.get_data()
        );

        assert_eq!(
            RawData::new(4.0f32).get_data(),
            result_array_f32_add_val_f32.get_data()
        );
    }

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
