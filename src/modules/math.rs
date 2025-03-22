use crate::modules::functions::*;
use crate::modules::settings::*;
use crate::modules::variable::*;
use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

/// 加算関数
#[derive(Debug, Clone)]
pub struct AddFunction;
impl<V: MathOps> Function<V> for AddFunction {
    // Add (加算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] + &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x0+x1 の微分であるため、dy/dx0=1, dy/dx1=1 である。
    fn backward(
        &self,
        _inputs: Vec<Variable<V>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

/// 加算関数
///
/// Arguments
/// * x1 (Variable<V>): 加算する変数
/// * x2 (Variable<V>): 加算する変数
///
/// Return
/// * Rc<RefCell<RawVariable>>: 加算結果
pub fn add<V: MathOps>(x1: Variable<V>, x2: Variable<V>) -> Variable<V> {
    let mut add = FunctionExecutor::new(Rc::new(RefCell::new(AddFunction)));
    // 加算の順伝播
    add.forward(vec![x1.clone(), x2.clone()])
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
impl<'a, 'b, V: MathOps> Add<&'b Variable<V>> for &'a Variable<V> {
    type Output = Variable<V>;
    fn add(self, rhs: &'b Variable<V>) -> Variable<V> {
        // 順伝播
        let mut add = FunctionExecutor::new(Rc::new(RefCell::new(AddFunction)));
        let result = add
            .forward(vec![self.clone(), rhs.clone()])
            .get(0)
            .unwrap()
            .clone();
        result
    }
}

/// 加算のオーバーロード (Variable 以外との計算)
impl<'a, 'b, V: MathOps> Add<&'b V> for &'a Variable<V> {
    type Output = Variable<V>;
    fn add(self, rhs: &'b V) -> Variable<V> {
        // 順伝播
        let rhs_val = Variable::new(RawVariable::new(*rhs));
        self + &rhs_val
    }
}

/// 加算のオーバーロード (Array との計算)
impl<'a, 'b, V: MathOps> Add<&'b Array<V, IxDyn>> for &'a Variable<V> {
    type Output = Variable<V>;
    fn add(self, rhs: &'b Array<V, IxDyn>) -> Variable<V> {
        // 順伝播
        let rhs_val = Variable::new(RawVariable::new(rhs.clone()));
        self + &rhs_val
    }
}

/// 乗算関数
#[derive(Debug, Clone)]
pub struct MulFunction;
impl<V: MathOps> Function<V> for MulFunction {
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
        let x1 = inputs[0].borrow().get_data();
        let x2 = inputs[1].borrow().get_data();
        let gx_x1 = &gys[0].clone() * &x2;
        let gx_x2 = &gys[0].clone() * &x1;

        let gxs = vec![gx_x1, gx_x2];
        gxs
    }
}

/// 乗算関数
///
/// Arguments
/// * x1 (Rc<RefCell<Variable>>): 乗算する変数
/// * x2 (Rc<RefCell<Variable>>): 乗算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 乗算結果
pub fn mul<V: MathOps>(x1: Variable<V>, x2: Variable<V>) -> Variable<V> {
    let mut mul = FunctionExecutor::new(Rc::new(RefCell::new(MulFunction)));
    // 乗算の順伝播
    mul.forward(vec![x1.clone(), x2.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// 乗算のオーバーロード
///
/// Arguments
/// * self (Variable<V>): 左オペランド
/// * rhs (Variable<V>): 右オペランド
///
/// Returns
/// * Variable<V>: 乗算結果
impl<'a, 'b, V: MathOps> Mul<&'b Variable<V>> for &'a Variable<V> {
    type Output = Variable<V>;
    fn mul(self, rhs: &'b Variable<V>) -> Variable<V> {
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

/// 乗算のオーバーロード (Variable 以外との計算)
impl<'a, 'b, V: MathOps> Mul<&'b V> for &'a Variable<V> {
    type Output = Variable<V>;
    fn mul(self, rhs: &'b V) -> Variable<V> {
        // 順伝播
        let rhs_val = Variable::new(RawVariable::new(*rhs));
        self * &rhs_val
    }
}

/// 乗算のオーバーロード (Array との計算)
impl<'a, 'b, V: MathOps> Mul<&'b Array<V, IxDyn>> for &'a Variable<V> {
    type Output = Variable<V>;
    fn mul(self, rhs: &'b Array<V, IxDyn>) -> Variable<V> {
        // 順伝播
        let rhs_val = Variable::new(RawVariable::new(rhs.clone()));
        self * &rhs_val
    }
}

/// 二乗関数
#[derive(Debug, Clone)]
pub struct SquareFunction;
impl<V: MathOps> Function<V> for SquareFunction {
    /// 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![xs[0].mapv(|x| x * x)];

        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let x = inputs[0].borrow().get_data();
        let x_gys = &gys[0].clone() * &x;
        let gxs = vec![x_gys.mapv(|x| x * V::from(2).unwrap())];
        gxs
    }
}

/// 二乗関数
///
/// Arguments
/// * input (Variable<V>): 加算する変数
///
/// Return
/// * Variable<V>: 二乗の結果
pub fn square<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut square = FunctionExecutor::new(Rc::new(RefCell::new(SquareFunction)));
    // 二乗の順伝播
    square.forward(vec![input]).get(0).unwrap().clone()
}

/// Exp 関数
#[derive(Debug, Clone)]
pub struct ExpFunction;
impl<V: MathOps> Function<V> for ExpFunction {
    // Exp (y=e^x) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let e = std::f64::consts::E;
        let result = vec![xs[0].mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];

        result
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let e = std::f64::consts::E;
        let x = inputs[0].borrow().get_data();
        let gys_val = gys[0].clone();
        let x_exp = vec![x.mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];
        let gxs = x_exp.iter().map(|x_exp| x_exp * &gys_val).collect();
        gxs
    }
}

/// Exp 関数
///
/// Arguments
/// * input (Rc<RefCell<RawVariable>>): 入力値
///
/// Return
/// * Rc<RefCell<RawVariable>>: 結果
pub fn exp<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut exp = FunctionExecutor::new(Rc::new(RefCell::new(ExpFunction)));
    // EXP の順伝播
    exp.forward(vec![input.clone()]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// 加算のテスト
    #[test]
    fn test_add() {
        // 加算値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x1 = rng.random::<f64>();
        let rand_x2 = rng.random::<f64>();
        let x1 = Variable::new(RawVariable::new(rand_x1));
        let x2 = Variable::new(RawVariable::new(rand_x2));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

        // 順伝播、逆伝播を実行する。
        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let result = add(x1, x2);

        // 足し算の結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }

    /// 二乗のテスト
    #[test]
    fn test_square() {
        // 2乗する値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x = rng.random::<f64>();
        let x = Variable::new(RawVariable::new(rand_x));

        // 2乗した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x * rand_x);

        // 順伝播実行する。
        let result = square(x);

        // 二乗の結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }

    /// Exp 関数のテスト。
    #[test]
    fn test_exp() {
        let x = Variable::new(RawVariable::new(2.0));

        let e = std::f64::consts::E;
        let expected_output_data = Array::from_elem(IxDyn(&[]), e.powf(2.0));

        // 順伝播、逆伝播を実行する。
        let result = exp(x);

        // exp 結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }

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
        // let mut raw_c = RawVariable::new(1.0f32);
        // raw_c.set_name("val_c".to_string());
        // let c = Variable::new(raw_c);

        // c は Variable ではなく i32 として計算する。
        let c = 1i32;

        // 計算する。a * b + c
        let result = &(&a * &b) + &c;

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
}
