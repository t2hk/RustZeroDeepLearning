use crate::modules::functions::*;
use crate::modules::settings::*;
use crate::modules::variable::*;
use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// 加算関数
#[derive(Debug, Clone)]
pub struct Add;
impl<V: MathOps> Function<V> for Add {
    // Add (加算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] + &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x0+x1 の微分であるため、dy/dx0=1, dy/dx1=1 である。
    fn backward(
        &self,
        _inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

/// 加算関数
///
/// Arguments
/// * x1 (Rc<RefCell<Variable>>): 加算する変数
/// * x2 (Rc<RefCell<Variable>>): 加算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 加算結果
pub fn add<V: MathOps>(
    x1: Rc<RefCell<Variable<V>>>,
    x2: Rc<RefCell<Variable<V>>>,
) -> Rc<RefCell<Variable<V>>> {
    let mut add = FunctionExecutor::new(Rc::new(RefCell::new(Add)));
    // 加算の順伝播
    add.forward(vec![x1.clone(), x2.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// 乗算関数
#[derive(Debug, Clone)]
pub struct Mul;
impl<V: MathOps> Function<V> for Mul {
    // Mul (乗算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] * &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x1 * x2 の微分であるため、dy/dx1=x2 * gy, dy/dx2= x1 * gy である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
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
pub fn mul<V: MathOps>(
    x1: Rc<RefCell<Variable<V>>>,
    x2: Rc<RefCell<Variable<V>>>,
) -> Rc<RefCell<Variable<V>>> {
    let mut mul = FunctionExecutor::new(Rc::new(RefCell::new(Mul)));
    // 乗算の順伝播
    mul.forward(vec![x1.clone(), x2.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// 二乗関数
#[derive(Debug, Clone)]
pub struct Square;
impl<V: MathOps> Function<V> for Square {
    /// 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        //let result = vec![xs[0].mapv(|x| x.pow(V::from(2).unwrap()))];
        let result = vec![xs[0].mapv(|x| x * x)];

        //dpg!(result);

        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let x = inputs[0].borrow().get_data();
        // let gxs = vec![V::from(2).unwrap() * &x * &gys[0].clone()];
        let x_gys = &gys[0].clone() * &x;
        let gxs = vec![x_gys.mapv(|x| x * V::from(2).unwrap())];
        gxs
    }
}

/// 二乗関数
///
/// Arguments
/// * input (Rc<RefCell<Variable>>): 加算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 二乗の結果
pub fn square<V: MathOps>(input: Rc<RefCell<Variable<V>>>) -> Rc<RefCell<Variable<V>>> {
    let mut square = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // 二乗の順伝播
    square.forward(vec![input]).get(0).unwrap().clone()
}

/// Exp 関数
#[derive(Debug, Clone)]
pub struct Exp;
impl<V: MathOps> Function<V> for Exp {
    // Exp (y=e^x) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let e = std::f64::consts::E;
        // let result = vec![xs[0].mapv(|x| e.powf(x.to_f64().unwrap()))];

        // let result = vec![xs[0].mapv(|x| V::from(x.to_f64().unwrap().exp()).unwrap())];
        let result = vec![xs[0].mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];

        result
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let e = std::f64::consts::E;
        let x = inputs[0].borrow().get_data().clone();
        let gys_val = gys[0].clone();
        let x_exp = vec![x.mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];
        let gxs = x_exp.iter().map(|x_exp| x_exp * &gys_val).collect();
        gxs
    }
}

/// Exp 関数
///
/// Arguments
/// * input (Rc<RefCell<Variable>>): 入力値
///
/// Return
/// * Rc<RefCell<Variable>>: 結果
pub fn exp<V: MathOps>(input: Rc<RefCell<Variable<V>>>) -> Rc<RefCell<Variable<V>>> {
    let mut exp = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
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
        let x1 = Rc::new(RefCell::new(Variable::new(rand_x1)));
        let x2 = Rc::new(RefCell::new(Variable::new(rand_x2)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

        // 順伝播、逆伝播を実行する。
        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let result = add(Rc::clone(&x1), Rc::clone(&x2));

        // 足し算の結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }

    /// 二乗のテスト
    #[test]
    fn test_square() {
        // 2乗する値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x = rng.random::<f64>();
        let x = Rc::new(RefCell::new(Variable::new(rand_x)));

        // 2乗した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x * rand_x);
        let expected_grad_val = rand_x * 2.0 * 1.0;
        let expected_output_grad = Array::from_elem(IxDyn(&[]), expected_grad_val);

        // 順伝播実行する。
        let result = square(Rc::clone(&x));

        // 二乗の結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }
}
