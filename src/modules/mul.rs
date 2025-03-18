use crate::functions::*;
use crate::settings::*;
use crate::variable::*;
use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// 乗算関数
#[derive(Debug, Clone)]
struct Mul;
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
        let x1 = inputs[0].borrow().data.clone();
        let x2 = inputs[1].borrow().data.clone();
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
fn mul<V: MathOps>(
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
