use crate::functions::*;
use crate::settings::*;
use crate::variable::*;
use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// 加算関数
#[derive(Debug, Clone)]
struct Add;
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
fn add<V: MathOps>(
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
