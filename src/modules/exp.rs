use crate::functions::*;
use crate::settings::*;
use crate::variable::*;

use ndarray::{Array, IxDyn};
use num_traits::NumCast;
use std::cell::RefCell;
use std::rc::Rc;

/// Exp 関数
#[derive(Debug, Clone)]
struct Exp;
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
        let x = inputs[0].borrow().data.clone();
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
fn exp<V: MathOps>(input: Rc<RefCell<Variable<V>>>) -> Rc<RefCell<Variable<V>>> {
    let mut exp = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
    // EXP の順伝播
    exp.forward(vec![input.clone()]).get(0).unwrap().clone()
}
