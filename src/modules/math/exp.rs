// ライブラリを一括でインポート
use crate::modules::math::*;

#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Exp 関数
#[derive(Debug, Clone)]
pub struct ExpFunction;
impl<V: MathOps> Function<V> for ExpFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Exp".to_string()
    }

    // Exp (y=e^x) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("exp(forward)");
        debug!("exp(forward): e ^ {:?}", &xs[0].flatten().to_vec());
        let e = std::f64::consts::E;
        let result = vec![xs[0].mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];

        result
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("exp(backward)");

        // let x = inputs[0].borrow().get_data();
        // let gys_val = gys[0].clone();
        let x_exp = exp(inputs[0].clone());
        // let x_exp = vec![x.mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];
        //let gxs = x_exp.iter().map(|x_exp| x_exp * &gys_val).collect();
        let gxs = vec![&x_exp * &gys[0].clone()];
        debug!(
            "exp(backward): (e ^ {:?}) * {:?}",
            &inputs[0].borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec()
        );

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

    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1, 100), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![1, 100],
            x0_var.flatten().to_vec(),
        ));

        let mut exp = FunctionExecutor::new(Rc::new(RefCell::new(ExpFunction {})));

        utils::gradient_check(&mut exp, vec![x0.clone()]);
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
}
