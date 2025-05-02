// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use ndarray::{s, Array, Axis, IxDyn};

#[derive(Debug, Clone)]
pub enum SliceElem {
    Slice {
        /// start index; negative are counted from the back of the axis
        start: Option<isize>,
        /// end index; negative are counted from the back of the axis; when not present
        /// the default is the full length of the axis.
        end: Option<isize>,
        /// step size in elements; the default is 1, for every element.
        step: isize,
    },
    /// A single index.
    Index(Vec<usize>),
    /// A new axis of length 1.
    NewAxis,
}

#[derive(Debug, Clone)]
pub struct GetItemFunction {
    slice: SliceElem,
}

impl GetItemFunction {
    fn new(slice: SliceElem) -> Self {
        Self { slice }
    }
}

/// GetItem関数
impl<V: MathOps> Function<V> for GetItemFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "GetItem".to_string()
    }

    // GetItem の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("get_item_grad(forward)");

        let result = utils::get_slice(xs[0].clone(), self.slice.clone()).unwrap();
        vec![result]
    }

    /// 逆伝播
    /// y=x0 / x1 の微分であるため、dy/dx0=1/x1 * gy, dy/dx1= -x0/(x1^2) * gy である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("get_item(backward)");

        let gx_x0 = &gys[0] / &inputs[1];
        let gx_x1 = &gys[0] * &(&(&inputs[0] * -1) / &(&inputs[1] ^ 2));

        debug!(
            "get_item(backward): dy/dx0 = (1 / {:?}) * {:?}, dy/dx1 = -{:?} / {:?}^2 * {:?}",
            &inputs[1].get_data().flatten().to_vec(),
            &gys[0].get_data().flatten().to_vec(),
            &inputs[0].get_data().flatten().to_vec(),
            &inputs[1].get_data().flatten().to_vec(),
            &gys[0].get_data().flatten().to_vec()
        );

        let gxs = vec![gx_x0, gx_x1];
        gxs
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, Array2, Axis, Slice, SliceInfo, SliceInfoElem};

    #[test]
    fn test_get_item() {
        let arr = array![0, 1, 2, 3];
        assert_eq!(arr.slice(s![1..3;-1]), array![2, 1]);
        assert_eq!(arr.slice(s![1..;-2]), array![3, 1]);
        assert_eq!(arr.slice(s![0..4;-2]), array![3, 1]);
        assert_eq!(arr.slice(s![0..;-2]), array![3, 1]);
        assert_eq!(arr.slice(s![..;-2]), array![3, 1]);
        println!("s 1..3 -> {:?}", arr.slice(s![1..3]));
        println!("s 1..3;-1 -> {:?}", arr.slice(s![1..3;-1]));
        println!("s 1..3;-2 -> {:?}", arr.slice(s![1..3;-2]));
        println!("s 1..3;-4 -> {:?}", arr.slice(s![1..3;-4]));

        unsafe {
            let array1 =
                Array::from_shape_vec(vec![3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                    .unwrap();

            // 0..2 のような範囲を表すSliceInfoElemを作成
            let start = 1;
            let end = 3;
            let row_slice = SliceInfoElem::from(Slice::new(start as isize, Some(end as isize), 1));
            // すべての要素を選択するSliceInfoElemを作成
            let col_slice = SliceInfoElem::from(Slice::from(..));
            let slice_info: SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> =
                SliceInfo::new(vec![row_slice, col_slice]).unwrap();

            // let elem = SliceInfoElem::Index(1);
            // let slice_info: SliceInfo<_, IxDyn, IxDyn> = SliceInfo::new(vec![elem]).unwrap();

            // let result = array1.slice(slice_info);
            let result = array1.slice(&slice_info);
            println!("result: {:?}", result);

            let index = 1;
            let row_index = SliceInfoElem::from(Slice::new(index as isize, Some(index), 1));
            let slice_index_info: SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> =
                SliceInfo::new(vec![row_index, col_slice]).unwrap();
            let result2 = array1.slice(&slice_index_info);
            println!("result2: {:?}", result2);

            //let indices = Array::from_vec(vec![0, 0, 1]);
            let indices = vec![0_usize, 0_usize, 1_usize];
            let result3 = array1.select(Axis(0), &indices);
            println!("result3: {:?}", result3);

            let array2 = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
            let indices_0 = vec![0_usize];
            let result4 = array2.select(Axis(0), &indices_0);
            println!("array2: {:?}\nresult4: {:?}", array2, result4);
        }
    }

    #[test]
    fn test_forward1() {
        // Python 参考
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, 0)
        // self.assertTrue(array_allclose(y.data, x_data[0]))

        // let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        // let indices = vec![0_usize];
        // let result = array.index_axis(Axis(0), indices[0]);
        // println!("array: {:?}\nresult: {:?}", array, result);

        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        let sim = SliceElem::Index([0, 0, 1].to_vec());
        let result = get_item(x, sim);

        assert_eq!(
            vec![3, 2, 3],
            result.clone().unwrap().get_data().shape().to_vec()
        );
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            result.unwrap().get_data().flatten().to_vec()
        );
    }

    #[test]
    fn test_forward2() {
        // Python 参考
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, (0, 0, slice(0, 2, 1)))
        // self.assertTrue(array_allclose(y.data, x_data[0, 0, 0:2:1]))

        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        let sim = SliceElem::Slice {
            start: Some(0),
            end: Some(3),
            step: 1,
        };

        let result = get_item(x.clone(), sim).unwrap();

        println!("result: {:?}", result);
        assert_eq!(vec![3], result.clone().get_data().shape().to_vec());
        assert_eq!(vec![0, 1, 2], result.clone().get_data().flatten().to_vec());
    }

    #[test]
    fn test_forward2_2() {
        // Python 参考
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, (0, 0, slice(0, 2, 1)))
        // self.assertTrue(array_allclose(y.data, x_data[0, 0, 0::1]))

        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        let sim = SliceElem::Slice {
            start: Some(0),
            end: None,
            step: 1,
        };

        let result = get_item(x.clone(), sim).unwrap();

        println!("result: {:?}", result);
        assert_eq!(vec![3], result.clone().get_data().shape().to_vec());
        assert_eq!(vec![0, 1, 2], result.clone().get_data().flatten().to_vec());
    }

    #[test]
    fn test_forward3() {
        // x_data = np.arange(12).reshape((2, 2, 3))
        //   x = Variable(x_data)
        //   y = F.get_item(x, (Ellipsis, 2))
        //   self.assertTrue(array_allclose(y.data, x_data[..., 2]))
        // result:  y: variable([[ 2  5]   [ 8 11]])

        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        let sim = SliceElem::Slice {
            start: None,
            end: None,
            step: 2,
        };

        // let result = array.slice(s![.., .., 1]).to_owned();
        let result = get_item(x.clone(), sim).unwrap();

        println!("result: {:?}", result);
        assert_eq!(vec![2, 2], result.clone().get_data().shape().to_vec());
        assert_eq!(
            vec![2, 5, 8, 11],
            result.clone().get_data().flatten().to_vec()
        );
    }
}
