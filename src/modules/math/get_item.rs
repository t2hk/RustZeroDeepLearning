// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::rc::Rc;
use std::{cell::RefCell, rc::Weak};

use ndarray::{s, Array, Axis, IxDyn};

#[derive(Debug, Clone)]
pub struct GetItemFunction {
    x_shape: Vec<usize>,
    slicer: DynamicSlicer,
}

impl GetItemFunction {
    fn new(x_shape: Vec<usize>, slicer: DynamicSlicer) -> Self {
        let shape = x_shape.clone();
        Self {
            x_shape: shape,
            slicer: slicer,
        }
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

        // let result = utils::get_slice(xs[0].clone(), self.slicer.clone()).unwrap();
        let result = self.slicer.slice(&xs[0].clone()).to_owned();
        dbg!(&result);
        vec![result]
    }

    /// 逆伝播
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        _outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        info!("get_item(backward)");

        // let gy = get_item_grad(gys[0].clone(), self.slicer.clone(), self.x_shape.clone());

        // vec![gy]
        gys
    }
}

/// GetItem関数
///
/// Arguments
/// * x (Variable<V>): 変数
/// * slice (SliceElem): スライス
///
/// Return
/// * Variable<V>: スライス結果
pub fn get_item<V: MathOps>(x: Variable<V>, slicer: DynamicSlicer) -> Variable<V> {
    debug!("GetItemFunction::get_item");

    let x_shape = x.get_data().shape().to_vec();
    let mut get_item =
        FunctionExecutor::new(Rc::new(RefCell::new(GetItemFunction::new(x_shape, slicer))));

    // GetItem の順伝播
    get_item.forward(vec![x.clone()]).first().unwrap().clone()
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, Axis, Slice, SliceInfo, SliceInfoElem};

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
    fn test_forward1_1() {
        // Python 参考
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, 0)
        // self.assertTrue(array_allclose(y.data, x_data[0]))

        // let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        // let indices = vec![0_usize];
        // let result = array.index_axis(Axis(0), indices[0]);
        // println!("array: {:?}\nresult: {:?}", array, result);
        // result: variable([[0 1 2] [3 4 5]])
        // shape: (2, 3)

        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        // let sim = SliceElem::Index([0].to_vec());
        // let result = get_item(x, sim);

        let mut slicer = DynamicSlicer::new(x.get_data().ndim());
        // 第1次元 (dim=0) のみをスライス
        slicer.set_slice(0, DynamicSlice::Index(0)); // 第1次元の0のみを選択

        // let slicer = slicer.slice(&x.get_data());

        let result = get_item(x.clone(), slicer);

        assert_eq!(vec![2, 3], result.get_data().shape().to_vec().clone());
        assert_eq!(vec![0, 1, 2, 3, 4, 5], result.get_data().flatten().to_vec());
    }

    #[test]
    fn test_forward1_2() {
        // Python の例
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, (0, 0, slice(0, 2, 1)))
        // self.assertTrue(array_allclose(y.data, x_data[0, 0, 0:2:1]))
        // [[0 1]

        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        // let sim = SliceElem::Index([0, 0, 1].to_vec());
        // let result = get_item(x, sim);

        let slice = DynamicSlice::Range {
            start: Some(0),
            end: Some(2),
            step: 1,
        };

        let mut slicer = DynamicSlicer::new(x.get_data().ndim());
        slicer
            .set_slice(0, DynamicSlice::Index(0)) // 第1次元: インデックス0
            .set_slice(1, DynamicSlice::Index(0)) // 第2次元: インデックス0
            .set_slice(
                2,
                DynamicSlice::Range {
                    // 第3次元: 0から2未満 (0, 1)
                    start: Some(0),
                    end: Some(2),
                    step: 1,
                },
            );

        // let slicer = slicer.slice(&x.get_data());

        let result = get_item(x.clone(), slicer);

        assert_eq!(vec![2], result.get_data().shape().to_vec());
        assert_eq!(vec![0, 1], result.get_data().flatten().to_vec());
    }

    #[test]
    fn test_forward2() {
        // Python の例
        // x_data = np.arange(12).reshape((2, 2, 3))
        //   x = Variable(x_data)
        //   y = F.get_item(x, (Ellipsis, 2))
        //   self.assertTrue(array_allclose(y.data, x_data[..., 2]))
        // [[ 2  5] [ 8 11]]
        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        let x = Variable::new(RawData::from_shape_vec(vec![2, 2, 3], (0..=11).collect()));

        let mut slicer = DynamicSlicer::new(array.ndim());
        // 全ての次元の最後の要素を取得
        slicer.set_slice(
            2,
            DynamicSlice::Range {
                start: Some(2),
                end: Some(3),
                step: 1,
            },
        );

        let result = get_item(x.clone(), slicer);

        println!("result: {:?}", result);
        assert_eq!(vec![2, 2], result.clone().get_data().shape().to_vec());
        assert_eq!(
            vec![2, 5, 8, 11],
            result.clone().get_data().flatten().to_vec()
        );
    }

    // #[test]
    // fn test_backward1() {
    //     // Python 参考
    //     // x_data = np.array([[1, 2, 3], [4, 5, 6]])
    //     // slices = 1
    //     // f = lambda x: F.get_item(x, slices)
    //     // gradient_check(f, x_data)

    //     let i_array = Array::from_shape_vec(vec![2, 3], (0..=5).collect()).unwrap();
    //     let array = i_array.mapv(|x| x as f64);

    //     let x = Variable::new(RawData::from_shape_vec(
    //         array.shape().to_vec(),
    //         array.flatten().to_vec(),
    //     ));

    //     let slice = SliceElem::Index([1].to_vec());

    //     let x_shape = x.get_data().shape().to_vec();
    //     let mut get_item =
    //         FunctionExecutor::new(Rc::new(RefCell::new(GetItemFunction::new(x_shape, slice))));

    //     utils::gradient_check(&mut get_item, vec![x.clone()]);
    // }

    // #[test]
    // fn test_backward2() {
    //     // Python 参考
    //     // x_data = np.arange(12).reshape(4, 3)
    //     // slices = slice(1, 3)
    //     // f = lambda x: F.get_item(x, slices)
    //     // gradient_check(f, x_data)

    //     let i_array = Array::from_shape_vec(vec![4, 3], (0..=11).collect()).unwrap();
    //     let array = i_array.mapv(|x| x as f64);
    //     let x = Variable::new(RawData::from_shape_vec(
    //         array.shape().to_vec(),
    //         array.flatten().to_vec(),
    //     ));

    //     let slice = SliceElem::Slice {
    //         start: Some(1),
    //         end: Some(2),
    //         step: 1,
    //     };

    //     // let test = utils::get_slice(array.clone(), slice.clone()).unwrap();
    //     // dbg!(&test);

    //     let x_shape = x.get_data().shape().to_vec();
    //     let mut get_item =
    //         FunctionExecutor::new(Rc::new(RefCell::new(GetItemFunction::new(x_shape, slice))));

    //     utils::gradient_check(&mut get_item, vec![x.clone()]);
    // }

    // #[test]
    // fn test_1dim() {
    //     let i_array = Array::from_vec((0..=11).collect()).into_dyn();
    //     let array = i_array.mapv(|x| x as f64);

    //     let slice = SliceElem::Slice {
    //         start: Some(1),
    //         end: Some(3),
    //         step: 1,
    //     };
    //     dbg!(&array);

    //     // let test = array.slice(s![..,..,1..3;1]);
    //     let test = array.slice(s![1..3;1]);

    //     dbg!(&test);
    // }

    // #[test]
    // fn test_2dim() {
    //     let i_array = Array::from_shape_vec(vec![4, 3], (0..=11).collect())
    //         .unwrap()
    //         .into_dyn();
    //     let array = i_array.mapv(|x| x as f64);

    //     let slice = SliceElem::Slice {
    //         start: Some(1),
    //         end: Some(3),
    //         step: 1,
    //     };
    //     dbg!(&array);

    //     // let test = array.slice(s![..,..,1..3;1]);
    //     let test = array.slice(s![ 1..3;1,..]);

    //     // let test = utils::get_slice(array.clone(), slice.clone()).unwrap();
    //     dbg!(&test);
    // }

    // #[test]
    // fn test_3dim() {
    //     let i_array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect())
    //         .unwrap()
    //         .into_dyn();
    //     let array = i_array.mapv(|x| x as f64);

    //     let slice = SliceElem::Slice {
    //         start: Some(1),
    //         end: Some(3),
    //         step: 1,
    //     };
    //     dbg!(&array);

    //     // let test = array.slice(s![..,..,1..3;1]);
    //     // [[6, 7, 8] [9, 10, 11]]
    //     // let test = array.slice(s![1..2, .., 0..]).into_dyn().squeeze();
    //     let ndim = array.ndim();
    //     let test = array.slice(s![1..2, .., ..]).into_dyn().squeeze();

    //     // let test = utils::get_slice(array.clone(), slice.clone()).unwrap();
    //     dbg!(&test);
    // }

    // #[test]
    // fn test_3dim_2() {
    //     let i_array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect())
    //         .unwrap()
    //         .into_dyn();
    //     let array = i_array.mapv(|x| x as f64);

    //     let slice = SliceElem::Slice {
    //         start: Some(1),
    //         end: Some(3),
    //         step: 1,
    //     };
    //     dbg!(&array);

    //     // let test = array.slice(s![..,..,1..3;1]);
    //     // [[6, 7, 8] [9, 10, 11]]
    //     // let test = array.slice(s![1..2, .., 0..]).into_dyn().squeeze();
    //     let ndim = array.ndim();

    //     let test_1_all_all = array.slice(s![1, .., ..]).into_dyn().squeeze();
    //     println!("============");
    //     println!("[1, .., ..] -> {:?}", test_1_all_all);

    //     println!("============");
    //     let test_all_1_all = array.slice(s![.., 1, ..]).into_dyn().squeeze();
    //     println!("[.., 1, ..] -> {:?}", test_all_1_all);

    //     println!("============");
    //     let test_all_all_1 = array.slice(s![.., .., 1]).into_dyn().squeeze();
    //     println!("[.., .., 1] -> {:?}", test_all_all_1);

    //     println!("============");
    //     let test_1_all_1 = array.slice(s![1, .., 1]).into_dyn().squeeze();
    //     println!("[1, .., 1] -> {:?}", test_1_all_1);
    // }

    // #[test]
    // fn test_5dim_1() {
    //     let i_array = Array::from_shape_vec(vec![3, 2, 3, 2, 3], (0..=107).collect())
    //         .unwrap()
    //         .into_dyn();
    //     let array = i_array.mapv(|x| x as f64);

    //     let slice = SliceElem::Slice {
    //         start: Some(1),
    //         end: Some(3),
    //         step: 1,
    //     };
    //     dbg!(&array);

    //     // let test = array.slice(s![..,..,1..3;1]);
    //     // [[6, 7, 8] [9, 10, 11]]
    //     // let test = array.slice(s![1..2, .., 0..]).into_dyn().squeeze();
    //     let ndim = array.ndim();

    //     let test_1_all_5 = array.slice(s![1, .., .., .., ..]).into_dyn().squeeze();
    //     println!("============");
    //     println!("[1, .., .., .., .., ] -> {:?}", test_1_all_5);

    //     let test_all_4_1 = array.slice(s![.., .., .., .., 1]).into_dyn().squeeze();
    //     println!("============");
    //     println!("[.., .., .., .., 1, ] -> {:?}", test_all_4_1);

    //     let test_all_1 = array.slice(s![1, 1, 1, 1, 1]).into_dyn().squeeze();
    //     println!("============");
    //     println!("[1, 1, 1, 1, 1, ] -> {:?}", test_all_1);

    //     let test_1_1_0to3 = array.slice(s![1, 1, 0..3, .., ..]).into_dyn().squeeze();
    //     println!("============");
    //     println!("[1, 1, 0..3, .., ..] -> {:?}", test_1_1_0to3);
    // }

    // #[test]
    // fn test_sample() {
    //     // 3次元配列の作成 (2x3x4)
    //     let mut array = Array::zeros((2, 3, 4));

    //     // 配列にデータを設定
    //     for i in 0..2 {
    //         for j in 0..3 {
    //             for k in 0..4 {
    //                 array[[i, j, k]] = (i * 100 + j * 10 + k) as i32;
    //             }
    //         }
    //     }

    //     println!("元の配列:\n{:?}\n", array);

    //     // 例1: 単一のインデックス指定
    //     let slice1 = array.slice(s![0, 0, 0]); // 単一要素にアクセス
    //     println!("例1 - 単一要素(0,0,0): {:?}", slice1);

    //     // 例2: 範囲指定
    //     let slice2 = array.slice(s![0, 0..2, 0..2]); // 部分配列の取得
    //     println!("例2 - 部分配列(0, 0..2, 0..2):\n{:?}", slice2);

    //     // 例3: 全要素指定
    //     let slice3 = array.slice(s![0, .., 0]); // 0番目の「面」の最初の列
    //     println!("例3 - 全要素(0, .., 0):\n{:?}", slice3);

    //     // 例4: 末尾からのインデックス指定
    //     let slice4 = array.slice(s![-1, .., ..]); // 最後の「面」
    //     println!("例4 - 末尾からのインデックス(-1, .., ..):\n{:?}", slice4);

    //     // 例5: ステップ付き範囲
    //     let slice5 = array.slice(s![0, 0..;2, 0..;2]); // 2ステップで要素を取得
    //     println!("例5 - ステップ付き(0, 0..;2, 0..;2):\n{:?}", slice5);

    //     // 例6: 含む範囲
    //     let slice6 = array.slice(s![0, 0..=1, 0..=2]); // 終端を含む範囲
    //     println!("例6 - 終端を含む(0, 0..=1, 0..=2):\n{:?}", slice6);

    //     // 例7: 複数の次元に対する範囲指定
    //     let slice7 = array.slice(s![0..2, 1, 0..3]);
    //     println!("例7 - 複数次元の範囲(0..2, 1, 0..3):\n{:?}", slice7);
    // }
}
