use ndarray::{s, Array, ArrayD, Dimension, IxDyn, SliceInfo, SliceInfoElem};

// GetItem相当の構造体
struct GetItem {
    slices: Vec<SliceInfoElem>,
}

// GetItemGrad相当の構造体（逆伝播用）
struct GetItemGrad {
    slices: Vec<SliceInfoElem>,
    input_shape: Vec<usize>,
}

impl GetItem {
    fn new(slices: Vec<SliceInfoElem>) -> Self {
        Self { slices }
    }

    // forwardメソッド
    // fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
    //     // SliceInfoを作成 - 3つのジェネリックパラメータを指定
    //     let slice_info = SliceInfo::<_, IxDyn, IxDyn>::new(self.slices.clone()).unwrap();

    //     // スライスを適用してコピーを返す
    //     x.slice(&slice_info).to_owned()
    // }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

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
    }
}
