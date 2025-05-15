// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use ndarray_rand::rand_distr::Normal;
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, Circle, Cross, IntoDrawingArea, IntoDynElement, TriangleMarker},
    series::PointSeries,
    style::{Color, IntoFont, BLACK, BLUE, GREEN, RED, WHITE},
};
use rand::{prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng};

/// スパイラルデータセット
///
/// Arguments:
/// * train (bool): true の場合は学習用データ、false の場合はテスト用データ
///
/// Retrun:
/// (Array<f64, IxDyn>, Array<usize, IxDyn>): (入力データ、教師ラベル)
pub fn get_spiral(train: bool) -> (Array<f64, IxDyn>, Array<usize, IxDyn>) {
    let seed = if train { 1984 } else { 2020 };

    let mut rng = StdRng::seed_from_u64(seed);

    let num_data = 100;
    let num_class = 3;
    let input_dim = 2;
    let data_size = num_class * num_data;

    let mut x: Array<f64, _> = Array::zeros((data_size, input_dim));
    let mut t: Array<usize, _> = Array::zeros((data_size));

    for j in 0..num_class {
        for i in 0..num_data {
            //  rate = i / num_data
            //  radius = 1.0 * rate
            //  theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            //  ix = num_data * j + i
            //  x[ix] = np.array([radius * np.sin(theta),
            //                    radius * np.cos(theta)]).flatten()
            //  t[ix] = j
            let rate = i as f64 / num_data as f64;
            let radius = 1.0 * rate;
            let theta =
                j as f64 * 4.0 + 4.0 * rate + Normal::new(0.0, 0.2).unwrap().sample(&mut rng);
            let ix = num_data * j + i;

            x[[ix, 0]] = radius * theta.sin();
            x[[ix, 1]] = radius * theta.cos();
            t[ix] = j;
        }
    }
    let mut indices: Vec<usize> = (0..data_size).collect();
    indices.shuffle(&mut rng);

    let mut x_shuffled: Array<f64, _> = Array::zeros((data_size, input_dim));
    let mut t_shuffled: Array<usize, _> = Array::zeros(data_size);

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        x_shuffled.row_mut(new_idx).assign(&x.row(old_idx));
        t_shuffled[new_idx] = t[old_idx];
    }

    (x_shuffled.into_dyn(), t_shuffled.into_dyn())
}

/// スパイラルデータをグラフ描画する。
pub fn draw_spiral_graph(train: bool) {
    let (x, t) = get_spiral(train);

    let file_path = if train {
        "graph/spiral_data_train.png"
    } else {
        "graph/spiral_data_test.png"
    };
    let caption = "spiral";

    // グラフ描画
    // 描画先の Backend を初期化する。
    let root = BitMapBackend::new(&file_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // データ点を生成し、値に応じて異なるマーカーを使用
    chart
        .draw_series((0..300).map(|i| {
            let x_x = x[[i, 0]];
            let x_y = x[[i, 1]];

            let point = (x_x, x_y);

            let t_var = t[i];

            match t_var {
                0 => Circle::new(point, 4, GREEN.filled()).into_dyn(),
                1 => TriangleMarker::new(point, 4, BLUE.filled()).into_dyn(),
                _ => Cross::new(point, 4, RED.filled()).into_dyn(),
            }
        }))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_spiral() {
        let (x, t) = get_spiral(true);

        println!("x shape: {:?}", x.shape());
        println!("t shape: {:?}", t.shape());

        println!("x[10]: ({}, {}), t[10]: {}", x[[10, 0]], x[[10, 1]], t[10]);
        println!(
            "x[110]: ({}, {}), t[110]: {}",
            x[[110, 0]],
            x[[110, 1]],
            t[110]
        );

        draw_spiral_graph(true);
        draw_spiral_graph(false);
    }
}
