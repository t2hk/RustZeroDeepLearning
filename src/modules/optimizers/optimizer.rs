// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

/// オプティマイザトレイト
pub trait Optimizer {
    type Val: MathOps;

    /// 1つのパラメータを更新する。
    ///
    /// Arguments
    /// * param (&mut Variable<V>): 更新するパラメータ
    fn update_one(&mut self, param: &mut Variable<Self::Val>);
}
