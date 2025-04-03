use env_logger;
// use log::{debug, error, info, trace, warn};
use std::env;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn setup() {
    INIT.call_once(|| {
        env::set_var("RUST_LOG", "debug");

        env_logger::init();
    });
}
