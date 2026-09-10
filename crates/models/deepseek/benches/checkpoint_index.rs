use std::{fmt::Write, hint::black_box, sync::LazyLock};

use deepseek::manifest::V41SafetensorsIndex;

static V41_SCALE_INDEX: LazyLock<String> = LazyLock::new(|| {
    let mut weight_map = String::from("{");
    for tensor in 0..96_085 {
        if tensor != 0 {
            weight_map.push(',');
        }
        let shard = (tensor % 48) + 1;
        write!(
            weight_map,
            "\"model.layers.{tensor}.weight\":\"model-{shard:05}-of-00048.safetensors\""
        )
        .expect("writing to a String cannot fail");
    }
    weight_map.push('}');
    format!("{{\"metadata\":{{\"total_size\":510286023000}},\"weight_map\":{weight_map}}}")
});

#[divan::bench]
fn parses_v41_scale_index() {
    let index = V41SafetensorsIndex::parse(black_box(V41_SCALE_INDEX.as_str()))
        .expect("generated V4.1-scale index is valid");
    black_box(index);
}

fn main() {
    divan::main();
}
