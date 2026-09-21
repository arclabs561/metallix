use std::num::NonZeroUsize;

use deepseek::{RotaryDirection, RotaryFrequencyParameters, RotaryTailLayout, rotate_tail};

#[test]
fn kv_rotary_preserves_compressed_prefix_and_rotates_tail() {
    let mut values = (0..512).map(|value| value as f32 + 1.0).collect::<Vec<_>>();
    let prefix = values[..448].to_vec();
    let before_tail = values[448..].to_vec();
    let layout = RotaryTailLayout::new(
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(32).unwrap(),
    )
    .unwrap();
    let parameters = RotaryFrequencyParameters::new(
        NonZeroUsize::new(64).unwrap(),
        65_536,
        10_000.0,
        16.0,
        32.0,
        1.0,
    )
    .unwrap();
    let frequencies = parameters
        .frequencies(1, NonZeroUsize::new(1).unwrap())
        .unwrap();
    let mut tail = values.split_off(448);
    rotate_tail(&mut tail, layout, &frequencies, RotaryDirection::Forward).unwrap();
    values.extend_from_slice(&tail);

    assert_eq!(&values[..448], prefix.as_slice());
    assert!(
        values[448..]
            .iter()
            .zip(before_tail)
            .any(|(actual, before)| (actual - before).abs() > 1e-4)
    );
    assert!(values.iter().all(|value| value.is_finite()));
}
