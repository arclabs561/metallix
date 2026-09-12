//! Test-only scalar mirror of the pinned `Compressor.forward` state contract.
//!
//! The fixture supplies BF16 activations and uses identity `wkv` plus a
//! reversed-and-scaled `wgate`. This harness covers only pool timing, pool
//! axis, the FP32-to-BF16 boundary before `RMSNorm`, and state replacement.
//! It is not a checkpoint reader, attention implementation, or runtime API.

use deepseek::rms_norm_bf16_reference;
use serde::Deserialize;

const BATCHES: usize = 2;
const WIDTH: usize = 4;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    receipt: Receipt,
    batches: usize,
    width: usize,
    epsilon: f32,
    norm_weight_bf16: Vec<u16>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    sha256: String,
    symbols: Vec<String>,
}

#[derive(Deserialize)]
struct Receipt {
    torch: String,
    device: String,
}

#[derive(Deserialize)]
struct Case {
    ratio: usize,
    prefix: usize,
    calls: Vec<Call>,
    full_output_bf16: Vec<u16>,
}

#[derive(Deserialize)]
struct Call {
    start: usize,
    positions: usize,
    input_bf16: Vec<u16>,
    output_bf16: Option<Vec<u16>>,
    output_shape: Option<[usize; 3]>,
}

#[derive(Debug, Eq, PartialEq)]
enum CompressorStepError {
    GeometryOverflow,
    InputLength { actual: usize, expected: usize },
    InvalidRatio,
    ChunkedDecode { positions: usize },
    NonSequentialDecode { expected: usize, actual: usize },
    PositionOverflow,
}

/// Bounded state with the source's group slots, restricted to sequential
/// single-token decode after a start-zero prefill.
#[derive(Clone)]
struct CompressorState {
    ratio: usize,
    next_start: usize,
    kv: [[[f32; WIDTH]; 3]; BATCHES],
    score: [[[f32; WIDTH]; 3]; BATCHES],
}

impl CompressorState {
    fn new(ratio: usize) -> Result<Self, CompressorStepError> {
        if !(1..=3).contains(&ratio) {
            return Err(CompressorStepError::InvalidRatio);
        }
        Ok(Self {
            ratio,
            next_start: 0,
            kv: [[[0.0; WIDTH]; 3]; BATCHES],
            score: [[[f32::NEG_INFINITY; WIDTH]; 3]; BATCHES],
        })
    }

    fn forward(
        &mut self,
        input_bf16: &[u16],
        positions: usize,
        start: usize,
        norm_weight: &[u16],
        epsilon: f32,
    ) -> Result<Option<Vec<u16>>, CompressorStepError> {
        let expected = BATCHES
            .checked_mul(positions)
            .and_then(|values| values.checked_mul(WIDTH))
            .ok_or(CompressorStepError::GeometryOverflow)?;
        if input_bf16.len() != expected {
            return Err(CompressorStepError::InputLength {
                actual: input_bf16.len(),
                expected,
            });
        }
        if start != 0 {
            if positions != 1 {
                return Err(CompressorStepError::ChunkedDecode { positions });
            }
            if start != self.next_start {
                return Err(CompressorStepError::NonSequentialDecode {
                    expected: self.next_start,
                    actual: start,
                });
            }
        } else {
            // A new prefill owns the state. This makes reset safety explicit;
            // the source overwrites every slot needed before a later pool.
            self.kv = [[[0.0; WIDTH]; 3]; BATCHES];
            self.score = [[[f32::NEG_INFINITY; WIDTH]; 3]; BATCHES];
        }

        let next_start = start
            .checked_add(positions)
            .ok_or(CompressorStepError::PositionOverflow)?;
        let output = if self.ratio == 1 {
            Some(normalize_rows(input_bf16, positions, norm_weight, epsilon))
        } else if start == 0 {
            self.prefill(input_bf16, positions, norm_weight, epsilon)
        } else {
            self.decode(input_bf16, start, norm_weight, epsilon)?
        };
        self.next_start = next_start;
        Ok(output)
    }

    fn prefill(
        &mut self,
        input_bf16: &[u16],
        positions: usize,
        norm_weight: &[u16],
        epsilon: f32,
    ) -> Option<Vec<u16>> {
        let complete = positions / self.ratio;
        let cutoff = complete * self.ratio;
        for batch in 0..BATCHES {
            for position in cutoff..positions {
                self.store(
                    batch,
                    position - cutoff,
                    input(input_bf16, positions, batch, position),
                );
            }
        }
        if complete == 0 {
            return None;
        }
        let mut pooled = Vec::with_capacity(BATCHES * complete);
        for batch in 0..BATCHES {
            for group in 0..complete {
                let start = group * self.ratio;
                pooled.push(pool_group(
                    &input_bf16[(batch * positions + start) * WIDTH..][..self.ratio * WIDTH],
                    self.ratio,
                ));
            }
        }
        Some(normalize_rows_f32(
            &pooled,
            BATCHES * complete,
            norm_weight,
            epsilon,
        ))
    }

    fn decode(
        &mut self,
        input_bf16: &[u16],
        start: usize,
        norm_weight: &[u16],
        epsilon: f32,
    ) -> Result<Option<Vec<u16>>, CompressorStepError> {
        let slot = start % self.ratio;
        for batch in 0..BATCHES {
            self.store(batch, slot, input(input_bf16, 1, batch, 0));
        }
        if self
            .next_start
            .checked_add(1)
            .ok_or(CompressorStepError::PositionOverflow)?
            % self.ratio
            != 0
        {
            return Ok(None);
        }
        let mut pooled = Vec::with_capacity(BATCHES);
        for batch in 0..BATCHES {
            pooled.push(pool_state(&self.kv[batch], &self.score[batch], self.ratio));
        }
        Ok(Some(normalize_rows_f32(
            &pooled,
            BATCHES,
            norm_weight,
            epsilon,
        )))
    }

    fn store(&mut self, batch: usize, slot: usize, row: [f32; WIDTH]) {
        self.kv[batch][slot] = row;
        self.score[batch][slot] = reversed_gate_score(row);
    }
}

fn input(bits: &[u16], positions: usize, batch: usize, position: usize) -> [f32; WIDTH] {
    std::array::from_fn(|feature| {
        bf16_to_f32(bits[(batch * positions + position) * WIDTH + feature])
    })
}

fn pool_group(bits: &[u16], ratio: usize) -> [f32; WIDTH] {
    let mut kv = [[0.0; WIDTH]; 3];
    for (slot, row) in kv.iter_mut().take(ratio).enumerate() {
        *row = input(bits, ratio, 0, slot);
    }
    let score = kv.map(reversed_gate_score);
    pool_state(&kv, &score, ratio)
}

fn reversed_gate_score(row: [f32; WIDTH]) -> [f32; WIDTH] {
    std::array::from_fn(|feature| row[WIDTH - 1 - feature] * 0.75)
}

fn pool_state(kv: &[[f32; WIDTH]; 3], score: &[[f32; WIDTH]; 3], ratio: usize) -> [f32; WIDTH] {
    std::array::from_fn(|feature| {
        let maximum = score[..ratio]
            .iter()
            .map(|row| row[feature])
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = score[..ratio]
            .iter()
            .map(|row| (row[feature] - maximum).exp())
            .sum::<f32>();
        (0..ratio)
            .map(|slot| {
                let probability = (score[slot][feature] - maximum).exp() / denominator;
                kv[slot][feature] * probability
            })
            .sum()
    })
}

fn normalize_rows(input_bf16: &[u16], positions: usize, weight: &[u16], epsilon: f32) -> Vec<u16> {
    input_bf16
        .chunks_exact(WIDTH)
        .take(BATCHES * positions)
        .flat_map(|row| normalize_row(row, weight, epsilon))
        .collect()
}

fn normalize_rows_f32(
    input: &[[f32; WIDTH]],
    rows: usize,
    weight: &[u16],
    epsilon: f32,
) -> Vec<u16> {
    input
        .iter()
        .take(rows)
        .flat_map(|row| {
            let bits = row.map(f32_to_bf16_rne);
            normalize_row(&bits, weight, epsilon)
        })
        .collect()
}

fn normalize_row(input: &[u16], weight: &[u16], epsilon: f32) -> Vec<u16> {
    let mut output = vec![0_u16; WIDTH];
    rms_norm_bf16_reference(input, weight, epsilon, &mut output)
        .expect("fixture scalar RMSNorm request is valid");
    output
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("FP32 high half fits BF16 storage")
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressor-reference.json"
    ))
    .expect("checked-in Compressor fixture")
}

#[test]
fn compressor_prefill_and_sequential_decode_match_pinned_bf16_fixture() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture.source.sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture.source.symbols,
        vec![
            String::from("Compressor.forward"),
            String::from("RMSNorm.forward")
        ]
    );
    assert_eq!(fixture.receipt.torch, "2.13.0");
    assert_eq!(fixture.receipt.device, "cpu");
    assert_eq!((fixture.batches, fixture.width), (BATCHES, WIDTH));
    assert_eq!(fixture.cases.len(), 9);

    for case in fixture.cases {
        let mut state = CompressorState::new(case.ratio).expect("fixture ratio");
        let mut combined: [Vec<u16>; BATCHES] = std::array::from_fn(|_| Vec::new());
        for call in &case.calls {
            let actual = state
                .forward(
                    &call.input_bf16,
                    call.positions,
                    call.start,
                    &fixture.norm_weight_bf16,
                    fixture.epsilon,
                )
                .expect("fixture sequence is safe");
            assert_eq!(
                actual.as_ref(),
                call.output_bf16.as_ref(),
                "ratio {} prefix {} start {}",
                case.ratio,
                case.prefix,
                call.start
            );
            match (&actual, call.output_shape) {
                (None, None) => {}
                (Some(output), Some([BATCHES, positions, WIDTH])) => {
                    assert_eq!(output.len(), BATCHES * positions * WIDTH);
                    for batch in 0..BATCHES {
                        combined[batch].extend_from_slice(
                            &output[batch * positions * WIDTH..(batch + 1) * positions * WIDTH],
                        );
                    }
                }
                _ => panic!(
                    "fixture output presence/shape disagrees at start {}",
                    call.start
                ),
            }
        }
        let expected_positions = case.full_output_bf16.len() / (BATCHES * WIDTH);
        let mut flattened = Vec::with_capacity(case.full_output_bf16.len());
        for batch in &combined {
            assert_eq!(batch.len(), expected_positions * WIDTH);
            flattened.extend_from_slice(batch);
        }
        assert_eq!(
            flattened, case.full_output_bf16,
            "ratio {} prefix {} combined",
            case.ratio, case.prefix
        );
    }
}

#[test]
fn source_pool_axis_is_observable() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.ratio == 2 && case.prefix == 1)
        .expect("ratio-two sequential fixture");
    let expected = case.calls[1].output_bf16.as_ref().expect("completed group");
    let wrong = wrong_feature_axis_pool(
        &case.calls[0].input_bf16,
        &case.calls[1].input_bf16,
        &fixture.norm_weight_bf16,
        fixture.epsilon,
    );
    assert_ne!(
        wrong, *expected,
        "fixture must distinguish a feature-axis softmax"
    );
}

#[test]
fn sequential_decode_rejects_chunking_and_position_gaps_and_reset_replaces_state() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.ratio == 3 && case.prefix == 1)
        .expect("ratio-three sequential fixture");
    let mut state = CompressorState::new(3).expect("ratio");
    state
        .forward(
            &case.calls[0].input_bf16,
            1,
            0,
            &fixture.norm_weight_bf16,
            fixture.epsilon,
        )
        .expect("prefill");
    assert_eq!(
        state.forward(
            &[0x3f80; BATCHES * 2 * WIDTH],
            2,
            1,
            &fixture.norm_weight_bf16,
            fixture.epsilon
        ),
        Err(CompressorStepError::ChunkedDecode { positions: 2 })
    );
    assert_eq!(
        state.forward(
            &case.calls[2].input_bf16,
            1,
            2,
            &fixture.norm_weight_bf16,
            fixture.epsilon
        ),
        Err(CompressorStepError::NonSequentialDecode {
            expected: 1,
            actual: 2
        })
    );

    let mut dirty = CompressorState::new(3).expect("ratio");
    for call in &case.calls {
        dirty
            .forward(
                &call.input_bf16,
                call.positions,
                call.start,
                &fixture.norm_weight_bf16,
                fixture.epsilon,
            )
            .expect("fixture sequence");
    }
    let mut fresh = CompressorState::new(3).expect("ratio");
    for call in &case.calls[..3] {
        let dirty_output = dirty
            .forward(
                &call.input_bf16,
                call.positions,
                call.start,
                &fixture.norm_weight_bf16,
                fixture.epsilon,
            )
            .expect("reset sequence");
        let fresh_output = fresh
            .forward(
                &call.input_bf16,
                call.positions,
                call.start,
                &fixture.norm_weight_bf16,
                fixture.epsilon,
            )
            .expect("fresh sequence");
        assert_eq!(dirty_output, fresh_output, "reset start {}", call.start);
    }
}

fn wrong_feature_axis_pool(
    first: &[u16],
    second: &[u16],
    weight: &[u16],
    epsilon: f32,
) -> Vec<u16> {
    let mut pooled = Vec::with_capacity(BATCHES * WIDTH);
    for batch in 0..BATCHES {
        let rows = [input(first, 1, batch, 0), input(second, 1, batch, 0)];
        let scores = rows.map(reversed_gate_score);
        pooled.push(std::array::from_fn(|feature| {
            (0..2)
                .map(|slot| {
                    let maximum = scores[slot]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let denominator = scores[slot]
                        .iter()
                        .map(|score| (score - maximum).exp())
                        .sum::<f32>();
                    rows[slot][feature] * (scores[slot][feature] - maximum).exp() / denominator
                })
                .sum()
        }));
    }
    normalize_rows_f32(&pooled, BATCHES, weight, epsilon)
}
