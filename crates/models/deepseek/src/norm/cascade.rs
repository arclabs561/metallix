//! Torch CPU float32 cascade summation, used where source parity depends on it.
//!
//! The pinned source computes `x.square().mean(-1)` with `ATen`'s CPU
//! `cascade_sum`: four-lane vectors, four independent row accumulators, and a
//! four-level cascade whose level width is `max(4, ceil_log2(n) / 4)`. Ordered
//! scalar accumulation rounds differently on real checkpoint rows. This mirrors
//! the arithmetic order only; it does not claim GPU or multithreaded parity.

const LANES: usize = 4;
const ROWS: usize = 4;
const LEVELS: usize = 4;

type Lane = [f32; LANES];
type Rows = [Lane; ROWS];

fn add(target: &mut Lane, value: &Lane) {
    for (left, right) in target.iter_mut().zip(value) {
        *left += *right;
    }
}

fn add_rows(target: &mut Rows, value: &Rows) {
    for (left, right) in target.iter_mut().zip(value) {
        add(left, right);
    }
}

fn level_power(size: usize) -> u32 {
    let ceil_log2 = if size <= 1 {
        0
    } else {
        usize::BITS - (size - 1).leading_zeros()
    };
    (ceil_log2 / 4).max(4)
}

/// Sums FP32 values in Torch's single-threaded CPU cascade order.
pub(crate) fn torch_cpu_sum(values: &[f32]) -> f32 {
    let lane_count = values.len() / LANES;
    let tail = &values[lane_count * LANES..];
    let lanes: Vec<Lane> = values
        .chunks_exact(LANES)
        .map(|chunk| chunk.try_into().expect("exact lane chunk"))
        .collect();
    let group_count = lanes.len() / ROWS;
    let leftover = &lanes[group_count * ROWS..];
    let groups: Vec<Rows> = lanes
        .chunks_exact(ROWS)
        .map(|chunk| chunk.try_into().expect("exact row chunk"))
        .collect();
    let power = level_power(groups.len());
    let step = 1_usize << power;
    let mask = step - 1;
    let mut levels = [[[0.0_f32; LANES]; ROWS]; LEVELS];
    let mut index = 0;
    while index + step <= groups.len() {
        for group in &groups[index..index + step] {
            add_rows(&mut levels[0], group);
        }
        index += step;
        let mut shift = power;
        for level in 1..LEVELS {
            let (lower, upper) = levels.split_at_mut(level);
            add_rows(&mut upper[0], &lower[level - 1]);
            lower[level - 1] = [[0.0; LANES]; ROWS];
            if index & (mask << shift) != 0 {
                break;
            }
            shift += power;
        }
    }
    for group in &groups[index..] {
        add_rows(&mut levels[0], group);
    }
    let (first, rest) = levels.split_at_mut(1);
    for level in rest.iter() {
        add_rows(&mut first[0], level);
    }
    let mut partial = first[0];
    for lane in leftover {
        add(&mut partial[0], lane);
    }
    let (head, others) = partial.split_at_mut(1);
    for lane in others.iter() {
        add(&mut head[0], lane);
    }
    let mut total = 0.0_f32;
    for &value in tail {
        total += value;
    }
    for value in head[0] {
        total += value;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::torch_cpu_sum;

    fn bits(value: f32) -> u32 {
        value.to_bits()
    }

    #[test]
    fn matches_exact_sums_and_torch_grouping() {
        assert_eq!(bits(torch_cpu_sum(&[])), bits(0.0));
        assert_eq!(bits(torch_cpu_sum(&vec![1.0; 5120])), bits(5120.0));
        assert_eq!(bits(torch_cpu_sum(&[0.5, 0.25, 0.125])), bits(0.875));
        // Serial order absorbs every 1.0 into 2^24; Torch 2.13 CPU keeps the
        // other lanes' terms and returns 16_777_276 for this input.
        let mut values = vec![1.0_f32; 64];
        values[0] = 16_777_216.0;
        let serial: f32 = values.iter().sum();
        assert_eq!(bits(serial), bits(16_777_216.0));
        assert_eq!(bits(torch_cpu_sum(&values)), bits(16_777_276.0));
    }
}
