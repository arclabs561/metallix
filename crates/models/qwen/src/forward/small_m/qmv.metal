// Few-row affine 4-bit matmul: out[m, n] = sum_k x[m, k] * (s * q + b),
// for 2..16 activation rows, on 8x8 simdgroup matrices.
//
// Mechanism from llama.cpp's few-row MMA mat-mul (ggml-org/llama.cpp PR
// 29869, MIT): a threadgroup owns 8 * NT output columns by 8 * RT activation
// rows, reads every weight once for all of them, and its NSG simdgroups
// split K. Written for MLX's affine layout: `w` is [N, K / 8] uint32 with
// value i of a word in bits 4i..4i+3; `scales` and `biases` are [N, K / GS].
// MLX's custom-kernel API wraps this body with the generated signature.
//
// Within a 32-value chunk, column c of MMA step s is k = 8 * (c / 2) +
// 4 * (c % 2) + s. A lane holding fragment row fm and columns fn, fn + 1
// then reads one packed word per output tile (its two columns over the
// four steps) and four consecutive activations per row.
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tile = threadgroup_position_in_grid.x;
    uint K = x_shape[1];
    uint N = (uint)NOut;
    uint chunks = K / 32;
    uint words = K / 8;
    uint groups = K / (uint)GroupSize;
    // The lane's slot in an 8x8 simdgroup matrix: row fm, columns fn, fn + 1.
    uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    uint n0 = tile * 8 * (uint)NT;

    const device uint32_t* wr[NT];
    const device T* sr[NT];
    const device T* br[NT];
    for (int t = 0; t < NT; t++) {
        // Rows past N repeat the last row; their outputs are not stored.
        uint n = min(n0 + 8 * t + fm, N - 1);
        wr[t] = w + n * words + fn / 2;
        sr[t] = scales + n * groups;
        br[t] = biases + n * groups;
    }
    const device T* xr[RT][2];
    for (int rt = 0; rt < RT; rt++) {
        for (int e = 0; e < 2; e++) {
            uint m = min(uint(8 * rt) + fn + e, uint(M - 1));
            xr[rt][e] = x + m * K + 8 * (fm / 2) + 4 * (fm % 2);
        }
    }

    simdgroup_float8x8 acc[RT][NT];
    for (int rt = 0; rt < RT; rt++) {
        for (int t = 0; t < NT; t++) {
            acc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }
    for (uint c = sg; c < chunks; c += (uint)NSG) {
        uint kb = c * 32;
        uint g = kb / (uint)GroupSize;
        simdgroup_float8x8 b[RT][4];
        for (int rt = 0; rt < RT; rt++) {
            vec<T, 4> x0 = *((const device vec<T, 4>*)(xr[rt][0] + kb));
            vec<T, 4> x1 = *((const device vec<T, 4>*)(xr[rt][1] + kb));
            for (int s = 0; s < 4; s++) {
                b[rt][s].thread_elements()[0] = float(x0[s]);
                b[rt][s].thread_elements()[1] = float(x1[s]);
            }
        }
        for (int t = 0; t < NT; t++) {
            uint32_t word = wr[t][c * 4];
            float scale = float(sr[t][g]);
            float bias = float(br[t][g]);
            for (int s = 0; s < 4; s++) {
                simdgroup_float8x8 a;
                a.thread_elements()[0] = scale * float((word >> (4 * s)) & 0xfu) + bias;
                a.thread_elements()[1] = scale * float((word >> (4 * s + 16)) & 0xfu) + bias;
                for (int rt = 0; rt < RT; rt++) {
                    simdgroup_multiply_accumulate(acc[rt][t], a, b[rt][s], acc[rt][t]);
                }
            }
        }
    }

    // Add up the simdgroups' K slices and store the tile.
    threadgroup float red[NSG * RT * NT * 64];
    for (int rt = 0; rt < RT; rt++) {
        for (int t = 0; t < NT; t++) {
            uint base = ((sg * RT + rt) * NT + t) * 64 + 2 * lane;
            red[base] = acc[rt][t].thread_elements()[0];
            red[base + 1] = acc[rt][t].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = sg * 32 + lane; idx < uint(RT * NT * 64); idx += (uint)NSG * 32) {
        float sum = 0.0f;
        for (int k = 0; k < NSG; k++) {
            sum += red[k * (RT * NT * 64) + idx];
        }
        uint rt = idx / (NT * 64);
        uint t = (idx / 64) % NT;
        uint l = (idx % 64) / 2;
        uint e = idx % 2;
        uint n = n0 + 8 * t + ((l / 4) & 4) + ((l / 2) % 4);
        uint m = 8 * rt + ((l / 4) & 2) * 2 + (l % 2) * 2 + e;
        if (n < N && m < uint(M)) {
            out[m * N + n] = T(sum);
        }
    }
