"""Small independent contract tests for the CPU V4.1 kernel backend."""

from __future__ import annotations

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import v41_cpu_kernels as kernels


class CpuKernelTests(unittest.TestCase):
    def test_fp4_pack_round_trip_preserves_pinned_lane_order(self) -> None:
        values = torch.tensor([[0.5, -4.0, -0.0, 6.0]], dtype=torch.float32)
        packed = kernels.pack_fp4_e2m1x2(values)
        self.assertEqual(packed.tolist(), [[0xE1, 0x78]])
        self.assertEqual(
            kernels.unpack_fp4_e2m1x2(packed).tolist(), [[0.5, -4.0, -0.0, 6.0]]
        )

    def test_act_quant_rounds_scales_and_inplace_reconstructs_bf16(self) -> None:
        x = torch.tensor([[9.25] + [0.0] * 31], dtype=torch.bfloat16)
        codes, scales = kernels.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        self.assertEqual(codes.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(scales.float().item(), 0.03125)
        kernels.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
        self.assertEqual(x.float()[0, 0].item(), 9.0)

    def test_activation_quantization_matches_source_contiguous_staging(self) -> None:
        base = (
            torch.arange(128, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2, 32)
        )
        non_contiguous = base.transpose(0, 1)
        self.assertFalse(non_contiguous.is_contiguous())
        codes, scales = kernels.act_quant(
            non_contiguous, 32, "ue8m0", torch.float8_e8m0fnu
        )
        self.assertEqual(tuple(codes.shape), tuple(non_contiguous.shape))
        self.assertEqual(tuple(scales.shape), (2, 2, 1))

    def test_source_power_of_two_scale_uses_ieee_exponent_not_logarithm(self) -> None:
        one = torch.tensor([1.0], dtype=torch.float32)
        above_one = torch.nextafter(one, torch.tensor([torch.inf]))
        self.assertEqual(kernels._source_pow2_ceil(one).item(), 1.0)
        self.assertEqual(kernels._source_pow2_ceil(above_one).item(), 2.0)

    def test_fp4_act_quant_rejects_packed_output_but_has_inplace_path(self) -> None:
        x = torch.tensor([[9.0] + [0.0] * 31], dtype=torch.bfloat16)
        with self.assertRaisesRegex(kernels.CpuKernelError, "only in inplace mode"):
            kernels.fp4_act_quant(x.clone(), 32)
        kernels.fp4_act_quant(x, 32, True)
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(x).all())

    def test_fp8_and_fp4_gemm_apply_scales_per_complete_group(self) -> None:
        activation = torch.ones((1, 32), dtype=torch.bfloat16)
        a, a_s = kernels.act_quant(activation, 32, "ue8m0", torch.float8_e8m0fnu)
        fp8_weight = torch.ones((2, 32), dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_scales = torch.tensor([[1.0]], dtype=torch.float32)
        fp8 = kernels.fp8_gemm(a, a_s, fp8_weight, fp8_scales, block_size=32)
        self.assertEqual(fp8.float().tolist(), [[32.0, 32.0]])

        fp4_weight = kernels.pack_fp4_e2m1x2(torch.ones((2, 32), dtype=torch.float32))
        fp4_scales = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        fp4 = kernels.fp4_gemm(a, a_s, fp4_weight, fp4_scales, act_block_size=32)
        self.assertEqual(fp4.float().tolist(), [[32.0, 64.0]])

        runtime_weight = torch.empty((2, 16), dtype=torch.float4_e2m1fn_x2)
        runtime_weight.view(torch.uint8).copy_(fp4_weight)
        self.assertEqual(
            kernels.fp4_gemm(a, a_s, runtime_weight, fp4_scales, act_block_size=32)
            .float()
            .tolist(),
            [[32.0, 64.0]],
        )

    def test_sparse_attention_counts_duplicate_slots_and_sink_is_denominator_only(
        self,
    ) -> None:
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.bfloat16)
        kv = torch.tensor([[[2.0, 3.0], [1.0, -4.0]]], dtype=torch.bfloat16)
        indices = torch.tensor([[[0, 0, 1, -1]]], dtype=torch.int32)
        output = kernels.sparse_attn(
            q, kv, torch.tensor([0.0], dtype=torch.float32), indices, 1.0
        )
        # Two copies of key 0 have weight exp(2), key 1 exp(1), and the sink
        # only expands the denominator.  This is a nonzero duplicate-sensitive probe.
        self.assertAlmostEqual(output.float()[0, 0, 0, 0].item(), 1.742, places=2)
        self.assertAlmostEqual(output.float()[0, 0, 0, 1].item(), 1.812, places=2)
        all_masked = kernels.sparse_attn(
            q,
            kv,
            torch.tensor([4.0], dtype=torch.float32),
            torch.tensor([[[-1]]], dtype=torch.int32),
            1.0,
        )
        self.assertEqual(all_masked.tolist(), torch.zeros_like(q).tolist())

    def test_sparse_attention_keeps_sink_outside_max_and_rounds_only_numerator(
        self,
    ) -> None:
        q = torch.tensor([[[[1.0]]]], dtype=torch.bfloat16)
        kv = torch.tensor([[[0.0], [-0.0625]]], dtype=torch.bfloat16)
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        output = kernels.sparse_attn(
            q, kv, torch.tensor([0.0], dtype=torch.float32), indices, 1.0
        )
        probability = torch.exp(torch.tensor(-0.0625)).to(torch.bfloat16).float()
        expected = (
            probability * -0.0625 / (1.0 + 1.0 + torch.exp(torch.tensor(-0.0625)))
        ).to(torch.bfloat16)
        self.assertEqual(
            int(output.view(torch.int16)[0, 0, 0, 0]) & 0xFFFF,
            int(expected.view(torch.int16)) & 0xFFFF,
        )

        sink_case = kernels.sparse_attn(
            torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16),
            torch.tensor([[[0.0, 1.0]]], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([[[0]]], dtype=torch.int32),
            1.0,
        )
        self.assertEqual(
            int(sink_case.view(torch.int16)[0, 0, 0, 1]) & 0xFFFF,
            int(
                (torch.tensor(1.0) / (1.0 + torch.exp(torch.tensor(1.0))))
                .to(torch.bfloat16)
                .view(torch.int16)
            )
            & 0xFFFF,
        )

    def test_sparse_attention_rescales_across_the_source_64_slot_boundary(self) -> None:
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.bfloat16)
        kv = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.bfloat16)
        together = kernels.sparse_attn(
            q,
            kv,
            torch.tensor([-100.0], dtype=torch.float32),
            torch.tensor([[[0, 1]]], dtype=torch.int32),
            1.0,
        )
        split_indices = torch.full((1, 1, 65), -1, dtype=torch.int32)
        split_indices[0, 0, 0] = 0
        split_indices[0, 0, 64] = 1
        split = kernels.sparse_attn(
            q, kv, torch.tensor([-100.0], dtype=torch.float32), split_indices, 1.0
        )
        self.assertNotEqual(
            int(together.view(torch.int16)[0, 0, 0, 1]),
            int(split.view(torch.int16)[0, 0, 0, 1]),
        )

    def test_hyper_connection_sinkhorn_is_nontrivial_and_normalized(self) -> None:
        mixes = torch.tensor(
            [[[0.2, -0.3, 0.5, -0.4, 0.7, -0.2, 0.1, 0.4]]], dtype=torch.float32
        )
        pre, post, comb = kernels.hc_split_sinkhorn(
            mixes, torch.tensor([1.0, 0.5, 1.5]), torch.zeros(8), 2, 3, 1e-6
        )
        self.assertEqual(tuple(pre.shape), (1, 1, 2))
        self.assertEqual(tuple(post.shape), (1, 1, 2))
        self.assertNotEqual(pre[0, 0, 0].item(), pre[0, 0, 1].item())
        # The pinned algorithm adds epsilon before every normalization, so it
        # approaches but does not claim an exact doubly stochastic matrix.
        self.assertGreater(comb.sum(dim=-1)[0, 0, 0].item(), 0.99)
        self.assertGreater(comb.sum(dim=-2)[0, 0, 0].item(), 0.99)


if __name__ == "__main__":
    unittest.main()
