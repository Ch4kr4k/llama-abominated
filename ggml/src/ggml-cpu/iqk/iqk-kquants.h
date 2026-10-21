#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// AVX2 Q4_K/Q5_K/Q6_K MUL_MAT and MUL_MAT_ID kernels ported from ik_llama.cpp (iqk_mul_mat, MIT).
// The activations are quantized to block_q8_2_x4 instead of the vec_dot_type (Q8_K).
// GGML_CPU_IQK=0 disables them.

// true if the iqk path handles src0 of this type and row length
bool ggml_iqk_mul_mat_id_supported(int type, int64_t ne00);

// true if expert tensors of this type should stay out of the repack buffer so that the
// iqk path gets them (it is faster for prompts, repack is a bit faster for decode);
// GGML_CPU_IQK_REPACK=1 keeps them in the repack buffer
bool ggml_iqk_skip_repack(int type, int64_t ne00);

// size of one quantized activation row of ne00 floats (never larger than a Q8_K row)
size_t ggml_iqk_q8_row_size(int64_t ne00);

// quantize one activation row of k floats (k % 128 == 0)
void ggml_iqk_quantize_row_q8(const float * x, void * y, int64_t k);

// one expert: Nx weight rows of A times the Ny activation rows picked by row_mapping
// (pairs of int32 {expert slot, token}), rows Nx split across nth threads
void ggml_iqk_mul_mat_moe(int64_t Nx, int64_t Ny, int64_t ne00, int64_t ne11,
                          int typeA, const void * A, size_t strideA,
                          const void * B, size_t strideB,
                          float * C, size_t nb1, size_t nb2,
                          const void * row_mapping, int ith, int nth);

// dense MUL_MAT: C (Ny rows of Nx floats, row stride strideC floats) = A (Nx weight rows)
// times B (Ny activation rows quantized with ggml_iqk_quantize_row_q8), work split over
// nth threads as ik_llama.cpp's iqk_mul_mat does
void ggml_iqk_mul_mat(int64_t Nx, int64_t Ny, int64_t ne00,
                      int typeA, const void * A, size_t strideA,
                      const void * B, size_t strideB,
                      float * C, size_t strideC, int ith, int nth);

#ifdef __cplusplus
}
#endif
