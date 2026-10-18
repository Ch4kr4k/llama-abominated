// AVX2 Q4_K/Q5_K/Q6_K x Q8_2_X4 MUL_MAT_ID kernels.
//
// Ported from ik_llama.cpp (ggml/src/iqk/iqk_gemm_kquants.cpp, iqk_mul_mat.cpp,
// iqk_common.h, iqk_quantize.cpp):
//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// The idea: unpacking the quants and block scales of a weight row takes time, so a row
// is unpacked once and multiplied with up to 8 activation columns.

#include "iqk-kquants.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)

#include <immintrin.h>

namespace {

#define QK8_2 32

struct block_q8_2 {
    uint16_t d;       // bf16 scale
    uint16_t s;       // int16 sum of the quants
    int8_t   qs[QK8_2];
};
static_assert(sizeof(block_q8_2) == 2*sizeof(uint16_t) + QK8_2, "wrong q8_2 block size/padding");

struct block_q8_2_x4 {
    uint16_t d[8];    // 4 bf16 scales, then 4 int16 sums
    int8_t   qs[4*QK8_2];
};
static_assert(sizeof(block_q8_2_x4) == 4*sizeof(block_q8_2), "wrong q8_2_x4 block size/padding");

// 8 interleaved rows of 32 unsigned 8-bit quants: 8 scales, then 8 mins (fp16)
struct block_q8_1_r8 {
    ggml_half d[16];
    uint8_t   qs[256];
};
static_assert(sizeof(block_q8_1_r8) == 16*sizeof(ggml_half) + 256, "wrong q8_1_r8 block size/padding");

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static inline float fp16_to_fp32(ggml_half h) {
    return _cvtsh_ss(h);
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffff) > 0x7f800000) { // nan
        return (u >> 16) | 64;
    }
    return (u + (0x7fff + ((u >> 16) & 1))) >> 16;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}

static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

static inline void make_q4_scales(const uint8_t * scales8, uint32_t * aux32) {
    const uint16_t * scales = (const uint16_t *)scales8;
    const uint32_t a0 = scales[0] | (scales[1] << 16);
    const uint32_t a1 = scales[2] | (scales[3] << 16);
    const uint32_t a2 = scales[4] | (scales[5] << 16);
    aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
    aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
    aux32[2] = a1 & 0x3f3f3f3f;
    aux32[0] = a0 & 0x3f3f3f3f;
}

struct DataInfo {
    float       * s;
    const char  * cy;
    size_t        bs;
    size_t        by;
    int           cur_y = 0;
    int           ne11;
    const mmid_row_mapping * row_mapping = nullptr;
    size_t        bs2 = 0;

    inline const char * src1_row(int iy) const {
        if (!row_mapping) return cy + (cur_y + iy)*by;
        int i11 = row_mapping[cur_y + iy].i1 % ne11;
        int i12 = row_mapping[cur_y + iy].i2;
        return cy + (i11 + i12*ne11)*by;
    }

    inline void store(int ix, int iy, float result) const {
        *(dst_row(iy) + ix) = result;
    }

    inline void store(int ix, int iy, __m256 result) const {
        _mm256_storeu_ps(dst_row(iy) + ix, result);
    }

    inline float * dst_row(int iy) const {
        if (!row_mapping) return s + (cur_y + iy)*bs;
        int i1 = row_mapping[cur_y + iy].i1;
        int i2 = row_mapping[cur_y + iy].i2;
        return s + i1*bs + i2*bs2;
    }
};

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x);

#define IQK_MAX_NY 8

#define IQK_SET_MUL_MAT_FUNCTIONS_T(kernel, Dequantizer, funcs) \
            funcs[0] = kernel<Dequantizer, 1>;\
            funcs[1] = kernel<Dequantizer, 2>;\
            funcs[2] = kernel<Dequantizer, 3>;\
            funcs[3] = kernel<Dequantizer, 4>;\
            funcs[4] = kernel<Dequantizer, 5>;\
            funcs[5] = kernel<Dequantizer, 6>;\
            funcs[6] = kernel<Dequantizer, 7>;\
            funcs[7] = kernel<Dequantizer, 8>;\

template <int nrc, typename block_q8> struct Q8 {
    constexpr static int nrc_y = nrc;

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

    const block_q8 * y[nrc_y];
};

template <typename Block>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx) : vx(vx), bx(bx) {}
    inline void new_row(int ix) {
        x = (const Block *)((const char *)vx + bx*ix);
    }

    const void *  vx;
    const size_t  bx;
    const Block * x;

    float d;
};

struct Q4Bits_AVX2 {
    inline void prepare(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[2] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0xf);
};

struct DequantizerQ4K_AVX2 final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K_AVX2(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
    }
    Q4Bits_AVX2 bits;
};

struct DequantizerQ5K_AVX2 final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K_AVX2(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        hbits = j == 0 ? _mm256_loadu_si256((const __m256i *)x[i].qh) : _mm256_srli_epi16(hbits, 4);
        apply_hbits();
    }
    inline void apply_hbits() {
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 3), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh));
    }

    const __m256i mh = _mm256_set1_epi8(0x10);
    Q4Bits_AVX2 bits;
    __m256i hbits;
};

struct DequantizerQ6K_AVX2 final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K_AVX2(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    inline void prepare(int i, int j) {
        auto lbits1 = _mm256_loadu_si256((const __m256i *)x[i].ql + 2*j+0);
        auto lbits2 = _mm256_loadu_si256((const __m256i *)x[i].ql + 2*j+1);
        auto hbits  = _mm256_loadu_si256((const __m256i *)x[i].qh + j);
        bits.values[0] = _mm256_or_si256(_mm256_and_si256(lbits1, bits.ml), _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm256_or_si256(_mm256_and_si256(lbits2, bits.ml), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), bits.ml), _mm256_and_si256(hbits, mh));
        bits.values[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), bits.ml), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), mh));
    }
    inline void prepare_signed(int i, int j, __m256i * us) {
        prepare(i, j);
        for (int k = 0; k < 4; ++k) {
            bits.values[k] = _mm256_add_epi8(bits.values[k], _mm256_set1_epi8(-32));
            us[k] = _mm256_sign_epi8(bits.values[k], bits.values[k]);
        }
    }
    inline __m256i make_scales(int i) const {
        return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)x[i].scales));
    }

    const __m256i mh = _mm256_set1_epi8(0x30);
    Q4Bits_AVX2 bits;
};

// q4_K, q5_K: unsigned quants with per sub-block scales and mins
template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_2_X4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_2_x4> q8(info);

    Dequantizer deq(vx, bx);

    uint32_t utmp[4];
    __m256  accd[nrc_y];
    __m256  scales[2];
    float   d8[8*nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq.d = fp16_to_fp32(deq.x[i].data.data.d);
            auto vm = _mm256_cvtph_ps(_mm_set1_epi16(deq.x[i].data.data.dmin));
            make_q4_scales(deq.x[i].scales, utmp);
            auto mins = _mm256_mul_ps(vm, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(utmp + 2)))));
            mins = _mm256_mul_ps(_mm256_set1_ps(-1.f), mins);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+0].d)));
                auto d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+1].d)));
                auto dy = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(d4_2, d4_1), 16));
                _mm256_storeu_ps(d8 + 8*iy, dy);
                auto m4_1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+0].d+4)));
                auto m4_2 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+1].d+4)));
                auto myi  = MM256_SET_M128I(m4_2, m4_1);
                auto my   = _mm256_mul_ps(dy, _mm256_cvtepi32_ps(myi));
                accd[iy]  = _mm256_fmadd_ps(my, mins, accd[iy]);
            }

            auto all_scales = _mm256_mul_ps(_mm256_set1_ps(deq.d), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)utmp))));
            scales[0] = _mm256_set_m128(_mm256_castps256_ps128(all_scales), _mm256_castps256_ps128(all_scales));
            auto scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[1] = _mm256_set_m128(scales_h, scales_h);

            for (int j = 0; j < QK_K/128; ++j) {

                deq.prepare(i, j);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4& y = q8.y[iy][2*i+j];
                    auto sumi1 = _mm256_maddubs_epi16(deq.bits.values[0], _mm256_loadu_si256((const __m256i*)y.qs+0));
                    auto sumi2 = _mm256_maddubs_epi16(deq.bits.values[1], _mm256_loadu_si256((const __m256i*)y.qs+1));
                    auto sumi3 = _mm256_maddubs_epi16(deq.bits.values[2], _mm256_loadu_si256((const __m256i*)y.qs+2));
                    auto sumi4 = _mm256_maddubs_epi16(deq.bits.values[3], _mm256_loadu_si256((const __m256i*)y.qs+3));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2));
                    sumi3 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi64(sumi1, sumi3), _mm256_unpackhi_epi64(sumi1, sumi3));
                    sumi1 = _mm256_madd_epi16(_mm256_set1_epi16(1), sumi1);
                    auto dy4 = _mm_loadu_ps(d8 + 8*iy + 4*j);
                    auto d4d8 = _mm256_mul_ps(scales[j], _mm256_set_m128(dy4, dy4));
                    accd[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi1), accd[iy]);
                }

            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }
}

// q6_K: signed quants with per sub-block scales
template <typename Dequantizer, int nrc_y>
static void mul_mat_qY_K_q8_2_X4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_2_x4> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accd[nrc_y];
    __m256  scales[2];
    float   d8[8*nrc_y];
    __m256i us[4];

    uint8_t k_shuff[32] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq.d = fp16_to_fp32(deq.x[i].d);
            auto vd = _mm256_set1_ps(deq.d);
            auto sc16 = _mm256_shuffle_epi8(deq.make_scales(i), shuff);
            scales[0] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sc16))));
            scales[1] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sc16, 1))));
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+0].d)));
                auto d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+1].d)));
                auto dy = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(d4_2, d4_1), 16));
                if constexpr (nrc_y == 1) {
                    auto dyh = _mm256_extractf128_ps(dy, 1);
                    scales[0] = _mm256_mul_ps(scales[0], _mm256_set_m128(_mm256_castps256_ps128(dy), _mm256_castps256_ps128(dy)));
                    scales[1] = _mm256_mul_ps(scales[1], _mm256_set_m128(dyh, dyh));
                } else {
                    _mm256_storeu_ps(d8 + 8*iy, dy);
                }
            }

            for (int j = 0; j < QK_K/128; ++j) {

                deq.prepare_signed(i, j, us);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto qs = q8.y[iy][2*i+j].qs;
                    auto sumi1 = _mm256_maddubs_epi16(us[0], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+0), deq.bits.values[0]));
                    auto sumi2 = _mm256_maddubs_epi16(us[1], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+1), deq.bits.values[1]));
                    auto sumi3 = _mm256_maddubs_epi16(us[2], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+2), deq.bits.values[2]));
                    auto sumi4 = _mm256_maddubs_epi16(us[3], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+3), deq.bits.values[3]));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2));
                    sumi3 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi64(sumi1, sumi3), _mm256_unpackhi_epi64(sumi1, sumi3));
                    sumi1 = _mm256_madd_epi16(_mm256_set1_epi16(1), sumi1);
                    if constexpr (nrc_y > 1) {
                        auto dy4 = _mm_loadu_ps(d8 + 8*iy + 4*j);
                        auto d4d8 = _mm256_mul_ps(scales[j], _mm256_set_m128(dy4, dy4));
                        accd[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi1), accd[iy]);
                    } else {
                        accd[iy] = _mm256_fmadd_ps(scales[j], _mm256_cvtepi32_ps(sumi1), accd[iy]);
                    }
                }

            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }
}

// q4_K, q5_K rows converted to q8_1_r8 (see the converters below) x q8_2_x4
template <int nrc_y>
static void mul_mat_q8_1_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    int nb = n / QK8_2;
    __m256 acc[nrc_y] = {};
    float d8[4*nrc_y];
    __m256i qx[4];
    auto dot = [&qx] (const int8_t * qy) {
        auto y128 = _mm_loadu_si128((const __m128i*)qy);
        auto y = MM256_SET_M128I(y128, y128);
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        return _mm256_add_epi32(_mm256_madd_epi16(_mm256_set1_epi16(1), sumi1), _mm256_madd_epi16(_mm256_set1_epi16(1), sumi2));
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_1_r8 * iq8 = (const block_q8_1_r8 *)((const char *)vx + ix*bx);
        for (int i4 = 0; i4 < nb/4; ++i4) {
            {
                // mins times the activation block sums
                __m256 mx[4];
                for (int ib32 = 0; ib32 < 4; ++ib32) mx[ib32] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d+1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][i4].d)), 16));
                    _mm_storeu_ps(d8 + 4*iy + 0, scales);
                    auto bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][i4].d+4))));
                    bsums4 = _mm_mul_ps(bsums4, scales);
                    auto bsums  = _mm256_set_m128(bsums4, bsums4);
                    acc[iy] = _mm256_fmadd_ps(mx[0], _mm256_shuffle_ps(bsums, bsums, 0x00), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[1], _mm256_shuffle_ps(bsums, bsums, 0x55), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[2], _mm256_shuffle_ps(bsums, bsums, 0xaa), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[3], _mm256_shuffle_ps(bsums, bsums, 0xff), acc[iy]);
                }
            }
            for (int ib32 = 0; ib32 < 4; ++ib32) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d));
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][i4].qs+32*ib32);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+4+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][i4].qs+32*ib32+16);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

// unpack 8 rows of q4_K (with_qh = false) or q5_K into q8_1_r8, lossless
template <typename Block, bool with_qh>
static void convert_qX_k_q8_1_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const Block * x8[8];

    block_q8_1_r8 * y = (block_q8_1_r8 *)vy;

    ggml_half dh[16];
    uint16_t all_ls[128];

    uint32_t utmp[4];
    const uint8_t * u8 = (const uint8_t *)utmp;
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                dh[k+0] = x8[k][i].data.data.d;
                dh[k+8] = x8[k][i].data.data.dmin;
                make_q4_scales(x8[k][i].scales, utmp);
                auto qs  = x8[k][i].qs;
                __m256i hbits;
                if constexpr (with_qh) {
                    hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                }
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    all_ls[8*(2*ib64 + 0) + k     ] = u8[2*ib64+0];
                    all_ls[8*(2*ib64 + 1) + k     ] = u8[2*ib64+1];
                    all_ls[8*(2*ib64 + 0) + k + 64] = u8[2*ib64+8];
                    all_ls[8*(2*ib64 + 1) + k + 64] = u8[2*ib64+9];
                    auto bits = _mm256_loadu_si256((const __m256i *)qs+ib64);
                    auto values1 = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    auto values2 = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    if constexpr (with_qh) {
                        values1 = _mm256_or_si256(values1, _mm256_and_si256(_mm256_set1_epi8(0x10), _mm256_slli_epi16(hbits, 4)));
                        values2 = _mm256_or_si256(values2, _mm256_and_si256(_mm256_set1_epi8(0x10), _mm256_slli_epi16(hbits, 3)));
                        hbits = _mm256_srli_epi16(hbits, 2);
                    }
                    _mm256_storeu_si256((__m256i *)block, values1);
                    auto q8 = (uint32_t *)y[2*ib64+0].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                    _mm256_storeu_si256((__m256i *)block, values2);
                    q8 = (uint32_t *)y[2*ib64+1].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            auto vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh+0));
            auto vm = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh+1));
            vm = _mm256_mul_ps(_mm256_set1_ps(-1.f), vm);
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                auto iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32);
                auto iscales32 = _mm256_cvtepi16_epi32(iscales16);
                auto scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d+0, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32 + 8);
                iscales32 = _mm256_cvtepi16_epi32(iscales16);
                scales = _mm256_mul_ps(vm, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d+1, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
}

#define IQK_SET_MUL_MAT_FUNCTIONS(kernel, funcs) \
            funcs[0] = kernel<1>;\
            funcs[1] = kernel<2>;\
            funcs[2] = kernel<3>;\
            funcs[3] = kernel<4>;\
            funcs[4] = kernel<5>;\
            funcs[5] = kernel<6>;\
            funcs[6] = kernel<7>;\
            funcs[7] = kernel<8>;\

struct MulMat {
    std::array<mul_mat_t, IQK_MAX_NY> funcs = {};

    void prepare_q8_1_r8() {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_1_r8_q8_2, funcs);
    }

    bool prepare(int typeA) {
        switch (typeA) {
            case GGML_TYPE_Q4_K:
                IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, DequantizerQ4K_AVX2, funcs);
                return true;
            case GGML_TYPE_Q5_K:
                IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, DequantizerQ5K_AVX2, funcs);
                return true;
            case GGML_TYPE_Q6_K:
                IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qY_K_q8_2_X4_T, DequantizerQ6K_AVX2, funcs);
                return true;
            default:
                return false;
        }
    }

    // walk the nrc_y columns in steps of up to IQK_MAX_NY, rows in tiles of k_x_step
    inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
        constexpr int k_x_step = 64;
        const int ny = IQK_MAX_NY;
        int n_left = nrc_y - info.cur_y;
        int n_step = n_left/ny;
        if (n_step > 0) {
            if (n_step*ny != n_left) {
                // spread the columns evenly over n_step+1 passes of ny1 or ny1+1 columns
                ++n_step;
                int ny1 = n_left/n_step;
                int ny2 = ny1 + 1;
                int my1 = n_step*ny2 - n_left;
                int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < my1; ++iy) {
                        funcs[ny1-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        funcs[ny2-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            } else {
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < n_step; ++iy) {
                        funcs[ny-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }
        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            funcs[n_left-1](n, vx, bx, info, nrc_x);
        }
    }
};

static bool iqk_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CPU_IQK");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

} // namespace

bool ggml_iqk_mul_mat_id_supported(int type, int64_t ne00) {
    if (!iqk_enabled() || ne00 % QK_K != 0) {
        return false;
    }
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

bool ggml_iqk_skip_repack(int type, int64_t ne00) {
    static const bool keep_repack = [] {
        const char * env = getenv("GGML_CPU_IQK_REPACK");
        return env != nullptr && atoi(env) != 0;
    }();
    return !keep_repack && ggml_iqk_mul_mat_id_supported(type, ne00);
}

size_t ggml_iqk_q8_row_size(int64_t ne00) {
    return (ne00/QK8_2)*sizeof(block_q8_2);
}

void ggml_iqk_quantize_row_q8(const float * x, void * vy, int64_t k) {
    assert(k % QK8_2 == 0);
    const int nb = k / QK8_2;

    const int nb4 = 4*(nb/4);
    block_q8_2    * y  = (block_q8_2 *)vy;
    block_q8_2_x4 * y4 = (block_q8_2_x4 *)vy;
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // the scale is stored as bf16, quantize with the rounded value
        float d = max_scalar / 127.f;
        const uint16_t t = fp32_to_bf16(d);
        d = bf16_to_fp32(t);
        if (i < nb4) {
            y4[i4].d[ir] = t;
        } else {
            y[i].d = t;
        }
        const float id = d > 0 ? 1/d : 0.f;
        const __m256 mul = _mm256_set1_ps( id );

        v0 = _mm256_round_ps( _mm256_mul_ps( v0, mul ), _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( _mm256_mul_ps( v1, mul ), _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( _mm256_mul_ps( v2, mul ), _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( _mm256_mul_ps( v3, mul ), _MM_ROUND_NEAREST );

        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // sum of the quants, used for the q4_K/q5_K mins
        const int isum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        const int16_t isum16 = (int16_t) isum;
        if (i < nb4) {
            memcpy(&y4[i4].d[ir+4], &isum16, sizeof(isum16));
        } else {
            memcpy(&y[i].s, &isum16, sizeof(isum16));
        }

        // int32 -> int16 -> int8, then fix the lane order of the packs
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
}

void ggml_iqk_mul_mat_moe(int64_t Nx, int64_t Ny, int64_t ne00, int64_t ne11,
                          int typeA, const void * A, size_t strideA,
                          const void * B, size_t strideB,
                          float * C, size_t nb1, size_t nb2,
                          const void * row_mapping, int ith, int nth) {
    MulMat mm;

    // with many activation rows, converting the weights to q8_1_r8 first pays off
    if (Ny >= 32 && Nx % 8 == 0 && (typeA == GGML_TYPE_Q4_K || typeA == GGML_TYPE_Q5_K)) {
        constexpr int k_x_step = 32;
        mm.prepare_q8_1_r8();
        int64_t nrc_x   = 8*((Nx/8 + nth - 1)/nth);
        int64_t first_x = ith*nrc_x;
        if (first_x >= Nx) {
            return;
        }
        if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;
        const size_t row_size_qx = (ne00/QK8_2)*sizeof(block_q8_2);
        DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float), strideB, 0, (int) ne11,
                      (const mmid_row_mapping *)row_mapping, nb2/sizeof(float)};
        thread_local std::vector<char> work;
        if (work.size() < row_size_qx*k_x_step) work.resize(row_size_qx*k_x_step);
        for (int64_t ix = 0; ix < nrc_x; ix += k_x_step) {
            auto this_info = info;
            this_info.s += ix;
            const int this_nrc_x = (int) (ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix);
            const char * a = (const char *)A + (first_x + ix)*strideA;
            if (typeA == GGML_TYPE_Q4_K) {
                convert_qX_k_q8_1_r8<block_q4_K, false>((int) ne00, a, strideA, work.data(), this_nrc_x);
            } else {
                convert_qX_k_q8_1_r8<block_q5_K, true>((int) ne00, a, strideA, work.data(), this_nrc_x);
            }
            mm.mul_mat_NxM((int) ne00, work.data(), row_size_qx, this_info, this_nrc_x, (int) Ny);
        }
        return;
    }

    if (!mm.prepare(typeA)) {
        GGML_ABORT("iqk: unsupported type %d", typeA);
    }
    const int64_t nrc_x   = (Nx + nth - 1)/nth;
    const int64_t first_x = ith*nrc_x;
    if (first_x >= Nx) {
        return;
    }
    const int this_nrc_x = (int) (first_x + nrc_x <= Nx ? nrc_x : Nx - first_x);
    DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float), strideB, 0, (int) ne11,
                  (const mmid_row_mapping *)row_mapping, nb2/sizeof(float)};
    mm.mul_mat_NxM((int) ne00, (const char *)A + strideA*first_x, strideA, info, this_nrc_x, (int) Ny);
}

#else

bool ggml_iqk_mul_mat_id_supported(int type, int64_t ne00) {
    (void) type;
    (void) ne00;
    return false;
}

bool ggml_iqk_skip_repack(int type, int64_t ne00) {
    (void) type;
    (void) ne00;
    return false;
}

size_t ggml_iqk_q8_row_size(int64_t ne00) {
    (void) ne00;
    return 0;
}

void ggml_iqk_quantize_row_q8(const float * x, void * y, int64_t k) {
    (void) x; (void) y; (void) k;
    GGML_ABORT("iqk: not available");
}

void ggml_iqk_mul_mat_moe(int64_t Nx, int64_t Ny, int64_t ne00, int64_t ne11,
                          int typeA, const void * A, size_t strideA,
                          const void * B, size_t strideB,
                          float * C, size_t nb1, size_t nb2,
                          const void * row_mapping, int ith, int nth) {
    (void) Nx; (void) Ny; (void) ne00; (void) ne11; (void) typeA; (void) A; (void) strideA;
    (void) B; (void) strideB; (void) C; (void) nb1; (void) nb2; (void) row_mapping; (void) ith; (void) nth;
    GGML_ABORT("iqk: not available");
}

#endif
