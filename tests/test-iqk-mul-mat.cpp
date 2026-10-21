// Numerical check of MUL_MAT_ID and MUL_MAT on the CPU backend against an f64 reference
// computed from the dequantized weights. Run once with GGML_CPU_IQK=0 and once
// with GGML_CPU_IQK=1 (the env var is cached per process).
// Covers Q4_K/Q5_K/Q6_K, per-expert token counts below and above 32 (the
// q8_1_r8 path), row counts not a multiple of 8, broadcast activations, and for
// MUL_MAT every way the dense kernel splits work over threads (row/column tiles
// for few rows, column halves, q8_1_r8 conversion, plain row split).
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Case { ggml_type type; int K, N, n_as, n_used, T, ne11; bool skew; };

static double run_case(const Case & c, unsigned seed, int n_threads) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t w_elems = (size_t) c.K * c.N * c.n_as;
    std::vector<float> wf(w_elems);
    for (auto & v : wf) v = nd(rng) * 0.05f;
    for (size_t i = 0; i < w_elems; i += 97) wf[i] *= 8.0f; // outliers
    const size_t row_bytes = ggml_row_size(c.type, c.K);
    std::vector<uint8_t> wq(row_bytes * c.N * c.n_as);
    ggml_quantize_chunk(c.type, wf.data(), wq.data(), 0, (int64_t) c.N * c.n_as, c.K, nullptr);
    std::vector<float> wd(w_elems);
    const auto * tt = ggml_get_type_traits(c.type);
    for (int64_t r = 0; r < (int64_t) c.N * c.n_as; ++r)
        tt->to_float(wq.data() + r * row_bytes, wd.data() + r * c.K, c.K);
    std::vector<float> x((size_t) c.K * c.ne11 * c.T);
    for (auto & v : x) v = nd(rng);
    std::vector<int32_t> ids((size_t) c.n_used * c.T);
    for (int t = 0; t < c.T; ++t) {
        std::vector<int> perm(c.n_as);
        for (int i = 0; i < c.n_as; ++i) perm[i] = i;
        if (!c.skew) std::shuffle(perm.begin(), perm.end(), rng);
        for (int u = 0; u < c.n_used; ++u) ids[(size_t) t * c.n_used + u] = perm[u];
    }
    ggml_init_params ip = { (size_t) 64*1024*1024 + wq.size() + x.size()*8 + (size_t) c.N*c.n_used*c.T*4, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.n_as);
    memcpy(as->data, wq.data(), wq.size());
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.ne11, c.T);
    memcpy(b->data, x.data(), x.size() * sizeof(float));
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.T);
    memcpy(id->data, ids.data(), ids.size() * sizeof(int32_t));
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, id);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    double err2 = 0.0, ref2 = 0.0;
    for (int t = 0; t < c.T; ++t)
        for (int u = 0; u < c.n_used; ++u) {
            const int e = ids[(size_t) t * c.n_used + u];
            const float * xr = x.data() + ((size_t) t * c.ne11 + (c.ne11 == 1 ? 0 : u)) * c.K;
            for (int n = 0; n < c.N; ++n) {
                const float * wr = wd.data() + ((size_t) e * c.N + n) * c.K;
                double ref = 0.0;
                for (int k = 0; k < c.K; ++k) ref += (double) wr[k] * xr[k];
                const float got = ((const float *) out->data)[((size_t) t * c.n_used + u) * c.N + n];
                if (!std::isfinite(got)) { ggml_free(ctx); return INFINITY; }
                err2 += (got - ref) * (got - ref);
                ref2 += ref * ref;
            }
        }
    ggml_free(ctx);
    return err2 / ref2;
}

// dense MUL_MAT: N weight rows of K times T activation rows
static double run_dense(ggml_type type, int K, int N, int T, unsigned seed, int n_threads) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf((size_t) K * N);
    for (auto & v : wf) v = nd(rng) * 0.05f;
    for (size_t i = 0; i < wf.size(); i += 97) wf[i] *= 8.0f; // outliers
    const size_t row_bytes = ggml_row_size(type, K);
    std::vector<uint8_t> wq(row_bytes * N);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, nullptr);
    std::vector<float> wd(wf.size());
    const auto * tt = ggml_get_type_traits(type);
    for (int r = 0; r < N; ++r) tt->to_float(wq.data() + r * row_bytes, wd.data() + (size_t) r * K, K);
    std::vector<float> x((size_t) K * T);
    for (auto & v : x) v = nd(rng);
    ggml_init_params ip = { (size_t) 64*1024*1024 + wq.size() + x.size()*8 + (size_t) N*T*4, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, K, N);
    memcpy(a->data, wq.data(), wq.size());
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    memcpy(b->data, x.data(), x.size() * sizeof(float));
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    double err2 = 0.0, ref2 = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += (double) wd[(size_t) n * K + k] * x[(size_t) t * K + k];
            const float got = ((const float *) out->data)[(size_t) t * N + n];
            if (!std::isfinite(got)) { ggml_free(ctx); return INFINITY; }
            err2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
        }
    }
    ggml_free(ctx);
    return err2 / ref2;
}

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? atoi(argv[1]) : 4;
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    const struct { int T; bool skew; int ne11; } shapes[] = {
        {1,false,1},{3,false,1},{7,false,4},{9,false,1},{13,true,1},{31,true,1},
        {32,true,1},{33,true,4},{64,true,1},{71,true,4},{128,false,1},
    };
    int fails = 0;
    for (ggml_type ty : types) {
        for (const auto & s : shapes) {
            Case c = { ty, 2048, 136, 8, 4, s.T, s.ne11, s.skew };
            const double nmse = run_case(c, 1234u + s.T, n_threads);
            const bool bad = !(nmse < 1e-3);
            fails += bad;
            printf("%-5s T=%-4d skew=%d ne11=%d nmse=%.3e%s\n", ggml_type_name(ty), s.T, s.skew, s.ne11, nmse, bad ? "  FAIL" : "");
        }
        Case c2 = { ty, 2048, 100, 8, 4, 64, 1, true }; // N % 8 != 0
        const double nmse2 = run_case(c2, 99u, n_threads);
        fails += !(nmse2 < 1e-3);
        printf("%-5s N=100 T=64 nmse=%.3e%s\n", ggml_type_name(ty), nmse2, nmse2 < 1e-3 ? "" : "  FAIL");
    }
    // dense: {N, T} chosen for each work split with 4 threads (N/4 < 32 tiles; N/4 <= 16
    // with even T >= 16 splits the columns; T >= 32 with N % 8 == 0 converts to q8_1_r8)
    const struct { int N, T; } dense[] = {
        {64,1},{100,7},{60,16},{60,40},{136,1},{136,9},{256,32},{256,33},{1000,64},{1024,128},
    };
    for (ggml_type ty : types) {
        for (const auto & d : dense) {
            const double nmse = run_dense(ty, 2048, d.N, d.T, 777u + d.N + d.T, n_threads);
            const bool bad = !(nmse < 1e-3);
            fails += bad;
            printf("%-5s dense N=%-5d T=%-4d nmse=%.3e%s\n", ggml_type_name(ty), d.N, d.T, nmse, bad ? "  FAIL" : "");
        }
    }
    printf("%s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
