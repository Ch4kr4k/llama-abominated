// Numerical check of MUL_MAT_ID on the CPU backend against an f64 reference
// computed from the dequantized weights. The kernel choice (GGML_CPU_IQK) is
// read once per process, so without it set the test runs itself twice, once
// with the stock kernels and once with the iqk kernels.
//
// Covers: Q4_K / Q5_K / Q6_K, per-expert token counts below and above 32
// (the q8_1_r8 conversion path), column counts that are not multiples of 8,
// and broadcast activations (ne11 == 1) as used by ffn_up/gate.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Case {
    ggml_type type;
    int K;        // ne00
    int N;        // ne01 (rows per expert)
    int n_as;     // experts
    int n_used;   // experts per token
    int T;        // tokens
    int ne11;     // 1 = broadcast activations, else n_used
    bool skew;    // route every token to the same experts (forces >= 32 rows/expert)
};

static double run_case(const Case & c, unsigned seed, int n_threads) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const size_t w_elems = (size_t) c.K * c.N * c.n_as;
    std::vector<float> wf(w_elems);
    for (auto & v : wf) v = nd(rng) * 0.05f;
    // a few outliers per row, as in real weights
    for (size_t i = 0; i < w_elems; i += 97) wf[i] *= 8.0f;

    const size_t row_bytes = ggml_row_size(c.type, c.K);
    std::vector<uint8_t> wq(row_bytes * c.N * c.n_as);
    ggml_quantize_chunk(c.type, wf.data(), wq.data(), 0, (int64_t) c.N * c.n_as, c.K, nullptr);

    // dequantized weights: the reference uses exactly what the kernel sees
    std::vector<float> wd(w_elems);
    const auto * tt = ggml_get_type_traits(c.type);
    for (int64_t r = 0; r < (int64_t) c.N * c.n_as; ++r) {
        tt->to_float(wq.data() + r * row_bytes, wd.data() + r * c.K, c.K);
    }

    std::vector<float> x((size_t) c.K * c.ne11 * c.T);
    for (auto & v : x) v = nd(rng);

    std::vector<int32_t> ids((size_t) c.n_used * c.T);
    for (int t = 0; t < c.T; ++t) {
        std::vector<int> perm(c.n_as);
        for (int i = 0; i < c.n_as; ++i) perm[i] = i;
        if (!c.skew) std::shuffle(perm.begin(), perm.end(), rng);
        for (int u = 0; u < c.n_used; ++u) ids[(size_t) t * c.n_used + u] = perm[u];
    }

    ggml_init_params ip = { (size_t) 64 * 1024 * 1024 + wq.size() + x.size() * 4 * 2 + (size_t) c.N * c.n_used * c.T * 4, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.n_as);
    memcpy(as->data, wq.data(), wq.size());
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.ne11, c.T);
    memcpy(b->data, x.data(), x.size() * sizeof(float));
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.T);
    memcpy(id->data, ids.data(), ids.size() * sizeof(int32_t));

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, id); // [N, n_used, T]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    double err2 = 0.0, ref2 = 0.0;
    for (int t = 0; t < c.T; ++t) {
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
    }
    ggml_free(ctx);
    return err2 / ref2; // NMSE vs the dequantized-weight reference (activation quantization error only)
}

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? atoi(argv[1]) : 2;
    const char * mode = getenv("GGML_CPU_IQK");
    if (mode == nullptr) {
        // run both kernel families, each in its own process
        int rc = 0;
        for (const char * m : { "0", "1" }) {
            std::string cmd = std::string("GGML_CPU_IQK=") + m + " '" + argv[0] + "' " + std::to_string(n_threads);
            const int status = std::system(cmd.c_str());
            rc |= status != 0;
        }
        return rc;
    }
    printf("GGML_CPU_IQK=%s threads=%d\n", mode ? mode : "(unset)", n_threads);

    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    // T, skew, ne11: per-expert rows = T (skewed) or ~T*n_used/n_as
    const struct { int T; bool skew; int ne11; } shapes[] = {
        {1, false, 1}, {3, false, 1}, {7, false, 4}, {9, false, 1},
        {13, true, 1}, {31, true, 1}, {32, true, 1}, {33, true, 4},
        {64, true, 1}, {71, true, 4}, {128, false, 1},
    };
    int fails = 0;
    for (ggml_type ty : types) {
        for (const auto & s : shapes) {
            Case c = { ty, 2048, 136, 8, 4, s.T, s.ne11, s.skew }; // N=136: not a multiple of 64, multiple of 8
            const double nmse = run_case(c, 1234u + s.T, n_threads);
            // q8 activation quantization alone gives ~1e-5..1e-4; anything >1e-3 is a kernel bug
            const bool bad = !(nmse < 1e-3);
            fails += bad;
            printf("%-5s T=%-4d skew=%d ne11=%d  nmse=%.3e %s\n", ggml_type_name(ty), s.T, s.skew, s.ne11, nmse, bad ? "  <-- FAIL" : "");
        }
        // N not a multiple of 8 (r8 path must not be taken, scalar tails exercised)
        Case c2 = { ty, 2048, 100, 8, 4, 64, 1, true };
        const double nmse2 = run_case(c2, 99u, n_threads);
        const bool bad2 = !(nmse2 < 1e-3);
        fails += bad2;
        printf("%-5s N=100 T=64 skew   nmse=%.3e %s\n", ggml_type_name(ty), nmse2, bad2 ? "  <-- FAIL" : "");
    }
    printf("%s (%d failing cases)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
