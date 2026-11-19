#include "bitnet.hpp"
#include "behemoth/ternary.hpp"
#include "kernels.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace bitnet {
using gpu::check;
namespace {
const char *linear_names[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
std::vector<float> to_float(const qtip::SafeTensorsReader &r, const std::string &name, size_t expect) {
    const auto &info = r.info(name);
    require(size_t(info.numel()) == expect, "unexpected size of " + name);
    std::vector<float> out(expect);
    if (info.dtype == "BF16") {
        std::vector<uint16_t> raw(expect);
        r.read_into(name, raw.data(), raw.size() * 2);
        for (size_t i = 0; i < expect; ++i)
            out[i] = std::bit_cast<float>(uint32_t(raw[i]) << 16);
    } else if (info.dtype == "F32") {
        r.read_into(name, out.data(), out.size() * 4);
    } else if (info.dtype == "F16") {
        std::vector<uint16_t> raw(expect);
        r.read_into(name, raw.data(), raw.size() * 2);
        for (size_t i = 0; i < expect; ++i)
            out[i] = behemoth::decode_checkpoint_f16(raw[i]);
    } else
        require(false, "unsupported dtype " + info.dtype + " for " + name);
    return out;
}
std::vector<uint16_t> to_bf16_host(const std::vector<float> &x) {
    std::vector<uint16_t> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = behemoth::BFloat16(x[i]).bits;
    return out;
}
template <class T> void upload(Dev<T> &d, const std::vector<T> &h, cudaStream_t s) {
    require(d.n == h.size(), "upload size mismatch");
    check(cudaMemcpyAsync(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, s), "upload");
}
template <class T> std::vector<T> download(const T *p, size_t n, cudaStream_t s) {
    std::vector<T> h(n);
    check(cudaMemcpyAsync(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, s), "download");
    check(cudaStreamSynchronize(s), "sync");
    return h;
}
} // namespace

template <class T> void Dev<T>::reset(size_t count) {
    free();
    n = count;
    if (count)
        check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
}
template <class T> void Dev<T>::free() {
    if (p)
        cudaFree(p);
    p = nullptr;
    n = 0;
}
template struct Dev<float>;
template struct Dev<uint16_t>;
template struct Dev<int32_t>;
template struct Dev<double>;

Geometry geometry(const json &c) {
    qat::Config q(c);
    Geometry g{q.n, q.inter, q.heads, q.kv, q.dim, q.vocab, q.layers, q.eps, q.theta,
               c.value("tie_word_embeddings", false)};
    require(g.n % 16 == 0 && g.inter % 16 == 0 && (g.heads * g.dim) % 16 == 0 && (g.kv * g.dim) % 16 == 0,
            "BitLinear inputs must be multiples of 16");
    require(g.dim <= 256, "head_dim above 256 unsupported");
    return g;
}

Model::Model(const Geometry &geo, bool q, bool sub, bool train) : g(geo), quant(q), subln(sub), trainable(train) {
    require(!quant || trainable, "a ternary model is always trainable");
    layers.resize(size_t(g.layers));
    size_t off = 0;
    for (int l = 0; l < g.layers; ++l) {
        const int outs[7] = {g.heads * g.dim, g.kv * g.dim, g.kv * g.dim, g.n, g.inter, g.inter, g.n};
        const int ins[7] = {g.n, g.n, g.n, g.heads * g.dim, g.n, g.n, g.inter};
        for (int i = 0; i < 7; ++i) {
            Linear &L = layers[l].lin[i];
            L.name = "model.layers." + std::to_string(l) + "." + linear_names[i];
            L.out = outs[i];
            L.in = ins[i];
            L.wq.reset(size_t(L.out) * L.in);
            L.dscale.reset(1);
            if (trainable) {
                L.offset = off;
                off += size_t(L.out) * L.in;
            }
        }
    }
    weight_params = off;
    for (int l = 0; l < g.layers; ++l) {
        layers[l].norm1 = off, off += g.n;
        layers[l].norm2 = off, off += g.n;
        layers[l].sub_attn = off, off += g.n;
        layers[l].sub_ffn = off, off += g.inter;
    }
    final_norm = off, off += g.n;
    total_params = off;
    norm_params = total_params - weight_params;
    params.reset(total_params);
    if (trainable) {
        grad.reset(total_params);
        m.reset(total_params);
        v.reset(total_params);
        check(cudaMemset(grad.p, 0, grad.bytes()), "memset");
        check(cudaMemset(m.p, 0, m.bytes()), "memset");
        check(cudaMemset(v.p, 0, v.bytes()), "memset");
    }
    embed.reset(size_t(g.vocab) * g.n);
    if (!g.tied)
        head.reset(size_t(g.vocab) * g.n);
    scratch.reset(64);
}
void Model::load_hf(const qtip::SafeTensorsReader &r, cudaStream_t s) {
    std::vector<float> host(total_params, 1.f); // SubLN gains start at one
    for (int l = 0; l < g.layers; ++l) {
        auto &L = layers[l];
        for (int i = 0; i < 7; ++i) {
            auto w = to_float(r, L.lin[i].name + ".weight", size_t(L.lin[i].out) * L.lin[i].in);
            if (trainable)
                std::copy(w.begin(), w.end(), host.begin() + long(L.lin[i].offset));
            else
                upload(L.lin[i].wq, to_bf16_host(w), s);
        }
        const std::string p = "model.layers." + std::to_string(l) + ".";
        auto n1 = to_float(r, p + "input_layernorm.weight", size_t(g.n));
        auto n2 = to_float(r, p + "post_attention_layernorm.weight", size_t(g.n));
        std::copy(n1.begin(), n1.end(), host.begin() + long(L.norm1));
        std::copy(n2.begin(), n2.end(), host.begin() + long(L.norm2));
    }
    auto fn = to_float(r, "model.norm.weight", size_t(g.n));
    std::copy(fn.begin(), fn.end(), host.begin() + long(final_norm));
    upload(params, host, s);
    {
        auto e = to_float(r, "model.embed_tokens.weight", size_t(g.vocab) * g.n);
        upload(embed, to_bf16_host(e), s);
        if (!g.tied) {
            auto h = to_float(r, "lm_head.weight", size_t(g.vocab) * g.n);
            upload(head, to_bf16_host(h), s);
        }
    }
    check(cudaStreamSynchronize(s), "sync");
    if (trainable)
        rebuild_operands(s);
    else
        for (auto &L : layers)
            for (auto &lin : L.lin) {
                lin.scale = 1;
                gpu::fill(lin.dscale.p, 1.f, 1, s);
            }
}
void Model::load_params(const std::vector<float> &host, cudaStream_t s) {
    upload(params, host, s);
    check(cudaStreamSynchronize(s), "sync");
    rebuild_operands(s);
}
std::vector<float> Model::host_params() const { return download(params.p, total_params, cudaStreamPerThread); }
void Model::rebuild_operands(cudaStream_t s) {
    require(trainable, "only a trainable model rebuilds operands");
    for (auto &L : layers)
        for (auto &lin : L.lin) {
            if (quant)
                gpu::ternarize(params.p + lin.offset, size_t(lin.out) * lin.in, lin.wq.p, lin.dscale.p, scratch.p, s);
            else {
                gpu::to_bf16(params.p + lin.offset, lin.wq.p, size_t(lin.out) * lin.in, s);
                gpu::fill(lin.dscale.p, 1.f, 1, s);
            }
        }
    for (auto &L : layers)
        for (auto &lin : L.lin)
            lin.scale = download(lin.dscale.p, 1, s)[0];
}
json Model::layout() const {
    return {{"hidden", g.n},     {"intermediate", g.inter}, {"heads", g.heads},
            {"kv", g.kv},        {"dim", g.dim},            {"vocab", g.vocab},
            {"layers", g.layers}, {"tied", g.tied},          {"weight_params", weight_params},
            {"norm_params", norm_params}, {"quant", quant},   {"subln", subln}, {"trainable", trainable}};
}

void gemm(cublasHandle_t h, bool transA, bool transB, int M, int N, int K, const uint16_t *A, int lda,
          const uint16_t *B, int ldb, float *C, int ldc, float alpha, float beta, int batch, long strideA,
          long strideB, long strideC) {
    // Row-major C[M,N] = op(A) op(B) is column-major C^T[N,M] = op(B)^T op(A)^T.
    auto st = cublasGemmStridedBatchedEx(h, transB ? CUBLAS_OP_T : CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                                         N, M, K, &alpha, B, CUDA_R_16BF, ldb, strideB, A, CUDA_R_16BF, lda,
                                         strideA, &beta, C, CUDA_R_32F, ldc, strideC, batch, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    require(st == CUBLAS_STATUS_SUCCESS, "cublas gemm failed: " + std::to_string(int(st)));
}

Batch upload_batch(const std::vector<qat::Batch> &rows, cudaStream_t s) {
    Batch b;
    std::vector<int32_t> tokens, labels, positions, starts;
    for (const auto &r : rows) {
        const int base = int(tokens.size());
        tokens.insert(tokens.end(), r.tokens.begin(), r.tokens.end());
        labels.insert(labels.end(), r.labels.begin(), r.labels.end());
        positions.insert(positions.end(), r.positions.begin(), r.positions.end());
        for (int st : r.starts)
            starts.push_back(st + base);
    }
    b.rows = int(tokens.size());
    b.tokens.reset(tokens.size()), b.labels.reset(tokens.size()), b.positions.reset(tokens.size()),
        b.starts.reset(tokens.size());
    upload(b.tokens, tokens, s), upload(b.labels, labels, s), upload(b.positions, positions, s),
        upload(b.starts, starts, s);
    b.h_labels = labels;
    b.targets = int(std::count_if(labels.begin(), labels.end(), [](int v) { return v >= 0; }));
    check(cudaStreamSynchronize(s), "sync");
    return b;
}

struct Workspace {
    int T = 0, loss_rows = 0;
    std::vector<Dev<float>> hin; // layer inputs [layers + 1]
    Dev<float> h, hmid, xn, xn2, rstd1, rstd2, rstda, rstdf, xs, xs2, as, ss;
    Dev<uint16_t> xq, xq2, qb, kb, vb, Pb, anq, snq, ab, dyb, dyb2, dSb, dlogits, xnfb, dRb;
    Dev<float> q, k, v, S, P, a, an, o, gate, up, sact, sn, d, xnf, rstdF;
    Dev<float> dh, dxn, dq, dk, dv, da, dan, dgate, dup, ds, dsn, logits, loss, teacher_logits, tq, tk, tv, Rt, Rs,
        dR, dxnf, row_loss;
    Dev<double> norm_acc;
};

Trainer::Trainer(Model &student, Model *teacher, const Options &o, cudaStream_t s)
    : student_(student), teacher_(teacher), opt_(o), s_(s) {
    require(cublasCreate(&blas_) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
    cublasSetStream(blas_, s_);
    cublasSetMathMode(blas_, CUBLAS_DEFAULT_MATH);
}
Trainer::~Trainer() {
    if (blas_)
        cublasDestroy(blas_);
}
void Trainer::ensure(int T) {
    if (ws_ && ws_->T >= T)
        return;
    const auto &g = student_.g;
    ws_ = std::make_unique<Workspace>();
    Workspace &w = *ws_;
    w.T = T;
    w.loss_rows = std::min(opt_.loss_rows, T);
    const size_t Tn = size_t(T) * g.n, Tq = size_t(T) * g.heads * g.dim, Tk = size_t(T) * g.kv * g.dim,
                 Ti = size_t(T) * g.inter, TT = size_t(g.heads) * T * T;
    w.hin.resize(size_t(g.layers) + 1);
    for (auto &b : w.hin)
        b.reset(Tn);
    for (auto *b : {&w.h, &w.hmid, &w.xn, &w.xn2, &w.o, &w.d, &w.xnf, &w.dh, &w.dxn, &w.dxnf})
        b->reset(Tn);
    for (auto *b : {&w.rstd1, &w.rstd2, &w.rstda, &w.rstdf, &w.xs, &w.xs2, &w.as, &w.ss, &w.rstdF, &w.row_loss})
        b->reset(size_t(T));
    for (auto *b : {&w.xq, &w.xq2, &w.xnfb})
        b->reset(Tn);
    for (auto *b : {&w.q, &w.a, &w.an, &w.dq, &w.da, &w.dan})
        b->reset(Tq);
    for (auto *b : {&w.k, &w.v, &w.dk, &w.dv})
        b->reset(Tk);
    w.qb.reset(Tq), w.ab.reset(Tq), w.anq.reset(Tq), w.kb.reset(Tk), w.vb.reset(Tk);
    w.S.reset(TT), w.P.reset(TT), w.Pb.reset(TT), w.dSb.reset(TT);
    for (auto *b : {&w.gate, &w.up, &w.sact, &w.sn, &w.dgate, &w.dup, &w.ds, &w.dsn})
        b->reset(Ti);
    w.snq.reset(Ti);
    const size_t maxout = size_t(T) * std::max({g.n, g.heads * g.dim, g.inter});
    w.dyb.reset(maxout), w.dyb2.reset(maxout);
    w.logits.reset(size_t(w.loss_rows) * g.vocab);
    w.dlogits.reset(size_t(w.loss_rows) * g.vocab);
    w.loss.reset(8);
    w.norm_acc.reset(1);
    if (teacher_) {
        w.teacher_logits.reset(size_t(T) * g.vocab);
        if (opt_.attn_layer >= 0 && opt_.attn_w > 0) {
            w.tq.reset(Tq), w.tk.reset(Tk), w.tv.reset(Tk);
            w.Rt.reset(TT), w.Rs.reset(TT), w.dR.reset(TT), w.dRb.reset(TT);
        }
    }
}
// y = Lin(x): act quant (or bf16 cast), GEMM, scale. xq/xs receive the operand and scales.
static void linear_forward(cublasHandle_t h, const Linear &L, bool quant, const float *x, uint16_t *xq, float *xs,
                           float *y, int T, cudaStream_t s, bool quantize_input = true) {
    if (quantize_input)
        gpu::act_quant(x, xq, xs, T, L.in, quant, s);
    gemm(h, false, true, T, L.out, L.in, xq, L.in, L.wq.p, L.in, y, L.out);
    if (quant)
        gpu::scale_rows(y, xs, L.scale, y, nullptr, T, L.out, s);
}
// dx (+)= dy Weff ; dW += (dy * xs)^T xq
static void linear_backward(cublasHandle_t h, Model &m, const Linear &L, const float *dy, const uint16_t *xq,
                            const float *xs, float *dx, bool accumulate, uint16_t *dyb, uint16_t *dyb2, int T,
                            cudaStream_t s) {
    gpu::to_bf16(dy, dyb, size_t(T) * L.out, s);
    gemm(h, false, false, T, L.in, L.out, dyb, L.out, L.wq.p, L.in, dx, L.in, L.scale, accumulate ? 1.f : 0.f);
    gpu::scale_rows(dy, xs, 1.f, nullptr, dyb2, T, L.out, s);
    gemm(h, true, false, L.out, L.in, T, dyb2, L.out, xq, L.in, m.grad_ptr(L.offset), L.in, 1.f, 1.f);
}
void Trainer::forward_layer(Model &m, int l, const Batch &b, Workspace &w, bool keep_qkv) {
    const auto &g = m.g;
    const int T = b.rows, qd = g.heads * g.dim, kd = g.kv * g.dim, group = g.heads / g.kv;
    const float *in = w.hin[l].p;
    auto &L = m.layers[l];
    auto rnd = [&](float *x, size_t n) {
        if (m.round)
            gpu::round_bf16(x, n, s_);
    };
    // attention
    gpu::rmsnorm_fwd(in, m.params.p + L.norm1, w.xn.p, nullptr, w.rstd1.p, T, g.n, g.eps, s_);
    rnd(w.xn.p, size_t(T) * g.n);
    linear_forward(blas_, L.lin[0], m.quant, w.xn.p, w.xq.p, w.xs.p, w.q.p, T, s_);
    linear_forward(blas_, L.lin[1], m.quant, w.xn.p, w.xq.p, w.xs.p, w.k.p, T, s_, false);
    linear_forward(blas_, L.lin[2], m.quant, w.xn.p, w.xq.p, w.xs.p, w.v.p, T, s_, false);
    rnd(w.q.p, size_t(T) * qd), rnd(w.k.p, size_t(T) * kd), rnd(w.v.p, size_t(T) * kd);
    if (keep_qkv && !opt_.attn_distill_rope) {
        check(cudaMemcpyAsync(w.tq.p, w.q.p, w.tq.bytes(), cudaMemcpyDeviceToDevice, s_), "copy");
        check(cudaMemcpyAsync(w.tk.p, w.k.p, w.tk.bytes(), cudaMemcpyDeviceToDevice, s_), "copy");
        check(cudaMemcpyAsync(w.tv.p, w.v.p, w.tv.bytes(), cudaMemcpyDeviceToDevice, s_), "copy");
    }
    gpu::rope(w.q.p, w.k.p, b.positions.p, T, g.heads, g.kv, g.dim, g.theta, false, s_);
    if (keep_qkv && opt_.attn_distill_rope) {
        check(cudaMemcpyAsync(w.tq.p, w.q.p, w.tq.bytes(), cudaMemcpyDeviceToDevice, s_), "copy");
        check(cudaMemcpyAsync(w.tk.p, w.k.p, w.tk.bytes(), cudaMemcpyDeviceToDevice, s_), "copy");
        check(cudaMemcpyAsync(w.tv.p, w.v.p, w.tv.bytes(), cudaMemcpyDeviceToDevice, s_), "copy");
    }
    gpu::to_bf16(w.q.p, w.qb.p, size_t(T) * qd, s_);
    gpu::to_bf16(w.k.p, w.kb.p, size_t(T) * kd, s_);
    gpu::to_bf16(w.v.p, w.vb.p, size_t(T) * kd, s_);
    const long TT = long(T) * T;
    for (int gi = 0; gi < g.kv; ++gi)
        gemm(blas_, false, true, T, T, g.dim, w.qb.p + size_t(gi) * group * g.dim, qd, w.kb.p + size_t(gi) * g.dim,
             kd, w.S.p + size_t(gi) * group * TT, T, 1.f, 0.f, group, g.dim, 0, TT);
    gpu::softmax_fwd(w.S.p, b.starts.p, 1.f / std::sqrt(float(g.dim)), w.P.p, w.Pb.p, g.heads, T, s_);
    for (int gi = 0; gi < g.kv; ++gi)
        gemm(blas_, false, false, T, g.dim, T, w.Pb.p + size_t(gi) * group * TT, T, w.vb.p + size_t(gi) * g.dim, kd,
             w.a.p + size_t(gi) * group * g.dim, qd, 1.f, 0.f, group, TT, 0, g.dim);
    rnd(w.a.p, size_t(T) * qd);
    const float *o_in = w.a.p;
    if (m.subln) {
        gpu::rmsnorm_fwd(w.a.p, m.params.p + L.sub_attn, w.an.p, nullptr, w.rstda.p, T, qd, g.eps, s_);
        rnd(w.an.p, size_t(T) * qd);
        o_in = w.an.p;
    }
    linear_forward(blas_, L.lin[3], m.quant, o_in, w.anq.p, w.as.p, w.o.p, T, s_);
    rnd(w.o.p, size_t(T) * g.n);
    check(cudaMemcpyAsync(w.hmid.p, in, size_t(T) * g.n * 4, cudaMemcpyDeviceToDevice, s_), "copy");
    gpu::add(w.hmid.p, w.o.p, size_t(T) * g.n, s_);
    rnd(w.hmid.p, size_t(T) * g.n);
    // mlp
    gpu::rmsnorm_fwd(w.hmid.p, m.params.p + L.norm2, w.xn2.p, nullptr, w.rstd2.p, T, g.n, g.eps, s_);
    rnd(w.xn2.p, size_t(T) * g.n);
    linear_forward(blas_, L.lin[4], m.quant, w.xn2.p, w.xq2.p, w.xs2.p, w.gate.p, T, s_);
    linear_forward(blas_, L.lin[5], m.quant, w.xn2.p, w.xq2.p, w.xs2.p, w.up.p, T, s_, false);
    rnd(w.gate.p, size_t(T) * g.inter), rnd(w.up.p, size_t(T) * g.inter);
    gpu::swiglu_fwd(w.gate.p, w.up.p, w.sact.p, size_t(T) * g.inter, s_);
    rnd(w.sact.p, size_t(T) * g.inter);
    const float *d_in = w.sact.p;
    if (m.subln) {
        gpu::rmsnorm_fwd(w.sact.p, m.params.p + L.sub_ffn, w.sn.p, nullptr, w.rstdf.p, T, g.inter, g.eps, s_);
        rnd(w.sn.p, size_t(T) * g.inter);
        d_in = w.sn.p;
    }
    linear_forward(blas_, L.lin[6], m.quant, d_in, w.snq.p, w.ss.p, w.d.p, T, s_);
    rnd(w.d.p, size_t(T) * g.n);
    float *out = w.hin[l + 1].p;
    check(cudaMemcpyAsync(out, w.hmid.p, size_t(T) * g.n * 4, cudaMemcpyDeviceToDevice, s_), "copy");
    gpu::add(out, w.d.p, size_t(T) * g.n, s_);
    rnd(out, size_t(T) * g.n);
}
void Trainer::relation_loss(const Batch &b, Workspace &w, int l) {
    // MiniLM-style self relations softmax(X X^T / sqrt(dim)) for X in {Q, K, V}, teacher vs student.
    const auto &g = student_.g;
    const int T = b.rows;
    const long TT = long(T) * T;
    const float scale = 1.f / std::sqrt(float(g.dim));
    struct Rel {
        const float *t, *s;
        float *ds;
        int nh;
    };
    const Rel rels[3] = {{w.tq.p, w.q.p, w.dq.p, g.heads}, {w.tk.p, w.k.p, w.dk.p, g.kv}, {w.tv.p, w.v.p, w.dv.p, g.kv}};
    (void)l;
    for (const Rel &r : rels) {
        const int ld = r.nh * g.dim;
        const float norm = opt_.attn_w / (float(r.nh) * float(T) * 3.f);
        // teacher relation
        gpu::to_bf16(r.t, w.qb.p, size_t(T) * ld, s_);
        gemm(blas_, false, true, T, T, g.dim, w.qb.p, ld, w.qb.p, ld, w.S.p, T, 1.f, 0.f, r.nh, g.dim, g.dim, TT);
        gpu::softmax_fwd(w.S.p, b.starts.p, scale, w.Rt.p, nullptr, r.nh, T, s_);
        // student relation (qb/ab hold T * heads * dim elements, enough for every relation)
        gpu::to_bf16(r.s, w.ab.p, size_t(T) * ld, s_);
        gemm(blas_, false, true, T, T, g.dim, w.ab.p, ld, w.ab.p, ld, w.S.p, T, 1.f, 0.f, r.nh, g.dim, g.dim, TT);
        gpu::softmax_fwd(w.S.p, b.starts.p, scale, w.Rs.p, nullptr, r.nh, T, s_);
        gpu::relation_kl(w.Rt.p, w.Rs.p, b.starts.p, norm, w.loss.p + 3, w.dR.p, r.nh, T, s_);
        gpu::to_bf16(w.dR.p, w.dRb.p, size_t(r.nh) * TT, s_);
        // dX += scale (dR + dR^T) X
        gemm(blas_, false, false, T, g.dim, T, w.dRb.p, T, w.ab.p, ld, r.ds, ld, scale, 1.f, r.nh, TT, g.dim, g.dim);
        gemm(blas_, true, false, T, g.dim, T, w.dRb.p, T, w.ab.p, ld, r.ds, ld, scale, 1.f, r.nh, TT, g.dim, g.dim);
    }
}
void Trainer::backward_layer(Model &m, int l, const Batch &b, Workspace &w) {
    const auto &g = m.g;
    const int T = b.rows, qd = g.heads * g.dim, kd = g.kv * g.dim, group = g.heads / g.kv;
    const long TT = long(T) * T;
    auto &L = m.layers[l];
    forward_layer(m, l, b, w, false);
    // down
    linear_backward(blas_, m, L.lin[6], w.dh.p, w.snq.p, w.ss.p, w.dsn.p, false, w.dyb.p, w.dyb2.p, T, s_);
    float *ds = w.dsn.p;
    if (m.subln) {
        gpu::rmsnorm_bwd(w.sact.p, m.params.p + L.sub_ffn, w.rstdf.p, w.dsn.p, w.ds.p, m.grad_ptr(L.sub_ffn), T,
                         g.inter, false, s_);
        ds = w.ds.p;
    }
    gpu::swiglu_bwd(w.gate.p, w.up.p, ds, w.dgate.p, w.dup.p, size_t(T) * g.inter, s_);
    linear_backward(blas_, m, L.lin[4], w.dgate.p, w.xq2.p, w.xs2.p, w.dxn.p, false, w.dyb.p, w.dyb2.p, T, s_);
    linear_backward(blas_, m, L.lin[5], w.dup.p, w.xq2.p, w.xs2.p, w.dxn.p, true, w.dyb.p, w.dyb2.p, T, s_);
    gpu::rmsnorm_bwd(w.hmid.p, m.params.p + L.norm2, w.rstd2.p, w.dxn.p, w.dh.p, m.grad_ptr(L.norm2), T, g.n, true,
                     s_);
    // o_proj
    linear_backward(blas_, m, L.lin[3], w.dh.p, w.anq.p, w.as.p, w.dan.p, false, w.dyb.p, w.dyb2.p, T, s_);
    float *da = w.dan.p;
    if (m.subln) {
        gpu::rmsnorm_bwd(w.a.p, m.params.p + L.sub_attn, w.rstda.p, w.dan.p, w.da.p, m.grad_ptr(L.sub_attn), T, qd,
                         false, s_);
        da = w.da.p;
    }
    // attention backward: dP = dA V^T, dV = P^T dA, dS = softmax', dQ = dS K, dK = dS^T Q
    gpu::to_bf16(da, w.ab.p, size_t(T) * qd, s_);
    for (int gi = 0; gi < g.kv; ++gi) {
        gemm(blas_, false, true, T, T, g.dim, w.ab.p + size_t(gi) * group * g.dim, qd, w.vb.p + size_t(gi) * g.dim,
             kd, w.S.p + size_t(gi) * group * TT, T, 1.f, 0.f, group, g.dim, 0, TT);
        for (int hh = 0; hh < group; ++hh) {
            const int hd = gi * group + hh;
            gemm(blas_, true, false, T, g.dim, T, w.Pb.p + size_t(hd) * TT, T, w.ab.p + size_t(hd) * g.dim, qd,
                 w.dv.p + size_t(gi) * g.dim, kd, 1.f, hh ? 1.f : 0.f);
        }
    }
    gpu::softmax_bwd(w.P.p, w.S.p, 1.f / std::sqrt(float(g.dim)), w.dSb.p, nullptr, g.heads, T, s_);
    for (int gi = 0; gi < g.kv; ++gi) {
        gemm(blas_, false, false, T, g.dim, T, w.dSb.p + size_t(gi) * group * TT, T, w.kb.p + size_t(gi) * g.dim,
             kd, w.dq.p + size_t(gi) * group * g.dim, qd, 1.f, 0.f, group, TT, 0, g.dim);
        for (int hh = 0; hh < group; ++hh) {
            const int hd = gi * group + hh;
            gemm(blas_, true, false, T, g.dim, T, w.dSb.p + size_t(hd) * TT, T, w.qb.p + size_t(hd) * g.dim, qd,
                 w.dk.p + size_t(gi) * g.dim, kd, 1.f, hh ? 1.f : 0.f);
        }
    }
    const bool distill_here = teacher_ && opt_.attn_layer == l && opt_.attn_w > 0;
    if (distill_here && opt_.attn_distill_rope)
        relation_loss(b, w, l);
    gpu::rope(w.dq.p, w.dk.p, b.positions.p, T, g.heads, g.kv, g.dim, g.theta, true, s_);
    if (distill_here && !opt_.attn_distill_rope) {
        // student pre-RoPE projections: undo the rotation on q/k (v is unrotated)
        gpu::rope(w.q.p, w.k.p, b.positions.p, T, g.heads, g.kv, g.dim, g.theta, true, s_);
        relation_loss(b, w, l);
    }
    linear_backward(blas_, m, L.lin[0], w.dq.p, w.xq.p, w.xs.p, w.dxn.p, false, w.dyb.p, w.dyb2.p, T, s_);
    linear_backward(blas_, m, L.lin[1], w.dk.p, w.xq.p, w.xs.p, w.dxn.p, true, w.dyb.p, w.dyb2.p, T, s_);
    linear_backward(blas_, m, L.lin[2], w.dv.p, w.xq.p, w.xs.p, w.dxn.p, true, w.dyb.p, w.dyb2.p, T, s_);
    gpu::rmsnorm_bwd(w.hin[l].p, m.params.p + L.norm1, w.rstd1.p, w.dxn.p, w.dh.p, m.grad_ptr(L.norm1), T, g.n,
                     true, s_);
}
// Final norm + head + loss (chunked over rows). Teacher logits (fp32 [T, vocab]) optional.
// With backward: dxnf receives the gradient wrt the normed final activations and dh wrt the final residual.
double Trainer::logits_loss(Model &m, const Batch &b, Workspace &w, bool backward, const float *teacher,
                            LossReport &rep, std::vector<float> *row_loss, std::vector<float> *logits_out) {
    const auto &g = m.g;
    const int T = b.rows;
    const float *hf = w.hin[g.layers].p;
    gpu::rmsnorm_fwd(hf, m.params.p + m.final_norm, w.xnf.p, w.xnfb.p, w.rstdF.p, T, g.n, g.eps, s_);
    const uint16_t *head = g.tied ? m.embed.p : m.head.p;
    check(cudaMemsetAsync(w.loss.p, 0, 4 * sizeof(float), s_), "memset");
    const float inv = b.targets ? 1.f / float(b.targets) : 0.f;
    const bool kl = teacher && opt_.kl_w > 0;
    if (logits_out)
        logits_out->resize(size_t(T) * g.vocab);
    for (int r0 = 0; r0 < T; r0 += w.loss_rows) {
        const int rows = std::min(w.loss_rows, T - r0);
        gemm(blas_, false, true, rows, g.vocab, g.n, w.xnfb.p + size_t(r0) * g.n, g.n, head, g.n, w.logits.p,
             g.vocab);
        gpu::lm_loss(w.logits.p, b.labels.p + r0, kl ? teacher + size_t(r0) * g.vocab : nullptr, rows, g.vocab,
                     opt_.temperature, opt_.ce_w, kl ? opt_.kl_w : 0.f, inv, inv, w.loss.p,
                     backward ? w.dlogits.p : nullptr, row_loss ? w.row_loss.p + r0 : nullptr, s_);
        if (logits_out)
            check(cudaMemcpyAsync(logits_out->data() + size_t(r0) * g.vocab, w.logits.p,
                                  size_t(rows) * g.vocab * 4, cudaMemcpyDeviceToHost, s_),
                  "download logits");
        if (backward)
            gemm(blas_, false, false, rows, g.n, g.vocab, w.dlogits.p, g.vocab, head, g.n,
                 w.dxnf.p + size_t(r0) * g.n, g.n);
    }
    if (backward)
        gpu::rmsnorm_bwd(hf, m.params.p + m.final_norm, w.rstdF.p, w.dxnf.p, w.dh.p, m.grad_ptr(m.final_norm), T,
                         g.n, false, s_);
    auto l = download(w.loss.p, 4, s_);
    rep.ce += l[0], rep.kl += l[1], rep.count += l[2];
    if (row_loss)
        *row_loss = download(w.row_loss.p, size_t(T), s_);
    return l[0];
}
void Trainer::teacher_forward(const Batch &b, Workspace &w) {
    Model &t = *teacher_;
    const auto &g = t.g;
    const int T = b.rows;
    gpu::embedding(t.embed.p, b.tokens.p, w.hin[0].p, T, g.n, s_);
    for (int l = 0; l < g.layers; ++l)
        forward_layer(t, l, b, w, opt_.attn_layer == l && opt_.attn_w > 0);
    const float *hf = w.hin[g.layers].p;
    gpu::rmsnorm_fwd(hf, t.params.p + t.final_norm, w.xnf.p, w.xnfb.p, w.rstdF.p, T, g.n, g.eps, s_);
    const uint16_t *head = g.tied ? t.embed.p : t.head.p;
    gemm(blas_, false, true, T, g.vocab, g.n, w.xnfb.p, g.n, head, g.n, w.teacher_logits.p, g.vocab);
}
LossReport Trainer::step(const Batch &b) {
    ensure(b.rows);
    Workspace &w = *ws_;
    LossReport rep;
    Model &m = student_;
    const auto &g = m.g;
    const int T = b.rows;
    if (teacher_)
        teacher_forward(b, w);
    gpu::embedding(m.embed.p, b.tokens.p, w.hin[0].p, T, g.n, s_);
    for (int l = 0; l < g.layers; ++l)
        forward_layer(m, l, b, w, false);
    logits_loss(m, b, w, true, teacher_ ? w.teacher_logits.p : nullptr, rep, nullptr, nullptr);
    for (int l = g.layers - 1; l >= 0; --l)
        backward_layer(m, l, b, w);
    if (teacher_ && opt_.attn_layer >= 0 && opt_.attn_w > 0)
        rep.attn += double(download(w.loss.p + 3, 1, s_)[0]);
    check(cudaGetLastError(), "step kernels");
    return rep;
}
LossReport Trainer::evaluate(Model &m, const Batch &b, std::vector<float> *row_loss, std::vector<float> *logits) {
    ensure(b.rows);
    Workspace &w = *ws_;
    LossReport rep;
    const auto &g = m.g;
    gpu::embedding(m.embed.p, b.tokens.p, w.hin[0].p, b.rows, g.n, s_);
    for (int l = 0; l < g.layers; ++l)
        forward_layer(m, l, b, w, false);
    logits_loss(m, b, w, false, nullptr, rep, row_loss, logits);
    check(cudaGetLastError(), "evaluate kernels");
    return rep;
}
std::vector<float> Trainer::gradient() const { return download(student_.grad.p, student_.total_params, s_); }
double Trainer::grad_norm() {
    Workspace &w = *ws_;
    check(cudaMemsetAsync(w.norm_acc.p, 0, sizeof(double), s_), "memset");
    gpu::sumsq(student_.grad.p, student_.total_params, w.norm_acc.p, s_);
    return std::sqrt(download(w.norm_acc.p, 1, s_)[0]);
}
void Trainer::optimizer_step(float lr, float norm_lr, float gscale, int step) {
    Model &m = student_;
    gpu::adamw(m.params.p, m.grad.p, m.m.p, m.v.p, m.weight_params, lr, opt_.beta1, opt_.beta2, opt_.eps,
               opt_.decay, step, gscale, s_);
    gpu::adamw(m.params.p + m.weight_params, m.grad.p + m.weight_params, m.m.p + m.weight_params,
               m.v.p + m.weight_params, m.norm_params, norm_lr, opt_.beta1, opt_.beta2, opt_.eps, 0.f, step, gscale,
               s_);
    m.rebuild_operands(s_);
}
void Trainer::zero_grad() { check(cudaMemsetAsync(student_.grad.p, 0, student_.grad.bytes(), s_), "memset"); }

// ---- checkpoints and export ----
void save_checkpoint(const fs::path &dir, const Model &m, const State &st, const Options &o, const json &extra) {
    fs::create_directories(dir);
    qtip::SafeTensorsWriter w;
    auto p = download(m.params.p, m.total_params, cudaStreamPerThread);
    w.add("params", "F32", {int64_t(m.total_params)}, p.data(), p.size() * 4);
    auto mm = download(m.m.p, m.total_params, cudaStreamPerThread);
    w.add("adam_m", "F32", {int64_t(m.total_params)}, mm.data(), mm.size() * 4);
    auto vv = download(m.v.p, m.total_params, cudaStreamPerThread);
    w.add("adam_v", "F32", {int64_t(m.total_params)}, vv.data(), vv.size() * 4);
    w.write((dir / "state.safetensors").string());
    json meta = {{"format", "bitnet-trainer-v1"},
                 {"step", st.step},
                 {"tokens", st.tokens},
                 {"position", st.position},
                 {"layout", m.layout()},
                 {"options", {{"lr", o.lr}, {"norm_lr", o.norm_lr}, {"temperature", o.temperature},
                              {"ce_w", o.ce_w}, {"kl_w", o.kl_w}, {"attn_w", o.attn_w},
                              {"attn_layer", o.attn_layer}, {"rows", o.rows}, {"accum", o.accum}}},
                 {"extra", extra}};
    qat::durable_json(dir / "metadata.json", meta);
}
State load_checkpoint(const fs::path &dir, Model &m, cudaStream_t s) {
    auto meta = qat::read_json(dir / "metadata.json");
    require(meta.at("format") == "bitnet-trainer-v1", "unsupported checkpoint format");
    require(meta.at("layout") == m.layout(), "checkpoint layout mismatch");
    qtip::SafeTensorsReader r(dir.string(), "state.safetensors");
    auto p = to_float(r, "params", m.total_params);
    upload(m.params, p, s);
    if (m.trainable) {
        upload(m.m, to_float(r, "adam_m", m.total_params), s);
        upload(m.v, to_float(r, "adam_v", m.total_params), s);
    }
    check(cudaStreamSynchronize(s), "sync");
    if (m.trainable)
        m.rebuild_operands(s);
    State st;
    st.step = meta.at("step");
    st.tokens = meta.at("tokens");
    st.position = meta.at("position");
    return st;
}
void export_model(const fs::path &dest, const fs::path &source, const Model &m, const json &info) {
    require(!fs::exists(dest), "export destination exists: " + dest.string());
    require(m.quant, "export requires a ternary model");
    const auto &g = m.g;
    fs::create_directories(dest);
    qtip::SafeTensorsWriter w;
    auto params = download(m.params.p, m.total_params, cudaStreamPerThread);
    // Norm gains are trained in FP32 and exported unchanged (F32) so that the runtime, which reads
    // normalisation vectors as FP32, reproduces the trainer's forward exactly.
    auto add_norm = [&](const std::string &name, size_t off, int n) {
        w.add(name, "F32", {n}, params.data() + off, size_t(n) * 4);
    };
    for (int l = 0; l < g.layers; ++l) {
        const auto &L = m.layers[l];
        const std::string p = "model.layers." + std::to_string(l) + ".";
        for (int i = 0; i < 7; ++i) {
            const Linear &lin = L.lin[i];
            auto wq = download(lin.wq.p, size_t(lin.out) * lin.in, cudaStreamPerThread);
            std::vector<int8_t> q(wq.size());
            for (size_t j = 0; j < wq.size(); ++j) {
                const float v = std::bit_cast<float>(uint32_t(wq[j]) << 16);
                require(v == -1.f || v == 0.f || v == 1.f, "non-ternary operand");
                q[j] = int8_t(v);
            }
            std::vector<uint8_t> packed(size_t(lin.out) * behemoth::ternary::packed_bytes(size_t(lin.in)));
            behemoth::ternary::pack(q, size_t(lin.out), size_t(lin.in), packed);
            w.add(lin.name + ".weight", "U8", {lin.out, lin.in / 4}, packed.data(), packed.size());
            const float sc = lin.scale;
            w.add(lin.name + ".weight_scale", "F32", {1}, &sc, 4);
        }
        add_norm(p + "input_layernorm.weight", L.norm1, g.n);
        add_norm(p + "post_attention_layernorm.weight", L.norm2, g.n);
        if (m.subln) {
            add_norm(p + "self_attn.attn_sub_norm.weight", L.sub_attn, g.heads * g.dim);
            add_norm(p + "mlp.ffn_sub_norm.weight", L.sub_ffn, g.inter);
        }
    }
    add_norm("model.norm.weight", m.final_norm, g.n);
    auto e = download(m.embed.p, size_t(g.vocab) * g.n, cudaStreamPerThread);
    w.add("model.embed_tokens.weight", "BF16", {g.vocab, g.n}, e.data(), e.size() * 2);
    if (!g.tied) {
        auto h = download(m.head.p, size_t(g.vocab) * g.n, cudaStreamPerThread);
        w.add("lm_head.weight", "BF16", {g.vocab, g.n}, h.data(), h.size() * 2);
    }
    w.write((dest / "model.safetensors").string());
    json cfg = qat::read_json(source / "config.json");
    cfg["quantization_config"] = {{"quant_method", "bitnet"},
                                  {"bits", 1.58},
                                  {"weight_scale", "absmean_per_tensor"},
                                  {"activation_bits", 8},
                                  {"activation_scale", "absmax_per_token"},
                                  {"packing", "u32x16_2bit_byte_interleaved"},
                                  {"sub_norm", m.subln}};
    cfg["bitnet_params"] = info;
    {
        std::ofstream f(dest / "config.json");
        f << cfg.dump(2) << "\n";
    }
    for (const char *name : {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                             "generation_config.json", "merges.txt", "vocab.json", "chat_template.jinja"})
        if (fs::exists(source / name))
            fs::copy_file(source / name, dest / name);
}
} // namespace bitnet
