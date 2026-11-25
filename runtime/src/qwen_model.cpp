// Qwen3.5 single-GPU QTIP runtime model. Loads the qtip_quant output (trellis linears + bf16 small
// tensors), runs hybrid Gated DeltaNet / gated full attention layers with a bf16 KV cache and
// per-layer fp32 delta state. Correctness-first: every projection goes through the trellis GEMV/WMMA
// kernels (rotate -> matmul -> slice sum -> hadU -> SV), the lm_head is a bf16 GEMM on the embedding.
#include "behemoth/qwen_model.hpp"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "behemoth/checkpoint.hpp"
#include "behemoth/qtip.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/qwen_kernels.hpp"
#include "behemoth/qwen_batch.hpp"

namespace behemoth {
// Minimal safetensors-directory reader for the qwen path (behemoth::Checkpoint validates the
// Mistral config and cannot load qwen3_5 checkpoints).
using json = nlohmann::json;
struct QwenCheckpoint {
    std::filesystem::path root;
    std::map<std::string, std::string> index;
    mutable std::map<std::string, std::shared_ptr<SafeTensorFile>> files;

    explicit QwenCheckpoint(std::filesystem::path p) : root(std::move(p)) {
        std::ifstream f(root / "model.safetensors.index.json");
        if (!f) throw std::runtime_error("no model.safetensors.index.json in " + root.string());
        json idx = json::parse(f);
        for (auto it = idx["weight_map"].begin(); it != idx["weight_map"].end(); ++it)
            index[it.key()] = it.value();
    }
    const SafeTensorFile &tensor(const std::string &name) const {
        auto it = index.find(name);
        if (it == index.end()) throw std::runtime_error("tensor not found: " + name);
        auto &ptr = files[it->second];
        if (!ptr) ptr = std::make_shared<SafeTensorFile>(root / it->second);
        return *ptr;
    }
    std::vector<BFloat16> bf16(const std::string &name) const {
        auto v = tensor(name).floats(name);
        std::vector<BFloat16> r(v.size());
        for (size_t i = 0; i < v.size(); ++i) r[i] = BFloat16(v[i]);
        return r;
    }
};
}  // namespace behemoth

namespace behemoth {
namespace {

json read_json(const std::filesystem::path &p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return json::parse(f);
}

std::vector<uint16_t> bf16_raw(const QwenCheckpoint &ck, const std::string &name) {
    auto v = ck.bf16(name);
    std::vector<uint16_t> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        float x = float(v[i]);
        uint32_t u;
        std::memcpy(&u, &x, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        r[i] = uint16_t(u >> 16);
    }
    return r;
}
// Row ranges x column window of a row-major [*][ncols] tensor of esz-byte elements (tensor parallel
// weight shards). rows empty -> all rows (nrows); nc < 0 -> all columns.
struct Slice {
    std::vector<std::pair<int, int>> rows;  // (first, count)
    int c0 = 0, nc = -1;
    int nrows(int full) const {
        if (rows.empty()) return full;
        int n = 0;
        for (auto &r : rows) n += r.second;
        return n;
    }
    int ncols(int full) const { return nc < 0 ? full : nc; }
    bool whole() const { return rows.empty() && nc < 0; }
    Slice blocks(int b) const {  // the same slice in units of b x b blocks (fp8 scales)
        Slice o;
        auto chk = [&](int v) {
            if (v % b) throw std::runtime_error("tensor-parallel slice not aligned to fp8 scale blocks");
            return v / b;
        };
        for (auto &r : rows) o.rows.push_back({chk(r.first), chk(r.second)});
        o.c0 = nc < 0 ? 0 : chk(c0);
        o.nc = nc < 0 ? -1 : chk(nc);
        return o;
    }
};
template <class T>
std::vector<T> slice2d(const T *src, int nrows, int ncols, const Slice &sl) {
    const int oc = sl.ncols(ncols);
    std::vector<std::pair<int, int>> rr = sl.rows.empty() ? std::vector<std::pair<int, int>>{{0, nrows}} : sl.rows;
    std::vector<T> out;
    out.reserve(size_t(sl.nrows(nrows)) * oc);
    for (auto &r : rr) {
        if (r.first < 0 || r.first + r.second > nrows || sl.c0 + oc > ncols)
            throw std::runtime_error("tensor-parallel slice out of range");
        for (int i = r.first; i < r.first + r.second; ++i)
            out.insert(out.end(), src + size_t(i) * ncols + sl.c0, src + size_t(i) * ncols + sl.c0 + oc);
    }
    return out;
}
}  // namespace

QwenRuntimeConfig QwenRuntimeConfig::read(const std::filesystem::path &dir) {
    json j = read_json(dir / "config.json");
    const json &t = j.contains("text_config") ? j["text_config"] : j;
    std::string mt = j.value("model_type", std::string(""));
    if (mt != "qwen3_5" && mt != "qwen3_5_moe" && t.value("model_type", "") != "qwen3_5_text")
        throw std::runtime_error("not a qwen3_5 checkpoint: " + dir.string());
    QwenRuntimeConfig c;
    c.hidden = t.at("hidden_size");
    c.layers = t.at("num_hidden_layers");
    c.vocab = t.at("vocab_size");
    c.eps = t.value("rms_norm_eps", 1e-6f);
    c.nq = t.at("num_attention_heads");
    c.nkv = t.at("num_key_value_heads");
    c.hd = t.value("head_dim", c.hidden / c.nq);
    c.attn_output_gate = t.value("attn_output_gate", true);
    c.lin_nk = t.at("linear_num_key_heads");
    c.lin_kd = t.at("linear_key_head_dim");
    c.lin_nv = t.at("linear_num_value_heads");
    c.lin_vd = t.at("linear_value_head_dim");
    c.conv_k = t.value("linear_conv_kernel_dim", 4);
    c.tied = t.value("tie_word_embeddings", false);
    float partial = 0.25f, theta = 1e7f;
    if (t.contains("rope_parameters")) {
        partial = t["rope_parameters"].value("partial_rotary_factor", partial);
        theta = t["rope_parameters"].value("rope_theta", theta);
    }
    partial = t.value("partial_rotary_factor", partial);
    theta = t.value("rope_theta", theta);
    c.theta = theta;
    c.rot_dim = int(float(c.hd) * partial) & ~1;
    c.num_experts = t.value("num_experts", t.value("n_routed_experts", 0));
    c.inter = t.value("intermediate_size", t.value("moe_intermediate_size", 0));
    if (c.num_experts > 0) {
        c.topk = t.value("num_experts_per_tok", t.value("n_experts_per_tok", 8));
        c.moe_inter = t.value("moe_intermediate_size", c.inter);
        c.inter = c.moe_inter;
        c.shared_inter = t.value("shared_expert_intermediate_size", c.moe_inter);
        c.shared_gate = t.value("shared_expert_gated", true);
        std::string scoring = t.value("scoring_func", "softmax");
        if (scoring != "softmax")
            throw std::runtime_error("only softmax router scoring supported (got " + scoring + ")");
    }
    c.full_attn.assign(size_t(c.layers), 0);
    if (t.contains("layer_types") && !t["layer_types"].is_null()) {
        for (int i = 0; i < c.layers; ++i)
            c.full_attn[size_t(i)] = (t["layer_types"][size_t(i)] == "full_attention") ? 1 : 0;
    } else {
        int interval = t.value("full_attention_interval", 4);
        for (int i = 0; i < c.layers; ++i) c.full_attn[size_t(i)] = ((i + 1) % interval == 0) ? 1 : 0;
    }
    return c;
}

// ------------------------------------------------------------------ impl
struct QwenStage {
    // One linear: either a QTIP trellis (rotate -> matmul -> slice sum -> hadU -> SV epilogue) or a
    // plain bf16 [out][in] weight (unquantized checkpoints; the bf16-vs-quantized comparison then
    // runs the identical surrounding code).
    struct Linear {
        qtip::GpuLinear lin;
        DeviceBuffer<float> sv;  // [out]
        bool bf16 = false;
        int m = 0, n = 0;               // bf16 dims
        DeviceBuffer<uint16_t> w;       // bf16 weights [m][n]
        DeviceBuffer<float> su;         // trellis: input SU (per-Hessian-group shared)
        bool fp8 = false;               // E4M3 [m][n] + 128x128 block scales (bf16 flag also set)
        DeviceBuffer<uint8_t> w8;
        DeviceBuffer<float> ws;
        int rows() const { return bf16 ? m : lin.rows; }
        int in_dim() const { return bf16 ? n : lin.in; }
    };
    struct LayerW {
        bool full = false;
        bool moe = false;
        Linear lq, lk, lv, lo, lgate, lup, ldown;   // full attention + dense mlp
        Linear l_in_qkv, l_in_z, l_out;                 // linear attention (b/a stay bf16)
        // fp8 checkpoints: input-sharing linears fused row-wise (one W8A8 GEMM): q|k|v, qkv|z, gate|up
        Linear f_qkv, f_qkvz, f_gu;
        bool fused = false;
        std::vector<Linear> e_gate, e_up, e_down;       // MoE routed experts
        Linear sh_gate, sh_up, sh_down;                 // MoE shared expert
        DeviceBuffer<uint16_t> w_router;                // MoE router bf16 [E][H]
        DeviceBuffer<uint16_t> w_sg;                    // shared_expert_gate bf16 [H]
        DeviceBuffer<uint16_t> w_ba;                     // [2*nv][hidden] bf16 in_proj_b|a
        DeviceBuffer<uint16_t> ln_in, ln_post;      // raw zc weights (kernels add 1)
        DeviceBuffer<uint16_t> q_norm, k_norm;      // full attention (raw zc)
        DeviceBuffer<uint16_t> conv_w, dn_norm;     // linear attention
        DeviceBuffer<float> a_dt;                   // [2nv] = A_log | dt_bias
        DeviceBuffer<float> ring, dstate;           // per-layer delta state (f32)
        DeviceBuffer<uint16_t> kc, vc;              // KV cache [maxctx][nkv][hd] bf16
    };
    QwenRuntimeConfig cfg;
    int device = 0, chunk = 256;
    // pipeline stage: layers [l_begin, l_end); first stage embeds, last stage owns norm + lm_head
    int l_begin = 0, l_end = 0;
    bool first = true, last = true;
    QwenStage *prev = nullptr, *next = nullptr;
    // done_ev: this stage's chunk finished (next stage may read `state`); took_ev: this stage has
    // copied the previous stage's state (previous stage may overwrite it). No host syncs.
    hipEvent_t done_ev{}, took_ev{};
    bool p2p = false;            // this device can read the peer / previous stage's memory
    int *tok_pinned = nullptr;   // pinned host staging for decode tokens
    DeviceBuffer<uint16_t> wtmp;  // fp8 prefill (legacy path): dequantised weight of the current linear
    DeviceBuffer<uint8_t> xq8;    // W8A8: fp8 activations [chunk][max in] + per-128 scales
    DeviceBuffer<float> xqs, gpart;
    Linear lm8;                   // fp8 lm_head (block-quantised at load) unless QWEN_LMHEAD_BF16
    // device-flag all-reduce (q_ar_push / q_ar_pull_add)
    char *ar_host = nullptr;      // [AR_SLOTS][ar_slot_bytes] pinned
    unsigned *ar_flags = nullptr; // [AR_SLOTS][16] pinned
    DeviceBuffer<unsigned> ar_ctr;  // [0] seq, [1] push ctr, [2] pull ctr
    long ar_slot_bytes = 0;
    unsigned ar_hseq = 0;         // host mirror of the device exchange counter (slot of DMA pulls)
    static constexpr int AR_SLOTS = 4;
    hipStream_t stream{};
    qtip::DenseMatmul dense;
    qtip::GpuCodebook codebook;
    std::vector<LayerW> layers;
    DeviceBuffer<uint16_t> embed, final_norm, lm_head_w;  // lm_head_w empty -> tied (use embed)
    // scratch (sized for chunk rows)
    DeviceBuffer<uint16_t> state, h_norm, u, q, k, v, gate, attn_out, o_in, gu, eX;
    DeviceBuffer<float> partial, z, z2, logits_d;
    DeviceBuffer<float> attn_part;  // split-K attention partials
    DeviceBuffer<float> ones_f;  // all-1 "SV" for every bf16 linear epilogue (widest output, incl. q_proj)
    // MoE scratch
    DeviceBuffer<float> rlog, rwts, y_moe, sg_logit, slot_w;
    DeviceBuffer<int> rids, rcnt, rbase, rrun, slot_tok;
    DeviceBuffer<int> tokens_d;
    std::vector<float> logits_h, logits_all;
    size_t max_slices = 0;
    // "model.layers." (quantized checkpoints written by qtip_quant) or "model.language_model.layers."
    // (original HF Qwen3.5 checkpoints, loaded through the bf16-linear fallback)
    std::string layer_prefix = "model.layers.";

    void load(const std::filesystem::path &dir);
    qtip::HostLinear load_host(const QwenCheckpoint &ck, int layer, const std::string &name, int in, int out);
    void upload_linear(Linear &L, const qtip::HostLinear &h);
    // dst = row-wise concatenation of fp8 linears sharing the input (parts released)
    static void fuse_fp8(Linear &dst, std::initializer_list<Linear *> parts) {
        size_t wb = 0, sb = 0;
        dst.bf16 = dst.fp8 = true;
        dst.m = 0;
        dst.n = (*parts.begin())->n;
        for (Linear *p : parts) {
            if (!p->fp8 || p->n != dst.n || p->m % 128) throw std::runtime_error("fuse_fp8: incompatible parts");
            wb += p->w8.size();
            sb += p->ws.size();
            dst.m += p->m;
        }
        dst.w8.resize(wb);
        dst.ws.resize(sb);
        size_t ow = 0, os = 0;
        for (Linear *p : parts) {
            HIP_CHECK(hipMemcpy(dst.w8.data() + ow, p->w8.data(), p->w8.bytes(), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(dst.ws.data() + os, p->ws.data(), p->ws.bytes(), hipMemcpyDeviceToDevice));
            ow += p->w8.size();
            os += p->ws.size();
            *p = Linear{};
        }
    }
    void load_linear(Linear &L, const QwenCheckpoint &ck, int layer, const std::string &name, int in,
                     int out, const Slice &sl = Slice{});  // trellis, fp8 or bf16; in/out: full dims
    // tensor parallel: rank tp_rank of tp_size holds a head / intermediate / vocab shard of every
    // layer (cfg = local dims, gcfg = checkpoint dims); row-parallel outputs are summed over ranks
    int tp_rank = 0, tp_size = 1;
    QwenRuntimeConfig gcfg;
    QwenStage *tp_peer = nullptr;
    std::barrier<> *tp_bar = nullptr;
    DeviceBuffer<float> zsend[2], zrecv;
    hipEvent_t send_ev[2]{}, pulled_ev[2]{};
    unsigned ar_round = 0;
    int voff = 0;  // first vocab row of this rank's lm_head shard
    void residual_reduce(const float *zpart, int rows, bool decode = false);  // state += sum over ranks of zpart
    // pinned host slots of the all-reduce, allocated while the PEER device is current: the runtime
    // places them on the peer's NUMA node, so the latency-bound zero-copy reads stay socket-local
    void alloc_ar_host() {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&ar_host), size_t(ar_slot_bytes) * AR_SLOTS,
                                hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&ar_flags), AR_SLOTS * 16 * sizeof(unsigned),
                                hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
        std::memset(ar_flags, 0, AR_SLOTS * 16 * sizeof(unsigned));
    }
    // output buffer for a row-parallel linear whose result goes to residual_reduce (TP: the round's
    // send buffer, after the peer finished reading it two rounds ago)
    float *rr_buf(float *fallback, int rows) {
        if (tp_size == 1) return fallback;
        (void)rows;
        return zsend[0].data();
    }
    float *hsend[2]{};  // no P2P: pinned coherent host buffers both ranks' kernels read directly
    void lm_head_enqueue(int row0, int nrows);
    void run_linear(const Linear &L, const uint16_t *x, int ldx, int rows, float *out, int ldy = 0);
    // W8A8 linear on activations already quantised into xq8 / xqs (q_norm_quant)
    void run_linear_q(const Linear &L, int rows, float *y, int ldy = 0) {
        q_w8a8(L.w8.data(), L.ws.data(), L.m, L.n, xq8.data(), xqs.data(), rows, y, ldy ? ldy : L.m, gpart.data(), stream);
    }
    void forward_chunk(const int *tokens_h, int rows, int slot, int pos0);
    // multi-sequence prefill: rows [row0, row0 + n) belong to (slot, positions pos0..). Empty: one
    // segment (slot, pos0) of all rows. Linears / all-reduce run over all rows at once.
    struct Seg {
        int slot, pos0, row0, n;
    };
    std::vector<Seg> segs;
    bool prefill_big = false;  // serving prefill: always the 128x128-tile GEMM + bf16 all-reduce (row-count invariant)
    // ---- sequence pools: KV in 64-token pages (pool rows [n_pages * 64]), DeltaNet ring / state per slot
    static constexpr int PAGE = 64;
    int n_slots = 1, n_pages = 0, maxpages = 0;
    DeviceBuffer<int> ptab_d;  // [n_slots][maxpages]
    size_t slot_bytes() const {  // DeltaNet state per slot (all layers of this stage)
        size_t b = 0;
        for (int L = l_begin; L < l_end; ++L)
            if (!cfg.is_full(L))
                b += (size_t(cfg.conv_dim()) * cfg.conv_k + size_t(cfg.lin_nv) * cfg.lin_kd * cfg.lin_vd) * 4;
        return b;
    }
    size_t page_bytes() const {
        size_t b = 0;
        for (int L = l_begin; L < l_end; ++L)
            if (cfg.is_full(L)) b += size_t(PAGE) * cfg.nkv * cfg.hd * 2 * 2;
        return b;
    }
    void alloc_pools(int slots, int pages, int mp) {
        n_slots = slots;
        n_pages = pages;
        maxpages = mp;
        for (int L = l_begin; L < l_end; ++L) {
            LayerW &w = layers[size_t(L)];
            if (w.full) {
                w.kc.resize(size_t(pages) * PAGE * cfg.nkv * cfg.hd);
                w.vc.resize(size_t(pages) * PAGE * cfg.nkv * cfg.hd);
                HIP_CHECK(hipMemset(w.kc.data(), 0, w.kc.bytes()));
                HIP_CHECK(hipMemset(w.vc.data(), 0, w.vc.bytes()));
            } else {
                w.ring.resize(size_t(slots) * cfg.conv_dim() * cfg.conv_k);
                w.dstate.resize(size_t(slots) * cfg.lin_nv * cfg.lin_kd * cfg.lin_vd);
            }
        }
        ptab_d.resize(size_t(slots) * mp);
        HIP_CHECK(hipMemset(ptab_d.data(), 0, ptab_d.bytes()));
    }
    void reset_slot(int slot) {
        for (int L = l_begin; L < l_end; ++L) {
            LayerW &w = layers[size_t(L)];
            if (w.full) continue;
            const size_t rn = size_t(cfg.conv_dim()) * cfg.conv_k, sn = size_t(cfg.lin_nv) * cfg.lin_kd * cfg.lin_vd;
            HIP_CHECK(hipMemsetAsync(w.ring.data() + slot * rn, 0, rn * 4, stream));
            HIP_CHECK(hipMemsetAsync(w.dstate.data() + slot * sn, 0, sn * 4, stream));
        }
    }
    void copy_slot(int src, int dst, const std::vector<int> &sp, const std::vector<int> &dp) {
        const size_t rn = size_t(cfg.conv_dim()) * cfg.conv_k, sn = size_t(cfg.lin_nv) * cfg.lin_kd * cfg.lin_vd;
        const size_t pe = size_t(PAGE) * cfg.nkv * cfg.hd;
        for (int L = l_begin; L < l_end; ++L) {
            LayerW &w = layers[size_t(L)];
            if (w.full) {
                for (size_t i = 0; i < dp.size(); ++i) {
                    HIP_CHECK(hipMemcpyAsync(w.kc.data() + dp[i] * pe, w.kc.data() + sp[i] * pe, pe * 2,
                                             hipMemcpyDeviceToDevice, stream));
                    HIP_CHECK(hipMemcpyAsync(w.vc.data() + dp[i] * pe, w.vc.data() + sp[i] * pe, pe * 2,
                                             hipMemcpyDeviceToDevice, stream));
                }
            } else {
                HIP_CHECK(hipMemcpyAsync(w.ring.data() + dst * rn, w.ring.data() + src * rn, rn * 4,
                                         hipMemcpyDeviceToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(w.dstate.data() + dst * sn, w.dstate.data() + src * sn, sn * 4,
                                         hipMemcpyDeviceToDevice, stream));
            }
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    // ---- batched decode (serving): one token per row, rows padded to a graph bucket
    int bmax = 0;
    DeviceBuffer<int> meta_d;    // [tokens | slots | pos] x bmax
    int *meta_h = nullptr;       // pinned
    DeviceBuffer<uint16_t> bq, bG, battn;
    DeviceBuffer<float> battn_scr, cand_v;
    DeviceBuffer<int> cand_i;
    float *cand_vh = nullptr;    // pinned [bmax][TOPK]
    int *cand_ih = nullptr;
    static constexpr int TOPK = 64;
    struct Graph {
        hipGraphExec_t exec{};
        unsigned exchanges = 0;
    };
    std::map<int, Graph> graphs;
    void alloc_batch(int bm);
    void decode_enqueue(int B);
    void decode_launch(int B);  // graph per bucket (captured on first use) unless QWEN_NO_GRAPH
    void topk_rows(int nrows);  // logits_d rows -> cand (device) -> host
    // candidates of the given state rows (gathered into eX): final norm, lm_head, top-k -> host
    void cand_rows(const std::vector<int> &rows) {
        const int H = cfg.hidden;
        for (size_t i = 0; i < rows.size(); ++i)
            HIP_CHECK(hipMemcpyAsync(eX.data() + i * H, state.data() + size_t(rows[i]) * H, size_t(H) * 2,
                                     hipMemcpyDeviceToDevice, stream));
        const int nr = int(rows.size());
        q_zc_norm(eX.data(), nr, H, final_norm.data(), cfg.eps, stream);
        if (lm8.fp8) run_linear(lm8, eX.data(), H, nr, logits_d.data());
        else q_bf16_gemv_any(lm_head_w.size() ? lm_head_w.data() : embed.data(), cfg.vocab, H, eX.data(), H, nr,
                             logits_d.data(), cfg.vocab, stream);
        topk_rows(nr);
    }
    void moe_block(LayerW &w, int rows);
    void lm_head_rows(int row0, int nrows);  // logits_all[row0..] from state rows
    void reset();
};

qtip::HostLinear QwenStage::load_host(const QwenCheckpoint &ck, int layer, const std::string &name,
                                            int in, int out) {
    const std::string base = layer_prefix + std::to_string(layer) + "." + name + ".";
    ck.tensor(base + "trellis");  // existence + index check
    const SafeTensorFile &f = ck.tensor(base + "trellis");
    qtip::HostLinear l;
    l.in = in;
    l.out = out;
    auto raw = f.bytes(base + "trellis");
    l.trellis.resize(raw.size() / 2);
    std::memcpy(l.trellis.data(), raw.data(), raw.size());
    auto floats = [&](const std::string &n, size_t size) {
        auto v = f.floats(n);
        if (v.size() != size) throw std::runtime_error("invalid QTIP vector " + n);
        std::vector<float> r(size);
        for (size_t i = 0; i < size; ++i) r[i] = v[i];
        return r;
    };
    l.su = floats(base + "SU", size_t(in));
    l.sv = floats(base + "SV", size_t(out));
    auto t = f.bytes(base + "tlut");
    l.tlut.resize(1024);
    std::memcpy(l.tlut.data(), t.data(), t.size());
    return l;
}

void QwenStage::upload_linear(Linear &L, const qtip::HostLinear &h) {
    L.lin.upload({std::span<const uint16_t>(h.trellis)}, h.in);
    L.sv.upload(h.sv);
}

// Loads `model.layers.<layer>.<name>.` as a QTIP trellis linear, or — when the checkpoint stores a
// plain BF16 `weight` tensor there instead (unquantized model) — as a bf16 linear.
void QwenStage::load_linear(Linear &L, const QwenCheckpoint &ck, int layer, const std::string &name,
                                  int in, int out, const Slice &sl) {
    const std::string base = layer_prefix + std::to_string(layer) + "." + name + ".";
    if (ck.index.count(base + "trellis")) {
        if (!sl.whole()) throw std::runtime_error("tensor parallel is not supported for QTIP trellis linears");
        auto h = load_host(ck, layer, name, in, out);
        if (h.in != in || h.out != out)
            throw std::runtime_error("linear dims mismatch for " + base + ": trellis " +
                                     std::to_string(h.out) + "x" + std::to_string(h.in) + " vs config " +
                                     std::to_string(out) + "x" + std::to_string(in));
        upload_linear(L, h);
        L.su.upload(h.su);
        L.bf16 = false;
        L.m = out;
        L.n = in;
    } else if (ck.index.count(base + "weight_scale_inv")) {
        const SafeTensorFile &f = ck.tensor(base + "weight");
        if (f.dtype(base + "weight") != "F8_E4M3")
            throw std::runtime_error("expected F8_E4M3 weight for " + base);
        auto shp = f.shape(base + "weight");
        if (shp.size() != 2 || int(shp[0]) != out || int(shp[1]) != in)
            throw std::runtime_error("fp8 linear dims mismatch for " + base);
        L.bf16 = L.fp8 = true;
        L.m = sl.nrows(out);
        L.n = sl.ncols(in);
        auto raw = f.bytes(base + "weight");
        const uint8_t *w8 = reinterpret_cast<const uint8_t *>(raw.data());
        if (sl.whole())
            L.w8.upload(std::span<const uint8_t>(w8, raw.size()));
        else
            L.w8.upload(slice2d(w8, out, in, sl));
        const SafeTensorFile &fs = ck.tensor(base + "weight_scale_inv");
        auto sv = fs.floats(base + "weight_scale_inv");
        const int bo = (out + 127) / 128, bi = (in + 127) / 128;
        if (sv.size() != size_t(bo) * bi) throw std::runtime_error("fp8 scale dims mismatch for " + base);
        std::vector<float> sc(sv.size());
        for (size_t i = 0; i < sc.size(); ++i) sc[i] = sv[i];
        L.ws.upload(sl.whole() ? sc : slice2d(sc.data(), bo, bi, sl.blocks(128)));
    } else if (ck.index.count(base + "weight")) {
        L.bf16 = true;
        L.m = sl.nrows(out);
        L.n = sl.ncols(in);
        auto w = bf16_raw(ck, base + "weight");
        L.w.upload(sl.whole() ? w : slice2d(w.data(), out, in, sl));
    } else {
        throw std::runtime_error("no trellis or bf16 weight for " + base);
    }
}

// y[rows][out] = x[rows][ldx] . L^T (fp32 out, contiguous rows; trellis slice-summed or dense bf16
// GEMM).
void QwenStage::run_linear(const Linear &L, const uint16_t *x, int ldx, int rows, float *y, int ldy_override) {
    const int ldy = ldy_override ? ldy_override : L.m;
    static const bool legacy = std::getenv("QWEN_FP8_LEGACY") != nullptr;
    if (L.fp8 && !legacy && prefill_big) {
        q_act_quant(x, ldx, rows, L.n, xq8.data(), xqs.data(), stream);
        q_w8a8_big(L.w8.data(), L.ws.data(), L.m, L.n, xq8.data(), xqs.data(), rows, y, ldy, stream);
    } else if (L.fp8 && !legacy) {
        q_act_quant(x, ldx, rows, L.n, xq8.data(), xqs.data(), stream);
        q_w8a8(L.w8.data(), L.ws.data(), L.m, L.n, xq8.data(), xqs.data(), rows, y, ldy,
               gpart.data(), stream);
    } else if (L.fp8) {
        if (rows <= 8) {
            q_fp8_gemv(L.w8.data(), L.ws.data(), L.m, L.n, x, ldx, rows, y, ldy, stream);
        } else {
            if (wtmp.size() < size_t(L.m) * L.n) throw std::runtime_error("fp8 scratch too small");
            q_fp8_dequant(L.w8.data(), L.ws.data(), L.m, L.n, wtmp.data(), stream);
            dense.gemm(L.m, L.n, reinterpret_cast<const BFloat16 *>(x), ldx, rows, y, ldy,
                       reinterpret_cast<const BFloat16 *>(wtmp.data()), stream);
        }
    } else if (L.bf16) {
        if (rows <= 8)
            q_bf16_gemv(L.w.data(), L.m, L.n, x, ldx, rows, y, ldy, stream);
        else
            dense.gemm(L.m, L.n, reinterpret_cast<const BFloat16 *>(x), ldx, rows, y, ldy,
                       reinterpret_cast<const BFloat16 *>(L.w.data()), stream);
    } else {
        int slices = qtip::matmul_slices(L.lin.view(), rows);
        qtip::matmul(L.lin.view(), codebook.lut.data(), reinterpret_cast<const BFloat16 *>(x), ldx, rows,
                     partial.data(), slices, stream);
        q_sum_slices(partial.data(), slices, size_t(rows) * L.lin.rows, y, stream);
    }
}

void QwenStage::load(const std::filesystem::path &dir) {
    QwenCheckpoint ck(dir);
    // cfg (local dims, max_context) is set by the constructor; gcfg = checkpoint dims
    gcfg = QwenRuntimeConfig::read(dir);
    gcfg.max_context = cfg.max_context;
    const QwenRuntimeConfig &G = gcfg;
    const int r = tp_rank;
    auto rows1 = [](int a, int n) { Slice s; s.rows = {{a, n}}; return s; };
    auto cols1 = [](int a, int n) { Slice s; s.c0 = a; s.nc = n; return s; };
    if (tp_size > 1 && G.num_experts > 0) throw std::runtime_error("tensor parallel: MoE not supported");
    layer_prefix = "model.layers.";
    if (!ck.index.count("model.layers.0.input_layernorm.weight") &&
        ck.index.count("model.language_model.layers.0.input_layernorm.weight"))
        layer_prefix = "model.language_model.layers.";
    const bool f0 = cfg.is_full(0);
    const std::string probe_name = f0 ? "self_attn.q_proj" : "linear_attn.in_proj_qkv";
    if (ck.index.count(layer_prefix + "0." + probe_name + ".trellis")) {
        // the codebook comes from the first trellis' tlut; a fully bf16 checkpoint never needs one
        qtip::HostLinear probe =
            load_host(ck, 0, probe_name, cfg.hidden, f0 ? cfg.nq * 2 * cfg.hd : cfg.conv_dim());
        qtip::Codebook cb = qtip::make_codebook(std::span<const uint16_t>(probe.tlut));
        codebook.upload(cb);
    }
    const int H = cfg.hidden, V = cfg.value_dim(), A = cfg.nq * cfg.hd;
    layers.resize(size_t(cfg.layers));
    for (int L = l_begin; L < l_end; ++L) {
        LayerW &w = layers[size_t(L)];
        w.full = cfg.is_full(L);
        w.moe = cfg.num_experts > 0;
        std::string p = layer_prefix + std::to_string(L) + ".";
        auto up16 = [&](DeviceBuffer<uint16_t> &d, const std::string &name) {
            d.upload(bf16_raw(ck, p + name));
        };
        auto up16s = [&](DeviceBuffer<uint16_t> &d, const std::string &name, int nrows, int ncols,
                         const Slice &sl) {
            auto v = bf16_raw(ck, p + name);
            if (v.size() != size_t(nrows) * ncols) throw std::runtime_error("unexpected size of " + p + name);
            d.upload(sl.whole() ? v : slice2d(v.data(), nrows, ncols, sl));
        };
        const int hq = cfg.nq, hk = cfg.nkv, hd = G.hd;            // local heads
        const int nkl = cfg.lin_nk, nvl = cfg.lin_nv, kd = G.lin_kd, vd = G.lin_vd;
        const int Fl = cfg.inter;
        Slice s_qkv;  // DeltaNet q | k | v channel shards (in_proj_qkv rows, conv channels)
        s_qkv.rows = {{r * nkl * kd, nkl * kd}, {G.key_dim() + r * nkl * kd, nkl * kd},
                      {2 * G.key_dim() + r * nvl * vd, nvl * vd}};
        if (w.full) {
            load_linear(w.lq, ck, L, "self_attn.q_proj", H, G.nq * 2 * hd, rows1(r * hq * 2 * hd, hq * 2 * hd));
            load_linear(w.lk, ck, L, "self_attn.k_proj", H, G.nkv * hd, rows1(r * hk * hd, hk * hd));
            load_linear(w.lv, ck, L, "self_attn.v_proj", H, G.nkv * hd, rows1(r * hk * hd, hk * hd));
            load_linear(w.lo, ck, L, "self_attn.o_proj", G.nq * hd, H, cols1(r * hq * hd, hq * hd));
            up16(w.q_norm, "self_attn.q_norm.weight");
            up16(w.k_norm, "self_attn.k_norm.weight");
        } else {
            load_linear(w.l_in_qkv, ck, L, "linear_attn.in_proj_qkv", H, G.conv_dim(), s_qkv);
            load_linear(w.l_in_z, ck, L, "linear_attn.in_proj_z", H, G.value_dim(), rows1(r * nvl * vd, nvl * vd));
            load_linear(w.l_out, ck, L, "linear_attn.out_proj", G.value_dim(), H, cols1(r * nvl * vd, nvl * vd));
            {  // b|a: raw bf16 rows [2nv][H]
                std::vector<uint16_t> b = bf16_raw(ck, p + "linear_attn.in_proj_b.weight");
                std::vector<uint16_t> a = bf16_raw(ck, p + "linear_attn.in_proj_a.weight");
                if (tp_size > 1) {
                    b = slice2d(b.data(), G.lin_nv, H, rows1(r * nvl, nvl));
                    a = slice2d(a.data(), G.lin_nv, H, rows1(r * nvl, nvl));
                }
                b.insert(b.end(), a.begin(), a.end());
                w.w_ba.upload(b);
            }
            up16s(w.conv_w, "linear_attn.conv1d.weight", G.conv_dim(), G.conv_k, tp_size > 1 ? s_qkv : Slice{});
            up16(w.dn_norm, "linear_attn.norm.weight");
            {
                auto al = ck.tensor(p + "linear_attn.A_log").floats(p + "linear_attn.A_log");
                auto db = ck.tensor(p + "linear_attn.dt_bias").floats(p + "linear_attn.dt_bias");
                std::vector<float> aux;
                for (int i = r * nvl; i < (r + 1) * nvl; ++i) aux.push_back(al[size_t(i)]);
                for (int i = r * nvl; i < (r + 1) * nvl; ++i) aux.push_back(db[size_t(i)]);
                w.a_dt.upload(aux);
            }
        }
        if (w.moe) {
            const int E = cfg.num_experts, F = cfg.moe_inter, S = cfg.shared_inter;
            up16(w.w_router, "mlp.gate.weight");
            if (cfg.shared_gate) up16(w.w_sg, "mlp.shared_expert_gate.weight");
            w.e_gate.resize(size_t(E));
            w.e_up.resize(size_t(E));
            w.e_down.resize(size_t(E));
            for (int e = 0; e < E; ++e) {
                std::string ep = "mlp.experts." + std::to_string(e) + ".";
                load_linear(w.e_gate[size_t(e)], ck, L, ep + "gate_proj", H, F);
                load_linear(w.e_up[size_t(e)], ck, L, ep + "up_proj", H, F);
                load_linear(w.e_down[size_t(e)], ck, L, ep + "down_proj", F, H);
            }
            load_linear(w.sh_gate, ck, L, "mlp.shared_expert.gate_proj", H, S);
            load_linear(w.sh_up, ck, L, "mlp.shared_expert.up_proj", H, S);
            load_linear(w.sh_down, ck, L, "mlp.shared_expert.down_proj", S, H);
        } else {
            load_linear(w.lgate, ck, L, "mlp.gate_proj", H, G.inter, rows1(r * Fl, Fl));
            load_linear(w.lup, ck, L, "mlp.up_proj", H, G.inter, rows1(r * Fl, Fl));
            load_linear(w.ldown, ck, L, "mlp.down_proj", G.inter, H, cols1(r * Fl, Fl));
        }
        up16(w.ln_in, "input_layernorm.weight");
        up16(w.ln_post, "post_attention_layernorm.weight");
        if (!w.moe && w.lgate.fp8 && w.ldown.fp8 && (w.full ? (w.lq.fp8 && w.lk.fp8 && w.lv.fp8 && w.lo.fp8)
                                                            : (w.l_in_qkv.fp8 && w.l_in_z.fp8 && w.l_out.fp8))) {
            w.fused = true;
            if (w.full) fuse_fp8(w.f_qkv, {&w.lq, &w.lk, &w.lv});
            else fuse_fp8(w.f_qkvz, {&w.l_in_qkv, &w.l_in_z});
            fuse_fp8(w.f_gu, {&w.lgate, &w.lup});
        }
    }
    std::string embed_name, norm_name;
    for (std::string nm : {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"})
        if (ck.index.count(nm)) embed_name = nm;
    for (std::string nm : {"model.language_model.norm.weight", "model.norm.weight"})
        if (ck.index.count(nm)) norm_name = nm;
    if (embed_name.empty() || norm_name.empty())
        throw std::runtime_error("cannot find embed_tokens / final norm in the checkpoint index");
    if (first || (last && cfg.tied)) embed.upload(bf16_raw(ck, embed_name));
    if (last) final_norm.upload(bf16_raw(ck, norm_name));
    if (last && !cfg.tied) {
        if (!ck.index.count("lm_head.weight"))
            throw std::runtime_error("tie_word_embeddings=false but no lm_head.weight");
        auto lw = bf16_raw(ck, "lm_head.weight");
        lm_head_w.upload(tp_size > 1 ? slice2d(lw.data(), G.vocab, H, rows1(voff, cfg.vocab)) : lw);
    }
    if (last && lm_head_w.size() && !std::getenv("QWEN_LMHEAD_BF16") && cfg.vocab % 128 == 0 && cfg.hidden % 128 == 0) {
        lm8.bf16 = lm8.fp8 = true;
        lm8.m = cfg.vocab;
        lm8.n = cfg.hidden;
        lm8.w8.resize(size_t(cfg.vocab) * cfg.hidden);
        lm8.ws.resize(size_t(cfg.vocab / 128) * (cfg.hidden / 128));
        q_quant_weight_bf16(lm_head_w.data(), cfg.vocab, cfg.hidden, lm8.w8.data(), lm8.ws.data(), stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        lm_head_w = DeviceBuffer<uint16_t>{};
    }
    // scratch
    const int R = chunk;
    const int FS = std::max(cfg.inter, cfg.shared_inter);
    const int Wz = std::max({cfg.nq * 2 * cfg.hd + 2 * cfg.nkv * cfg.hd,
                             cfg.conv_dim() + V + 2 * cfg.lin_nv, 2 * FS});
    state.resize(size_t(R) * H);
    h_norm.resize(size_t(R) * H);
    u.resize(size_t(R) * std::max({H, V, A, FS}));
    q.resize(size_t(R) * cfg.nq * 2 * cfg.hd);
    k.resize(size_t(R) * cfg.nkv * cfg.hd);
    v.resize(size_t(R) * cfg.nkv * cfg.hd);
    gate.resize(size_t(R) * A);
    attn_out.resize(size_t(R) * A);
    o_in.resize(size_t(R) * V);
    gu.resize(size_t(R) * FS);
    eX.resize(size_t(R) * H);
    z.resize(size_t(R) * Wz);
    z2.resize(size_t(R) * Wz);
    tokens_d.resize(size_t(R));
    attn_part.resize(size_t(std::max(R, 1024 / std::max(1, cfg.nkv) + 1)) * cfg.nq * 8 * size_t(cfg.hd + 2));
    // bf16 linears reuse the trellis epilogues with SV = 1: must cover the widest output row, which
    // is q_proj (nq * 2 * hd = 4096 on 0.8B) -- sizing by max(inter, hidden) read past the end
    ones_f.resize(size_t(std::max({FS, H, cfg.nq * 2 * cfg.hd, cfg.nkv * cfg.hd, cfg.conv_dim(), V})));
    {
        std::vector<float> ones(ones_f.size(), 1.0f);
        ones_f.upload(ones);
    }
    rlog.resize(size_t(R) * std::max(1, cfg.num_experts));
    rwts.resize(size_t(R) * std::max(1, cfg.topk));
    y_moe.resize(size_t(R) * H);
    sg_logit.resize(size_t(R));
    slot_w.resize(size_t(R) * std::max(1, cfg.topk));
    rids.resize(size_t(R) * std::max(1, cfg.topk));
    rcnt.resize(size_t(std::max(1, cfg.num_experts)));
    rbase.resize(size_t(std::max(1, cfg.num_experts)));
    rrun.resize(size_t(std::max(1, cfg.num_experts)));
    slot_tok.resize(size_t(R) * std::max(1, cfg.topk));
    for (auto &w : layers) {
        std::vector<const Linear *> lins = {&w.lq, &w.lk,      &w.lv,   &w.lo,   &w.lgate, &w.lup,
                                            &w.ldown, &w.l_in_qkv, &w.l_in_z, &w.l_out, &w.sh_gate,
                                            &w.sh_up, &w.sh_down};
        for (auto &e : w.e_gate) lins.push_back(&e);
        for (auto &e : w.e_up) lins.push_back(&e);
        for (auto &e : w.e_down) lins.push_back(&e);
        for (const Linear *gv : lins)
            if (!gv->bf16 && gv->lin.rows)
                max_slices = std::max<size_t>(max_slices, qtip::matmul_slices(gv->lin.view(), R));
    }
    partial.resize(max_slices * size_t(R) * Wz);
    {
        size_t wmax = 0, nmax = 0, pmax = 0;
        for (auto &w : layers)
            for (const Linear *gv : {&w.lq, &w.lk, &w.lv, &w.lo, &w.lgate, &w.lup, &w.ldown, &w.l_in_qkv,
                                     &w.l_in_z, &w.l_out, &w.f_qkv, &w.f_qkvz, &w.f_gu, &lm8})
                if (gv->fp8) {
                    wmax = std::max(wmax, size_t(gv->m) * gv->n);
                    nmax = std::max<size_t>(nmax, size_t(gv->n));
                    pmax = std::max(pmax, q_w8a8_small_scratch(gv->m, gv->n, 64));
                }
        if (R > 8 && std::getenv("QWEN_FP8_LEGACY")) wtmp.resize(wmax);
        if (nmax) {
            const size_t rr = size_t(std::max(R, 64));
            xq8.resize(rr * nmax);
            xqs.resize(rr * (nmax / 128));
            gpart.resize(std::max<size_t>(pmax, 1));
        }
    }
    if (tp_size > 1) {
        ar_slot_bytes = long(std::max(R, 64)) * H * 4;  // host slots: alloc_ar_host() (reader's NUMA node)
        ar_ctr.resize(4);
        HIP_CHECK(hipMemset(ar_ctr.data(), 0, ar_ctr.bytes()));
    }
    if (tp_size > 1) {
        for (auto &b : zsend) b.resize(size_t(R) * H);
        zrecv.resize(size_t(R) * H);
        for (int i = 0; i < 2; ++i) {
            HIP_CHECK(hipEventCreateWithFlags(&send_ev[i], hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&pulled_ev[i], hipEventDisableTiming));
            HIP_CHECK(hipEventRecord(send_ev[i], stream));
            HIP_CHECK(hipEventRecord(pulled_ev[i], stream));
        }
    }
    if (!last) return;
    logits_all.resize(size_t(std::min(R, 16)) * cfg.vocab);
    logits_d.resize(size_t(std::min(R, 16)) * cfg.vocab);
    logits_h.resize(size_t(cfg.vocab));
}

// logits of state rows [row0, row0+nrows) -> logits_all (sub-batches of <= 16 rows)
void QwenStage::lm_head_rows(int row0, int nrows) {
    lm_head_enqueue(row0, nrows);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (const char *lp = std::getenv("QWEN_LOGITS_OUT")) {
        static bool once = false;
        if (!once) {
            once = true;
            std::ofstream f(lp, std::ios::binary);
            f.write(reinterpret_cast<const char *>(logits_all.data()), std::streamsize(size_t(nrows) * cfg.vocab * 4));
        }
    }
    if (std::getenv("QWEN_RT_DBG")) {
        float *row = logits_all.data();
        for (int k = 0; k < 3; ++k) {
            int am = 0;
            for (long v = 1; v < cfg.vocab; ++v)
                if (row[size_t(v)] > row[size_t(am)]) am = int(v);
            std::fprintf(stderr, "logit top%d: id %d val %.4f; samples: %.3f %.3f %.3f\n", k, am,
                         row[size_t(am)], row[0], row[1], row[100]);
            row[size_t(am)] = -1e30f;
        }
    }
}

void QwenStage::lm_head_enqueue(int row0, int nrows) {
    const int H = cfg.hidden;
    const uint16_t *head = lm_head_w.size() ? lm_head_w.data() : embed.data();
    q_zc_norm(state.data() + size_t(row0) * H, nrows, H, final_norm.data(), cfg.eps, stream);
    for (int r0 = 0; r0 < nrows; r0 += 16) {
        int nr = std::min(16, nrows - r0);
        if (lm8.fp8)
            run_linear(lm8, state.data() + size_t(row0 + r0) * H, H, nr, logits_d.data() + size_t(r0) * cfg.vocab);
        else if (nr <= 8)
            q_bf16_gemv(head, cfg.vocab, H, state.data() + size_t(row0 + r0) * H, H, nr,
                        logits_d.data() + size_t(r0) * cfg.vocab, cfg.vocab, stream);
        else
            dense.gemm(cfg.vocab, H,
                       reinterpret_cast<const BFloat16 *>(state.data() + size_t(row0 + r0) * H), H, nr,
                       logits_d.data() + size_t(r0) * cfg.vocab, cfg.vocab,
                       reinterpret_cast<const BFloat16 *>(head), stream);
        HIP_CHECK(hipMemcpyAsync(logits_all.data() + size_t(r0) * cfg.vocab,
                                 logits_d.data() + size_t(r0) * cfg.vocab, size_t(nr) * cfg.vocab * 4,
                                 hipMemcpyDeviceToHost, stream));
    }
}

// MoE MLP block: router (softmax -> top-k -> renorm), routed experts on compact per-expert token
// lists, sigmoid-gated shared expert. Experts are QTIP trellis linears or plain bf16 (same Linear).
void QwenStage::moe_block(LayerW &w, int rows) {
    const QwenRuntimeConfig &c = cfg;
    const int H = c.hidden, E = c.num_experts, K = c.topk, F = c.moe_inter, S = c.shared_inter;
    const bool e16 = !w.e_gate.empty() && w.e_gate[0].bf16;
    // h_norm = zc-norm(state, ln_post): router + shared-gate + bf16-expert input
    q_zc_norm_to(h_norm.data(), state.data(), rows, H, w.ln_post.data(), c.eps, stream);
    dense.gemm(E, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows, rlog.data(), E,
               reinterpret_cast<const BFloat16 *>(w.w_router.data()), stream);
    HIP_CHECK(hipMemsetAsync(rcnt.data(), 0, rcnt.bytes(), stream));
    q_moe_route(rlog.data(), rows, E, K, rwts.data(), rids.data(), rcnt.data(), stream);
    q_moe_scan(rcnt.data(), rbase.data(), E, stream);
    HIP_CHECK(hipMemsetAsync(rrun.data(), 0, rrun.bytes(), stream));
    q_moe_route_fill(rlog.data(), rows, E, K, rbase.data(), rrun.data(), slot_tok.data(), slot_w.data(),
                     stream);
    std::vector<int> cnt{size_t(E)}, base{size_t(E)};
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(cnt.data(), rcnt.data(), size_t(E) * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(base.data(), rbase.data(), size_t(E) * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemsetAsync(y_moe.data(), 0, size_t(rows) * H * 4, stream));
    // routed experts
    for (int e = 0; e < E; ++e) {
        const int ne = cnt[size_t(e)];
        if (ne <= 0) continue;
        const Linear &lg = w.e_gate[size_t(e)];
        const Linear &lu = w.e_up[size_t(e)];
        const Linear &ld = w.e_down[size_t(e)];
        const int *tok_e = slot_tok.data() + base[size_t(e)];
        const float *w_e = slot_w.data() + base[size_t(e)];
        q_moe_gather(state.data(), H, tok_e, eX.data(), ne, stream);  // raw rows
        if (!e16) {
            q_input_transform(eX.data(), ne, H, w.ln_post.data(), c.eps, lg.su.data(), H, u.data(), H,
                              stream);
            run_linear(lg, u.data(), H, ne, z.data());
            q_hadamard_rows(z.data(), ne, F, F, false, stream);
            run_linear(lu, u.data(), H, ne, z2.data());
            q_hadamard_rows(z2.data(), ne, F, F, false, stream);
            q_gateup(z.data(), z2.data(), lg.sv.data(), lu.sv.data(), gu.data(), ne, F, F, stream);
            q_input_transform(gu.data(), ne, F, nullptr, 0, ld.su.data(), F, u.data(), F, stream);
            run_linear(ld, u.data(), F, ne, z.data());
            q_hadamard_rows(z.data(), ne, H, H, false, stream);
            q_moe_scatter_sv_add(y_moe.data(), z.data(), ld.sv.data(), tok_e, w_e, ne, H, stream);
        } else {
            q_zc_norm(eX.data(), ne, H, w.ln_post.data(), c.eps, stream);
            run_linear(lg, eX.data(), H, ne, z.data());
            run_linear(lu, eX.data(), H, ne, z2.data());
            q_gateup(z.data(), z2.data(), ones_f.data(), ones_f.data(), gu.data(), ne, F, F, stream);
            run_linear(ld, gu.data(), F, ne, z.data());
            q_moe_scatter_sv_add(y_moe.data(), z.data(), ones_f.data(), tok_e, w_e, ne, H, stream);
        }
    }
    // shared expert (+ sigmoid gate)
    const bool s16 = w.sh_gate.bf16;
    if (!s16) {
        q_input_transform(state.data(), rows, H, w.ln_post.data(), c.eps, w.sh_gate.su.data(), H,
                          u.data(), H, stream);
        run_linear(w.sh_gate, u.data(), H, rows, z.data());
        q_hadamard_rows(z.data(), rows, S, S, false, stream);
        run_linear(w.sh_up, u.data(), H, rows, z2.data());
        q_hadamard_rows(z2.data(), rows, S, S, false, stream);
        q_gateup(z.data(), z2.data(), w.sh_gate.sv.data(), w.sh_up.sv.data(), gu.data(), rows, S, S,
                 stream);
        q_input_transform(gu.data(), rows, S, nullptr, 0, w.sh_down.su.data(), S, u.data(), S, stream);
        run_linear(w.sh_down, u.data(), S, rows, z.data());
        q_hadamard_rows(z.data(), rows, H, H, false, stream);
        if (c.shared_gate) {
            dense.gemm(1, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows, sg_logit.data(),
                       1, reinterpret_cast<const BFloat16 *>(w.w_sg.data()), stream);
            q_moe_sv_add_gated(z.data(), w.sh_down.sv.data(), sg_logit.data(), state.data(), rows, H,
                               stream);
        } else {
            q_moe_sv_add_gated(z.data(), w.sh_down.sv.data(), nullptr, state.data(), rows, H, stream);
        }
    } else {
        run_linear(w.sh_gate, h_norm.data(), H, rows, z.data());
        run_linear(w.sh_up, h_norm.data(), H, rows, z2.data());
        q_gateup(z.data(), z2.data(), ones_f.data(), ones_f.data(), gu.data(), rows, S, S, stream);
        run_linear(w.sh_down, gu.data(), S, rows, z.data());
        if (c.shared_gate) {
            dense.gemm(1, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows, sg_logit.data(),
                       1, reinterpret_cast<const BFloat16 *>(w.w_sg.data()), stream);
            q_moe_sv_add_gated(z.data(), ones_f.data(), sg_logit.data(), state.data(), rows, H, stream);
        } else {
            q_moe_sv_add_gated(z.data(), ones_f.data(), nullptr, state.data(), rows, H, stream);
        }
    }
    q_moe_residual_add(state.data(), y_moe.data(), rows, H, stream);
}

void QwenStage::forward_chunk(const int *tokens_h, int rows, int slot, int pos0) {
    const QwenRuntimeConfig &c = cfg;
    const int H = c.hidden, A = c.nq * c.hd, V = c.value_dim(), F = c.inter;
    const std::vector<Seg> sv = segs.empty() ? std::vector<Seg>{{slot, pos0, 0, rows}} : segs;
    if (sv.size() > 1 && (c.hd != 256 || std::getenv("QWEN_DELTA_V1") || std::getenv("QWEN_DELTA_STEP")))
        throw std::runtime_error("multi-sequence prefill needs the split attention + DeltaNet v2 kernels");
    const size_t ring_off = size_t(slot) * c.conv_dim() * c.conv_k;
    const size_t st_off = size_t(slot) * c.lin_nv * c.lin_kd * c.lin_vd;
    if (next) HIP_CHECK(hipStreamWaitEvent(stream, next->took_ev, 0));  // state free to overwrite
    if (first) {
        if (rows == 1 && tok_pinned) {  // decode: callers sync every step, the pinned slot is free
            *tok_pinned = *tokens_h;
            HIP_CHECK(hipMemcpyAsync(tokens_d.data(), tok_pinned, 4, hipMemcpyHostToDevice, stream));
        } else {
            HIP_CHECK(hipMemcpyAsync(tokens_d.data(), tokens_h, size_t(rows) * 4, hipMemcpyHostToDevice, stream));
        }
        q_embed_gather(embed.data(), tokens_d.data(), state.data(), rows, H, stream);
    } else {  // hidden state from the previous pipeline stage (other GPU)
        HIP_CHECK(hipStreamWaitEvent(stream, prev->done_ev, 0));
        if (p2p)
            q_copy_u16(state.data(), prev->state.data(), long(rows) * H, stream);
        else
            HIP_CHECK(hipMemcpyPeerAsync(state.data(), device, prev->state.data(), prev->device,
                                         size_t(rows) * H * 2, stream));
        HIP_CHECK(hipEventRecord(took_ev, stream));
    }
    if (std::getenv("QWEN_RT_DBG")) {
        HIP_CHECK(hipStreamSynchronize(stream));  // download() is a null-stream copy
        std::vector<uint16_t> h = state.download();
        std::fprintf(stderr, "embed row0: %.4f %.4f %.4f %.4f  raw %04x %04x\n",
                     float(BFloat16::from_bits(h[0])), float(BFloat16::from_bits(h[1])),
                     float(BFloat16::from_bits(h[2])), float(BFloat16::from_bits(h[3])), h[0], h[1]);
    }
    // one trellis projection: rotated input is already in u (row stride n); z <- sum slices (dense f32)
    auto proj = [&](const Linear &lin, int n, float *zdst) {
        int slices = qtip::matmul_slices(lin.lin.view(), rows);
        qtip::matmul(lin.lin.view(), codebook.lut.data(), reinterpret_cast<const BFloat16 *>(u.data()), n,
                     rows, partial.data(), slices, stream);
        q_sum_slices(partial.data(), slices, size_t(rows) * lin.lin.rows, zdst, stream);
    };
    for (int L = l_begin; L < l_end; ++L) {
        LayerW &w = layers[size_t(L)];
        const bool lin16 = w.fused || (w.full ? w.lq.bf16 : w.l_in_qkv.bf16);  // bf16 / fp8 (non-trellis) linears
        auto zc_hidden = [&]() {  // h_norm = zc-norm(state, ln_in): plain input of bf16 linears
            q_zc_norm_to(h_norm.data(), state.data(), rows, H, w.ln_in.data(), c.eps, stream);
        };
        if (w.full) {
            const int Wq = c.nq * 2 * c.hd, Wk = c.nkv * c.hd;
            if (!lin16) {
                q_input_transform(state.data(), rows, H, w.ln_in.data(), c.eps, w.lq.su.data(), H,
                                  u.data(), H, stream);
                proj(w.lq, H, z.data());
                q_hadamard_rows(z.data(), rows, Wq, Wq, false, stream);
                q_sv_cast(z.data(), w.lq.sv.data(), q.data(), rows, Wq, Wq, stream);
                proj(w.lk, H, z.data());
                q_hadamard_rows(z.data(), rows, Wk, Wk, false, stream);
                q_sv_cast(z.data(), w.lk.sv.data(), k.data(), rows, Wk, Wk, stream);
                proj(w.lv, H, z.data());
                q_hadamard_rows(z.data(), rows, Wk, Wk, false, stream);
                q_sv_cast(z.data(), w.lv.sv.data(), v.data(), rows, Wk, Wk, stream);
            } else if (w.fused) {
                zc_hidden();
                const int Wf = Wq + 2 * Wk;
                run_linear(w.f_qkv, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), q.data(), rows, Wq, Wf, stream);
                q_sv_cast(z.data() + Wq, ones_f.data(), k.data(), rows, Wk, Wf, stream);
                q_sv_cast(z.data() + Wq + Wk, ones_f.data(), v.data(), rows, Wk, Wf, stream);
            } else {
                zc_hidden();
                run_linear(w.lq, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), q.data(), rows, Wq, Wq, stream);
                run_linear(w.lk, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), k.data(), rows, Wk, Wk, stream);
                run_linear(w.lv, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), v.data(), rows, Wk, Wk, stream);
            }
            for (const Seg &sg : sv) {
                const int *spt = ptab_d.data() + size_t(sg.slot) * maxpages;
                q_qk_norm_rope(q.data() + size_t(sg.row0) * Wq, k.data() + size_t(sg.row0) * Wk,
                               v.data() + size_t(sg.row0) * Wk, gate.data() + size_t(sg.row0) * A, w.kc.data(),
                               w.vc.data(), sg.n, sg.pos0, c.nq, c.nkv, c.hd, c.rot_dim, c.theta, w.q_norm.data(),
                               w.k_norm.data(), c.eps, stream, spt);
                if (c.hd == 256)
                    q_attn_split(q.data() + size_t(sg.row0) * Wq, w.kc.data(), w.vc.data(),
                                 gate.data() + size_t(sg.row0) * A, attn_out.data() + size_t(sg.row0) * A, sg.n,
                                 sg.pos0, c.nq, c.nkv, c.hd, 1.0f / std::sqrt(float(c.hd)), attn_part.data(),
                                 attn_part.size(), stream, spt);
            }
            if (c.hd != 256)  // contiguous-cache fallback: single sequence only (identity page table)
                q_attn(q.data(), w.kc.data(), w.vc.data(), gate.data(), attn_out.data(), rows, pos0, c.nq,
                       c.nkv, c.hd, 1.0f / std::sqrt(float(c.hd)), stream);
            if (!w.lo.bf16) {
                q_input_transform(attn_out.data(), rows, A, nullptr, 0, w.lo.su.data(), A, u.data(), A,
                                  stream);
                proj(w.lo, A, z.data());
                q_hadamard_rows(z.data(), rows, H, H, false, stream);
                q_sv_add(z.data(), w.lo.sv.data(), state.data(), rows, H, H, H, stream);
            } else {
                float *zr = rr_buf(z.data(), rows);
                run_linear(w.lo, attn_out.data(), A, rows, zr);
                residual_reduce(zr, rows);
            }
        } else {
            const int Wq = c.conv_dim(), nv = c.lin_nv;
            const int Wz = Wq + V + 2 * nv;
            if (!lin16) {
                auto proj_scatter = [&](const Linear &lin, int n, long col_off) {
                    q_input_transform(state.data(), rows, H, w.ln_in.data(), c.eps, lin.su.data(), H,
                                      u.data(), H, stream);
                    proj(lin, n, z2.data());
                    q_hadamard_rows(z2.data(), rows, lin.lin.rows, lin.lin.rows, false, stream);
                    q_sv_cast_f32(z2.data(), lin.sv.data(), z.data() + col_off, rows, lin.lin.rows,
                                  lin.lin.rows, Wz, stream);
                };
                proj_scatter(w.l_in_qkv, H, 0);
                proj_scatter(w.l_in_z, H, Wq);
            } else if (w.fused) {
                zc_hidden();
                run_linear(w.f_qkvz, h_norm.data(), H, rows, z.data(), Wz);
            } else {
                zc_hidden();
                run_linear(w.l_in_qkv, h_norm.data(), H, rows, z2.data());
                q_scatter_f32(z.data(), Wz, 0, z2.data(), Wq, rows, Wq, stream);
                run_linear(w.l_in_z, h_norm.data(), H, rows, z2.data());
                q_scatter_f32(z.data(), Wz, Wq, z2.data(), V, rows, V, stream);
            }
            // b/a: unquantised bf16 GEMMs on the plain (zc-normed) hidden state
            if (!lin16) {
                q_zc_norm_to(h_norm.data(), state.data(), rows, H, w.ln_in.data(), c.eps, stream);
            }
            if (prefill_big) {  // row-count invariant (serving: packed prompts == alone)
                q_bf16_gemv_any(w.w_ba.data(), 2 * nv, H, h_norm.data(), H, rows, z.data() + Wq + V, Wz, stream);
            } else if (rows <= 8) {  // b|a rows are contiguous [2nv][H] -> one GEMV into cols Wq+V..
                q_bf16_gemv(w.w_ba.data(), 2 * nv, H, h_norm.data(), H, rows, z.data() + Wq + V, Wz, stream);
            } else {
                dense.gemm(nv, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows,
                           z.data() + Wq + V, Wz, reinterpret_cast<const BFloat16 *>(w.w_ba.data()), stream);
                dense.gemm(nv, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows,
                           z.data() + Wq + V + nv, Wz,
                           reinterpret_cast<const BFloat16 *>(w.w_ba.data()) + size_t(cfg.lin_nv) * H,
                           stream);
            }
            static const bool delta_v1 = std::getenv("QWEN_DELTA_V1") != nullptr;
            if (c.lin_kd == 128 && c.lin_vd == 128 && !std::getenv("QWEN_DELTA_STEP") && !delta_v1)
                for (const Seg &sg : sv)
                    q_delta_chunk2(z.data() + size_t(sg.row0) * Wz, w.conv_w.data(), w.dn_norm.data(), w.a_dt.data(),
                                   w.ring.data() + size_t(sg.slot) * c.conv_dim() * c.conv_k,
                                   w.dstate.data() + size_t(sg.slot) * c.lin_nv * c.lin_kd * c.lin_vd,
                                   o_in.data() + size_t(sg.row0) * V, z2.data(), sg.n, sg.pos0, Wq, V, c.lin_nk, nv,
                                   c.conv_k, c.eps, stream);
            else if (c.lin_kd == 128 && c.lin_vd == 128 && !std::getenv("QWEN_DELTA_STEP"))
                q_delta_chunk(z.data(), w.conv_w.data(), w.dn_norm.data(), w.a_dt.data(), w.ring.data() + ring_off,
                              w.dstate.data() + st_off, o_in.data(), z2.data(), rows, pos0, Wq, V, c.lin_kd, c.lin_vd,
                              c.lin_nk, nv, c.conv_k, c.eps, stream);
            else for (int t = 0; t < rows; ++t)
                q_delta_step(z.data(), w.conv_w.data(), w.dn_norm.data(), w.a_dt.data(), w.ring.data() + ring_off,
                             w.dstate.data() + st_off, o_in.data(), rows, t, pos0, Wq, V, c.lin_kd, c.lin_vd,
                             c.lin_nk, nv, c.conv_k, c.eps, stream);
            if (!w.l_out.bf16) {
                q_input_transform(o_in.data(), rows, V, nullptr, 0, w.l_out.su.data(), V, u.data(), V,
                                  stream);
                proj(w.l_out, V, z.data());
                q_hadamard_rows(z.data(), rows, H, H, false, stream);
                q_sv_add(z.data(), w.l_out.sv.data(), state.data(), rows, H, H, H, stream);
            } else {
                float *zr = rr_buf(z.data(), rows);
                run_linear(w.l_out, o_in.data(), V, rows, zr);
                residual_reduce(zr, rows);
            }
        }
        if (std::getenv("QWEN_RT_DBG")) {
            HIP_CHECK(hipStreamSynchronize(stream));  // download() is a null-stream copy
            std::vector<uint16_t> h = state.download();
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                float v;
                uint16_t b = h[size_t(i)];
                uint32_t u = uint32_t(b) << 16;
                std::memcpy(&v, &u, 4);
                ss += double(v) * v;
            }
            std::fprintf(stderr, "L%d %s post-attn norm %.4f\n", L, w.full ? "full" : "lin", std::sqrt(ss / H));
        }
        if (w.moe) {
            moe_block(w, rows);
        } else if (!w.fused && !w.lgate.bf16) {
            q_input_transform(state.data(), rows, H, w.ln_post.data(), c.eps, w.lgate.su.data(), H,
                              u.data(), H, stream);
            proj(w.lgate, H, z.data());
            q_hadamard_rows(z.data(), rows, F, F, false, stream);
            proj(w.lup, H, z2.data());
            q_hadamard_rows(z2.data(), rows, F, F, false, stream);
            q_gateup(z.data(), z2.data(), w.lgate.sv.data(), w.lup.sv.data(), gu.data(), rows, F, F,
                     stream);
            q_input_transform(gu.data(), rows, F, nullptr, 0, w.ldown.su.data(), F, u.data(), F, stream);
            proj(w.ldown, F, z.data());
            q_hadamard_rows(z.data(), rows, H, H, false, stream);
            q_sv_add(z.data(), w.ldown.sv.data(), state.data(), rows, H, H, H, stream);
        } else if (w.fused) {
            q_zc_norm_to(h_norm.data(), state.data(), rows, H, w.ln_post.data(), c.eps, stream);
            run_linear(w.f_gu, h_norm.data(), H, rows, z.data());
            q_gateup(z.data(), z.data() + F, ones_f.data(), ones_f.data(), gu.data(), rows, F, 2 * F, stream);
            float *zr = rr_buf(z.data(), rows);
            run_linear(w.ldown, gu.data(), F, rows, zr);
            residual_reduce(zr, rows);
        } else {
            q_zc_norm_to(h_norm.data(), state.data(), rows, H, w.ln_post.data(), c.eps, stream);
            run_linear(w.lgate, h_norm.data(), H, rows, z.data());
            run_linear(w.lup, h_norm.data(), H, rows, z2.data());
            q_gateup(z.data(), z2.data(), ones_f.data(), ones_f.data(), gu.data(), rows, F, F, stream);
            float *zr = rr_buf(z.data(), rows);
            run_linear(w.ldown, gu.data(), F, rows, zr);
            residual_reduce(zr, rows);
        }
        if (std::getenv("QWEN_RT_DBG")) {
            HIP_CHECK(hipStreamSynchronize(stream));  // download() is a null-stream copy
            std::vector<uint16_t> h = state.download();
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                float v;
                uint16_t b = h[size_t(i)];
                uint32_t u = uint32_t(b) << 16;
                std::memcpy(&v, &u, 4);
                ss += double(v) * v;
            }
            std::fprintf(stderr, "L%d mlp norm %.4f\n", L, std::sqrt(ss / H));
        }
        if (const char *dbg = std::getenv("QWEN_DUMP_LAYERS")) {
            static int dump_rows = 0;
            if (!dump_rows) dump_rows = rows;
            if (rows == dump_rows && pos0 == 0) {
                HIP_CHECK(hipStreamSynchronize(stream));  // download() is a null-stream copy
                std::vector<uint16_t> h = state.download();
                std::ofstream f(std::string(dbg) + "/rt_layer" + std::to_string(L) + ".bin",
                                std::ios::binary);
                f.write(reinterpret_cast<const char *>(h.data()),
                        std::streamsize(size_t(rows) * H * 2));
            }
        }
    }
    HIP_CHECK(hipEventRecord(done_ev, stream));
}

// Tensor parallel all-reduce of a row-parallel linear's output (2 ranks): copy into the round's
// send buffer, barrier (host: every rank has recorded its send event), pull the peer's buffer over
// P2P, add both partials in rank order. Send buffers alternate so a rank only waits for the peer's
// pull of the round before last.
void QwenStage::residual_reduce(const float *zpart, int rows, bool decode) {
    const int H = cfg.hidden;
    if (tp_size == 1) {
        q_moe_residual_add(state.data(), zpart, rows, H, stream);
        return;
    }
    // device-flag exchange through pinned host slots; prefill partials travel as bf16
    static const bool fp32_only = std::getenv("QWEN_AR_FP32") != nullptr;
    static const bool dec_bf16 = std::getenv("QWEN_AR_DECODE_BF16") != nullptr;
    const bool bf = !fp32_only && (decode ? dec_bf16 : (rows > 8 || prefill_big));
    const long n = long(rows) * H;
    float *part = const_cast<float *>(zpart);
    static const bool dma_push = std::getenv("QWEN_AR_DMA_PUSH") != nullptr;
    const int hslot = int(ar_hseq % unsigned(AR_SLOTS));
    if (dma_push && bf && !decode) {
        q_ar_round_bf16(part, zsend[1].data(), n, stream);
        HIP_CHECK(hipMemcpyAsync(ar_host + long(hslot) * ar_slot_bytes, zsend[1].data(), size_t(n) * 2,
                                 hipMemcpyDeviceToHost, stream));
        q_ar_flag(ar_flags, ar_ctr.data(), AR_SLOTS, stream);
    } else {
        q_ar_push(part, n, ar_host, ar_slot_bytes, bf, ar_flags, ar_ctr.data(), ar_ctr.data() + 1, AR_SLOTS, stream);
    }
    static const bool zero_copy = std::getenv("QWEN_AR_ZC") != nullptr;
    if ((rows <= 8 && !prefill_big) || zero_copy || decode) {
        q_ar_pull_add(state.data(), part, tp_peer->ar_host, ar_slot_bytes, bf, n, tp_rank, tp_peer->ar_flags,
                      ar_ctr.data(), ar_ctr.data() + 2, AR_SLOTS, stream);
    } else {  // prefill: DMA the peer slot (copy engines cope with the cross-socket latency far better)
        q_ar_wait(tp_peer->ar_flags, ar_ctr.data(), AR_SLOTS, stream);
        HIP_CHECK(hipMemcpyAsync(zrecv.data(), tp_peer->ar_host + long(hslot) * ar_slot_bytes, size_t(n) * (bf ? 2 : 4),
                                 hipMemcpyHostToDevice, stream));
        q_ar_add_local(state.data(), part, zrecv.data(), bf, n, tp_rank, ar_ctr.data(), ar_ctr.data() + 2, stream);
    }
    ++ar_hseq;

}

void QwenStage::reset() {
    reset_slot(0);
    HIP_CHECK(hipStreamSynchronize(stream));
}

// ------------------------------------------------------------------ batched decode (serving)
void QwenStage::alloc_batch(int bm) {
    bmax = bm;
    const int H = cfg.hidden;
    if (chunk < bm) {  // per-row scratch must cover a full decode bucket
        state.resize(size_t(bm) * H);
        h_norm.resize(size_t(bm) * H);
    }
    if (size_t(bm) * H > zsend[0].size()) zsend[0].resize(size_t(bm) * H);
    meta_d.resize(size_t(3) * bm);
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&meta_h), size_t(3) * bm * 4, hipHostMallocDefault));
    std::memset(meta_h, 0, size_t(3) * bm * 4);
    bq.resize(size_t(bm) * cfg.nq * cfg.hd);
    bG.resize(size_t(bm) * cfg.nq * cfg.hd);
    battn.resize(size_t(bm) * cfg.nq * cfg.hd);
    battn_scr.resize(q_attn_b_scratch(bm, cfg.nq, cfg.max_context));
    if (logits_d.size() < size_t(bm) * cfg.vocab) logits_d.resize(size_t(bm) * cfg.vocab);
    cand_v.resize(size_t(bm) * TOPK);
    cand_i.resize(size_t(bm) * TOPK);
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&cand_vh), size_t(bm) * TOPK * 4, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&cand_ih), size_t(bm) * TOPK * 4, hipHostMallocDefault));
}

void QwenStage::topk_rows(int nrows) {
    q_topk(logits_d.data(), nrows, cfg.vocab, TOPK, voff, cand_v.data(), cand_i.data(), TOPK, stream);
    HIP_CHECK(hipMemcpyAsync(cand_vh, cand_v.data(), size_t(nrows) * TOPK * 4, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(cand_ih, cand_i.data(), size_t(nrows) * TOPK * 4, hipMemcpyDeviceToHost, stream));
}

// One decode step for rows [0, B) of meta (B <= bmax): fp8 W8A8 decode GEMMs, paged attention,
// per-slot DeltaNet, fp32 zero-copy all-reduce, fp8 lm_head + top-k candidates -> pinned host.
void QwenStage::decode_enqueue(int B) {
    const QwenRuntimeConfig &c = cfg;
    const int H = c.hidden, A = c.nq * c.hd, V = c.value_dim(), F = c.inter;
    HIP_CHECK(hipMemcpyAsync(meta_d.data(), meta_h, size_t(3) * bmax * 4, hipMemcpyHostToDevice, stream));
    q_embed_gather(embed.data(), meta_d.data(), state.data(), B, H, stream);
    float *zr = zsend[0].data();
    for (int L = l_begin; L < l_end; ++L) {
        LayerW &w = layers[size_t(L)];
        if (!w.fused) throw std::runtime_error("batched decode needs fp8 (fused) linears");
        q_norm_quant(state.data(), B, H, w.ln_in.data(), c.eps, h_norm.data(), xq8.data(), xqs.data(), stream);
        if (w.full) {
            const int Wf = c.nq * 2 * c.hd + 2 * c.nkv * c.hd;
            run_linear_q(w.f_qkv, B, z.data());
            q_rope_b(z.data(), Wf, bq.data(), bG.data(), w.kc.data(), w.vc.data(), meta_d.data(), bmax, B, ptab_d.data(),
                     maxpages, c.nq, c.nkv, c.hd, c.rot_dim, c.theta, w.q_norm.data(), w.k_norm.data(), c.eps, stream);
            q_attn_b(bq.data(), w.kc.data(), w.vc.data(), bG.data(), battn.data(), meta_d.data(), bmax, B, ptab_d.data(),
                     maxpages, c.nq, c.nkv, c.max_context, 1.0f / std::sqrt(float(c.hd)), battn_scr.data(), stream);
            run_linear(w.lo, battn.data(), A, B, zr);
        } else {
            const int Wq = c.conv_dim(), nv = c.lin_nv, Wz = Wq + V + 2 * nv;
            run_linear_q(w.f_qkvz, B, z.data(), Wz);
            q_bf16_gemv_any(w.w_ba.data(), 2 * nv, H, h_norm.data(), H, B, z.data() + Wq + V, Wz, stream);
            q_delta_b(z.data(), w.conv_w.data(), w.dn_norm.data(), w.a_dt.data(), w.ring.data(), w.dstate.data(),
                      o_in.data(), meta_d.data(), bmax, B, Wq, V, c.lin_nk, nv, c.conv_k, c.eps, stream);
            run_linear(w.l_out, o_in.data(), V, B, zr);
        }
        residual_reduce(zr, B, true);
        q_norm_quant(state.data(), B, H, w.ln_post.data(), c.eps, h_norm.data(), xq8.data(), xqs.data(), stream);
        run_linear_q(w.f_gu, B, z.data());
        q_gateup(z.data(), z.data() + F, ones_f.data(), ones_f.data(), gu.data(), B, F, 2 * F, stream);
        run_linear(w.ldown, gu.data(), F, B, zr);
        residual_reduce(zr, B, true);
    }
    q_zc_norm(state.data(), B, H, final_norm.data(), c.eps, stream);
    if (lm8.fp8)
        run_linear(lm8, state.data(), H, B, logits_d.data());
    else
        q_bf16_gemv_any(lm_head_w.size() ? lm_head_w.data() : embed.data(), c.vocab, H, state.data(), H, B,
                        logits_d.data(), c.vocab, stream);
    topk_rows(B);
}

void QwenStage::decode_launch(int B) {
    static const bool no_graph = std::getenv("QWEN_NO_GRAPH") != nullptr;
    if (no_graph) {
        decode_enqueue(B);
        return;
    }
    auto it = graphs.find(B);
    if (it == graphs.end()) {
        const unsigned h0 = ar_hseq;
        hipGraph_t g;
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        decode_enqueue(B);
        HIP_CHECK(hipStreamEndCapture(stream, &g));
        Graph gr;
        HIP_CHECK(hipGraphInstantiate(&gr.exec, g, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(g));
        gr.exchanges = ar_hseq - h0;
        ar_hseq = h0;  // capture did not execute the exchanges
        it = graphs.emplace(B, gr).first;
    }
    HIP_CHECK(hipGraphLaunch(it->second.exec, stream));
    ar_hseq += it->second.exchanges;
}

// ------------------------------------------------------------------ pipeline (layer split over GPUs)
struct QwenModel::Impl {
    QwenRuntimeConfig cfg;
    int chunk = 256;
    std::vector<std::unique_ptr<QwenStage>> st;
    std::vector<float> logits_h, logits_all;  // logits_all: [<=16][vocab] of the last lm_head_rows
    int pos = 0;  // legacy single-sequence API: position of slot 0
    bool tp = false;
    std::unique_ptr<std::barrier<>> bar;
    QwenStage &tail() { return *st.back(); }
    // ---- serving pools (host side; identical on every stage)
    int n_slots = 1, maxpages = 0, max_seqs = 0;
    std::vector<int> seq_len;                 // per slot
    std::vector<std::vector<int>> seq_pages;  // per slot
    std::vector<int> free_pages;
    std::vector<int> ptab_h;                  // [n_slots][maxpages]
    int dummy_page = 0;
    void run_chunk(const int *tokens, int rows, int slot = 0, int pos0 = -1) {
        if (pos0 < 0) pos0 = pos;
        if (!tp) {
            for (auto &s : st) {
                HIP_CHECK(hipSetDevice(s->device));
                s->forward_chunk(tokens, rows, slot, pos0);
            }
        } else {  // one host thread per rank (ranks meet at the all-reduce barriers)
            std::vector<std::thread> th;
            std::exception_ptr err;
            std::mutex mu;
            for (size_t i = 1; i < st.size(); ++i)
                th.emplace_back([&, i]() {
                    try {
                        HIP_CHECK(hipSetDevice(st[i]->device));
                        st[i]->forward_chunk(tokens, rows, slot, pos0);
                    } catch (...) {
                        std::lock_guard<std::mutex> g(mu);
                        err = std::current_exception();
                    }
                });
            try {
                HIP_CHECK(hipSetDevice(st[0]->device));
                st[0]->forward_chunk(tokens, rows, slot, pos0);
            } catch (...) {
                std::lock_guard<std::mutex> g(mu);
                err = std::current_exception();
            }
            for (auto &t : th) t.join();
            if (err) std::rethrow_exception(err);
        }
        if (slot == 0 && max_seqs == 0) pos += rows;
    }
    // page table entry -> every stage's device table
    void set_page(int slot, int idx, int page) {
        ptab_h[size_t(slot) * maxpages + idx] = page;
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            HIP_CHECK(hipMemcpy(s->ptab_d.data() + size_t(slot) * maxpages + idx, &page, 4, hipMemcpyHostToDevice));
        }
    }
    bool ensure_pages(int slot, int len) {  // pages for positions [0, len)
        const int need = (len + QwenStage::PAGE - 1) / QwenStage::PAGE;
        if (need > maxpages) return false;
        auto &pg = seq_pages[size_t(slot)];
        if (need - int(pg.size()) > int(free_pages.size())) return false;
        while (int(pg.size()) < need) {
            const int p = free_pages.back();
            free_pages.pop_back();
            set_page(slot, int(pg.size()), p);
            pg.push_back(p);
        }
        return true;
    }
    void setup_pools(int seqs, long kv_tokens, double reserve_gb) {
        max_seqs = seqs;
        const int mp = (cfg.max_context + QwenStage::PAGE - 1) / QwenStage::PAGE;
        int pages = mp;
        n_slots = 1;
        if (seqs > 0) {
            n_slots = seqs + 1;  // + dummy slot for bucket padding rows
            for (auto &s : st) {
                HIP_CHECK(hipSetDevice(s->device));
                s->alloc_batch(std::min(64, seqs));
                s->prefill_big = true;
            }
            size_t minfree = SIZE_MAX;
            for (auto &s : st) {
                HIP_CHECK(hipSetDevice(s->device));
                size_t fr = 0, tot = 0;
                HIP_CHECK(hipMemGetInfo(&fr, &tot));
                minfree = std::min(minfree, fr);
            }
            const size_t reserve = size_t(reserve_gb * 1e9);
            const size_t sb = st[0]->slot_bytes(), pb = st[0]->page_bytes();
            const size_t need_slots = sb * size_t(n_slots);
            if (minfree < reserve + need_slots + pb * 64)
                throw std::runtime_error("not enough GPU memory for " + std::to_string(seqs) + " sequence slots");
            long avail_pages = long((minfree - reserve - need_slots) / pb);
            if (kv_tokens > 0) avail_pages = std::min(avail_pages, long((kv_tokens + QwenStage::PAGE - 1) / QwenStage::PAGE) + 1);
            pages = int(std::min<long>(avail_pages, long(mp) * seqs + 1));
        }
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            s->alloc_pools(n_slots, pages, mp);
        }
        maxpages = mp;
        ptab_h.assign(size_t(n_slots) * mp, 0);
        seq_len.assign(size_t(n_slots), 0);
        seq_pages.assign(size_t(n_slots), {});
        free_pages.clear();
        if (seqs == 0) {  // legacy: slot 0 owns every page in order
            for (int i = 0; i < mp; ++i) {
                ptab_h[size_t(i)] = i;
                seq_pages[0].push_back(i);
            }
            for (auto &s : st) {
                HIP_CHECK(hipSetDevice(s->device));
                HIP_CHECK(hipMemcpy(s->ptab_d.data(), ptab_h.data(), size_t(mp) * 4, hipMemcpyHostToDevice));
            }
        } else {
            dummy_page = pages - 1;  // padding rows write their K/V here
            for (int p = pages - 2; p >= 0; --p) free_pages.push_back(p);
            set_page(n_slots - 1, 0, dummy_page);
        }
    }
    // ---- serving API
    void seq_start(int slot) {
        seq_free(slot);
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            s->reset_slot(slot);
        }
    }
    void seq_free(int slot) {
        for (int p : seq_pages[size_t(slot)]) free_pages.push_back(p);
        seq_pages[size_t(slot)].clear();
        seq_len[size_t(slot)] = 0;
    }
    void seq_fork(int src, int dst) {
        seq_start(dst);
        const int len = seq_len[size_t(src)];
        if (!ensure_pages(dst, len)) throw std::runtime_error("seq_fork: out of KV pages");
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            s->copy_slot(src, dst, seq_pages[size_t(src)], seq_pages[size_t(dst)]);
        }
        seq_len[size_t(dst)] = len;
    }
    // merged top-K candidates of `rows` rows (sorted by logit desc, id asc)
    void merge_cands(int rows, QwenCand *out) {
        const int K = QwenStage::TOPK;
        std::vector<QwenCand> tmp;
        for (int r = 0; r < rows; ++r) {
            tmp.clear();
            for (auto &s : st)
                for (int k = 0; k < K; ++k) tmp.push_back({s->cand_vh[size_t(r) * K + k], s->cand_ih[size_t(r) * K + k]});
            std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end(), [](const QwenCand &a, const QwenCand &b) {
                return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
            });
            std::copy(tmp.begin(), tmp.begin() + K, out + size_t(r) * K);
        }
    }
    bool seq_prefill(int slot, const int *tokens, size_t n, QwenCand *cand) {
        const int p0 = seq_len[size_t(slot)];
        if (!ensure_pages(slot, p0 + int(n))) return false;
        for (size_t i = 0; i < n; i += size_t(chunk)) {
            const int rows = int(std::min<size_t>(size_t(chunk), n - i));
            run_chunk(tokens + i, rows, slot, p0 + int(i));
            if (cand && i + size_t(rows) == n) {  // candidates of the last token
                for (auto &s : st) {
                    HIP_CHECK(hipSetDevice(s->device));
                    s->lm_head_enqueue(rows - 1, 1);
                    s->topk_rows(1);
                }
                for (auto &s : st) {
                    HIP_CHECK(hipSetDevice(s->device));
                    HIP_CHECK(hipStreamSynchronize(s->stream));
                }
            }
        }
        seq_len[size_t(slot)] = p0 + int(n);
        if (cand) merge_cands(1, cand);
        return true;
    }
    // several sequences' prompt pieces in one forward pass (rows <= chunk); cand: [parts][K] for the
    // parts with want (packed in order); false: out of KV pages (nothing run)
    bool prefill_multi(const std::vector<QwenModel::PrefillPart> &parts, QwenCand *cand) {
        std::vector<int> toks;
        std::vector<QwenStage::Seg> sg;
        std::vector<int> want_rows;
        for (auto &p : parts) {
            const int len = seq_len[size_t(p.slot)];
            if (!ensure_pages(p.slot, len + int(p.n))) return false;
            sg.push_back({p.slot, len, int(toks.size()), int(p.n)});
            toks.insert(toks.end(), p.tokens, p.tokens + p.n);
            if (p.want) want_rows.push_back(int(toks.size()) - 1);
        }
        if (int(toks.size()) > chunk) throw std::runtime_error("prefill_multi: more rows than the chunk");
        for (auto &s : st) s->segs = sg;
        try {
            run_chunk(toks.data(), int(toks.size()), sg[0].slot, sg[0].pos0);
        } catch (...) {
            for (auto &s : st) s->segs.clear();
            throw;
        }
        for (auto &s : st) s->segs.clear();
        if (!want_rows.empty()) {
            for (auto &s : st) {
                HIP_CHECK(hipSetDevice(s->device));
                s->cand_rows(want_rows);
            }
        }
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            HIP_CHECK(hipStreamSynchronize(s->stream));
        }
        for (auto &p : parts) seq_len[size_t(p.slot)] += int(p.n);
        if (!want_rows.empty() && cand) merge_cands(int(want_rows.size()), cand);
        return true;
    }
    // one token for each of n sequences; false (nothing run) if a sequence cannot get a KV page
    bool decode_batch(const int *slots, const int *tokens, int n, QwenCand *cand) {
        const int bm = st[0]->bmax;
        if (n < 1 || n > bm) throw std::runtime_error("decode_batch: 1..bmax sequences");
        for (int i = 0; i < n; ++i)
            if (seq_len[size_t(slots[i])] + 1 > cfg.max_context || !ensure_pages(slots[i], seq_len[size_t(slots[i])] + 1))
                return false;
        static const int buckets[] = {1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64};
        int B = bm;
        for (int b : buckets)
            if (b >= n) {
                B = std::min(b, bm);
                break;
            }
        for (auto &s : st) {
            int *m = s->meta_h;
            for (int i = 0; i < B; ++i) {
                const bool real = i < n;
                m[i] = real ? tokens[i] : 0;
                m[bm + i] = real ? slots[i] : n_slots - 1;
                m[2 * bm + i] = real ? seq_len[size_t(slots[i])] : 0;
            }
        }
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            s->decode_launch(B);
        }
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            HIP_CHECK(hipStreamSynchronize(s->stream));
        }
        for (int i = 0; i < n; ++i) seq_len[size_t(slots[i])] += 1;
        if (cand) merge_cands(n, cand);
        return true;
    }
    void lm_head_rows(int row0, int nrows) {
        if (!tp) {
            HIP_CHECK(hipSetDevice(tail().device));
            tail().lm_head_rows(row0, nrows);
            std::memcpy(logits_all.data(), tail().logits_all.data(), size_t(std::min(nrows, 16)) * cfg.vocab * 4);
            return;
        }
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            s->lm_head_enqueue(row0, nrows);
        }
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            HIP_CHECK(hipStreamSynchronize(s->stream));
            const int vl = s->cfg.vocab;
            for (int r = 0; r < std::min(nrows, 16); ++r)
                std::memcpy(logits_all.data() + size_t(r) * cfg.vocab + s->voff, s->logits_all.data() + size_t(r) * vl,
                            size_t(vl) * 4);
        }
    }
    void reset() {
        for (auto &s : st) {
            HIP_CHECK(hipSetDevice(s->device));
            s->reset();
        }
        pos = 0;
    }
    void prefill(const int *tokens, size_t n, float *last_logits) {
        for (size_t i = 0; i < n; i += size_t(chunk)) {
            int rows = int(std::min<size_t>(chunk, n - i));
            run_chunk(tokens + i, rows);
            if (last_logits && i + size_t(rows) == n) {
                lm_head_rows(rows - 1, 1);
                std::memcpy(last_logits, logits_all.data(), size_t(cfg.vocab) * 4);
            }
        }
    }
    void decode(int token, float *logits) {
        static const bool timing = std::getenv("QWEN_TIMING") != nullptr;
        auto t0 = std::chrono::steady_clock::now();
        run_chunk(&token, 1);
        if (timing) {
            auto t1 = std::chrono::steady_clock::now();
            for (auto &s : st) {
                HIP_CHECK(hipSetDevice(s->device));
                HIP_CHECK(hipStreamSynchronize(s->stream));
            }
            auto t2 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "decode: host enqueue %.2f ms, gpu tail %.2f ms\n",
                         std::chrono::duration<double, std::milli>(t1 - t0).count(),
                         std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        if (logits) {
            lm_head_rows(0, 1);
            std::memcpy(logits, logits_all.data(), size_t(cfg.vocab) * 4);
        }
    }
    int argmax() const {
        int next = 0;
        for (size_t i = 0; i < size_t(cfg.vocab); ++i)
            if (logits_h[i] > logits_h[size_t(next)]) next = int(i);
        return next;
    }
    std::vector<int> generate(const int *prompt, size_t n_prompt, size_t n_new) {
        reset();
        std::vector<int> out;
        prefill(prompt, n_prompt, logits_h.data());
        int next = argmax();
        for (size_t g = 0; g < n_new; ++g) {
            out.push_back(next);
            decode(next, logits_h.data());
            next = argmax();
        }
        return out;
    }
    double perplexity(const int *tokens, size_t n) {
        if (n < 2) throw std::runtime_error("perplexity needs at least 2 tokens");
        reset();
        double nll = 0;
        long count = 0;
        // chunk [i, i+rows): logits of row r (position i+r) predict token at i+r+1
        for (size_t i = 0; i + 1 < n; i += size_t(chunk)) {
            int rows = int(std::min<size_t>(size_t(chunk), n - i));
            run_chunk(tokens + i, rows);
            int nrows_log = int(std::min<size_t>(size_t(rows), n - 1 - i));  // rows with a target
            for (int r0 = 0; r0 < nrows_log; r0 += 16) {
                int nr = std::min(16, nrows_log - r0);
                lm_head_rows(r0, nr);
                for (int r = 0; r < nr; ++r) {
                    size_t predict_at = i + size_t(r0 + r) + 1;
                    const float *row = logits_all.data() + size_t(r) * cfg.vocab;
                    float mx = -INFINITY;
                    for (long v = 0; v < cfg.vocab; ++v) mx = std::max(mx, row[size_t(v)]);
                    double lse = 0;
                    for (long v = 0; v < cfg.vocab; ++v) lse += std::exp(double(row[size_t(v)]) - mx);
                    lse = std::log(lse) + mx;
                    nll += lse - row[size_t(tokens[predict_at])];
                    count++;
                }
            }
        }
        return std::exp(nll / double(count));
    }
};

// ------------------------------------------------------------------ public
QwenModel::QwenModel(const std::filesystem::path &dir, int device, int chunk, int max_context)
    : QwenModel(dir, std::vector<int>{device}, chunk, max_context) {}

QwenModel::QwenModel(const std::filesystem::path &dir, const std::vector<int> &devices, int chunk,
                     int max_context, bool tensor_parallel)
    : QwenModel(dir, devices, chunk, max_context, tensor_parallel, QwenServeOptions{}) {}

QwenModel::QwenModel(const std::filesystem::path &dir, const std::vector<int> &devices, int chunk,
                     int max_context, bool tensor_parallel, const QwenServeOptions &opt)
    : impl_(std::make_unique<Impl>()) {
    if (devices.empty()) throw std::runtime_error("QwenModel: no devices");
    impl_->chunk = std::max(opt.max_seqs > 0 ? 64 : 1, chunk);
    impl_->cfg = QwenRuntimeConfig::read(dir);
    impl_->cfg.max_context = max_context;
    const int n = int(devices.size()), NL = impl_->cfg.layers;
    // even layer split; QWEN_SPLIT=a,b,... (first layer of each later stage) overrides
    std::vector<int> bounds;
    for (int s = 0; s <= n; ++s) bounds.push_back(NL * s / n);
    if (const char *sp = std::getenv("QWEN_SPLIT")) {
        std::string str(sp);
        size_t p = 0;
        for (int s = 1; s < n; ++s) {
            size_t q = str.find(',', p);
            bounds[size_t(s)] = std::stoi(str.substr(p, q == std::string::npos ? std::string::npos : q - p));
            p = q == std::string::npos ? str.size() : q + 1;
        }
    }
    impl_->logits_all.resize(size_t(16) * impl_->cfg.vocab);
    impl_->tp = tensor_parallel && n > 1;
    if (impl_->tp) {
        const QwenRuntimeConfig &G = impl_->cfg;
        if (n != 2) throw std::runtime_error("tensor parallel: exactly 2 GPUs supported");
        if (G.tied || G.num_experts > 0 || G.nkv % n || G.nq % n || G.lin_nk % n || G.lin_nv % n ||
            G.inter % n || G.vocab % n)
            throw std::runtime_error("tensor parallel: model dims not divisible / tied / MoE");
        impl_->bar = std::make_unique<std::barrier<>>(n);
        for (int s = 0; s < n; ++s) {
            HIP_CHECK(hipSetDevice(devices[size_t(s)]));
            auto stg = std::make_unique<QwenStage>();
            stg->device = devices[size_t(s)];
            stg->chunk = impl_->chunk;
            stg->l_begin = 0;
            stg->l_end = NL;
            stg->tp_rank = s;
            stg->tp_size = n;
            stg->tp_bar = impl_->bar.get();
            stg->cfg = G;
            stg->cfg.nq /= n;
            stg->cfg.nkv /= n;
            stg->cfg.lin_nk /= n;
            stg->cfg.lin_nv /= n;
            stg->cfg.inter /= n;
            stg->cfg.vocab /= n;
            stg->voff = s * stg->cfg.vocab;
            HIP_CHECK(hipStreamCreateWithFlags(&stg->stream, hipStreamNonBlocking));
            HIP_CHECK(hipEventCreateWithFlags(&stg->done_ev, hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&stg->took_ev, hipEventDisableTiming));
            stg->load(dir);
            impl_->st.push_back(std::move(stg));
        }
        impl_->st[0]->tp_peer = impl_->st[1].get();
        impl_->st[1]->tp_peer = impl_->st[0].get();
        for (int a = 0; a < n; ++a) {
            HIP_CHECK(hipSetDevice(devices[size_t(1 - a)]));
            impl_->st[size_t(a)]->alloc_ar_host();
        }
        for (int a = 0; a < n; ++a) {
            HIP_CHECK(hipSetDevice(devices[size_t(a)]));
            int can = 0;
            (void)hipDeviceCanAccessPeer(&can, devices[size_t(a)], devices[size_t(1 - a)]);
            if (can) {
                hipError_t e = hipDeviceEnablePeerAccess(devices[size_t(1 - a)], 0);
                impl_->st[size_t(a)]->p2p = e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled;
            }
            (void)hipGetLastError();
            if (std::getenv("QWEN_NO_P2P")) impl_->st[size_t(a)]->p2p = false;
            HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&impl_->st[size_t(a)]->tok_pinned), 64));
            HIP_CHECK(hipDeviceSynchronize());
        }
        impl_->logits_h.resize(size_t(impl_->cfg.vocab));
        impl_->setup_pools(opt.max_seqs, opt.kv_tokens, opt.reserve_gb);
        impl_->reset();
        return;
    }
    if (opt.max_seqs > 0 && n > 1) throw std::runtime_error("serving mode: single GPU or tensor parallel only");
    for (int s = 0; s < n; ++s) {
        HIP_CHECK(hipSetDevice(devices[size_t(s)]));
        auto stg = std::make_unique<QwenStage>();
        stg->device = devices[size_t(s)];
        stg->chunk = impl_->chunk;
        stg->l_begin = bounds[size_t(s)];
        stg->l_end = bounds[size_t(s) + 1];
        stg->first = s == 0;
        stg->last = s == n - 1;
        stg->prev = s ? impl_->st.back().get() : nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stg->stream, hipStreamNonBlocking));
        HIP_CHECK(hipEventCreateWithFlags(&stg->done_ev, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&stg->took_ev, hipEventDisableTiming));
        if (s) impl_->st.back()->next = stg.get();
        stg->cfg = impl_->cfg;
        stg->load(dir);
        impl_->st.push_back(std::move(stg));
        if (s) {  // peer access for the hidden-state hand-off (fall back to staged copies if absent)
            int can = 0;
            (void)hipDeviceCanAccessPeer(&can, devices[size_t(s)], devices[size_t(s) - 1]);
            if (can) {
                hipError_t e = hipDeviceEnablePeerAccess(devices[size_t(s) - 1], 0);
                impl_->st.back()->p2p = e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled;
            }
            (void)hipGetLastError();
            if (std::getenv("QWEN_NO_P2P")) impl_->st.back()->p2p = false;
        }
    }
    HIP_CHECK(hipSetDevice(devices[0]));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&impl_->st[0]->tok_pinned), 64));
    impl_->logits_h.resize(size_t(impl_->cfg.vocab));
    impl_->setup_pools(opt.max_seqs, opt.kv_tokens, opt.reserve_gb);
    impl_->reset();
}
QwenModel::~QwenModel() {
    for (auto &s : impl_->st) {
        (void)hipSetDevice(s->device);
        (void)hipStreamDestroy(s->stream);
    }
}
const QwenRuntimeConfig &QwenModel::config() const { return impl_->cfg; }
void QwenModel::reset() { impl_->reset(); }
int QwenModel::position() const { return impl_->pos; }
void QwenModel::prefill(const int *tokens, size_t n, float *last_logits) {
    impl_->prefill(tokens, n, last_logits);
}
void QwenModel::decode(int token, float *logits) { impl_->decode(token, logits); }
std::vector<int> QwenModel::generate(const int *prompt, size_t n_prompt, size_t n_new) {
    return impl_->generate(prompt, n_prompt, n_new);
}
double QwenModel::perplexity(const int *tokens, size_t n) { return impl_->perplexity(tokens, n); }
int QwenModel::max_seqs() const { return impl_->max_seqs; }
long QwenModel::kv_pages_total() const { return impl_->st[0]->n_pages - 1; }
long QwenModel::kv_pages_free() const { return long(impl_->free_pages.size()); }
int QwenModel::page_tokens() const { return QwenStage::PAGE; }
int QwenModel::topk() const { return QwenStage::TOPK; }
int QwenModel::seq_len(int slot) const { return impl_->seq_len[size_t(slot)]; }
void QwenModel::seq_start(int slot) { impl_->seq_start(slot); }
void QwenModel::seq_free(int slot) { impl_->seq_free(slot); }
bool QwenModel::seq_prefill(int slot, const int *tokens, size_t n, QwenCand *cand) {
    return impl_->seq_prefill(slot, tokens, n, cand);
}
bool QwenModel::decode_batch(const int *slots, const int *tokens, int n, QwenCand *cand) {
    return impl_->decode_batch(slots, tokens, n, cand);
}
void QwenModel::seq_fork(int src, int dst) { impl_->seq_fork(src, dst); }
bool QwenModel::prefill_multi(const std::vector<PrefillPart> &parts, QwenCand *cand) {
    return impl_->prefill_multi(parts, cand);
}
int QwenModel::chunk() const { return impl_->chunk; }

}  // namespace behemoth

