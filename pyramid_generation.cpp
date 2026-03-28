// Pyramid KV Prefill — Generation Benchmark for RTX 4090
//
// Implements pyramid-guided visual token dropping for Qwen2.5-VL video inference.
// Uses attention weights from scout layers (0-1) during prefill to score visual
// tokens, then evicts low-importance tokens from KV cache using a cosine retention
// schedule (100% at layer 0 → 20% at deepest layer).
//
// Architecture: Two-pass prefill with GPU-accelerated scoring
//   Pass 1: Normal prefill with cb_eval capturing attention from scout layers
//   Pass 2: Post-prefill KV eviction using pyramid importance scores
//   Generation: Autoregressive decoding with reduced KV cache
//
// Usage:
//   llama-pyramid-gen -m <model.gguf> --mmproj <mmproj.gguf> [options]

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "pyramid_prefill.h"

#ifdef HAVE_CUDA_PYRAMID
#include "visual_pyramid.cuh"
#include <cuda_runtime.h>
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>
#include <cinttypes>
#include <map>
#include <set>
#include <random>

using json = nlohmann::ordered_json;

// ============================================================================
// Section 1: Parameters
// ============================================================================

struct gen_params {
    std::string sintel_dir  = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel";
    std::string video_name  = "alley_1";
    int         n_frames    = 8;
    std::string output_json;
    std::string prompt      = "Describe what is happening in this video sequence in detail.";

    // Pyramid config
    pyramid_config pyramid;

    // Benchmark
    bool baseline           = false;   // run without pyramid (for comparison)
    bool no_prellm          = false;   // disable pre-LLM pruning (attention-only eviction)
    int  n_predict          = 256;     // max tokens to generate
    int  scout_layer        = 1;       // which layer's attention to capture (0-indexed)
    bool ignore_eos         = false;   // force generating exactly n_predict tokens

    // Adaptive frame selection (complexity-weighted probabilistic sampling)
    bool  adaptive_frames       = false;
    int   base_frames           = 8;      // base frame count for complexity step function
    int   min_selected_frames   = 3;      // floor for adaptive count
    int   max_selected_frames   = 24;     // ceiling for adaptive count
    float fps                   = 4.0f;
    float motion_threshold      = 2.0f;
    float motion_percentile     = 0.95f;
    float selection_percentile  = 70.0f;  // percentile for importance weight threshold
    float complexity_floor      = 0.05f;  // remove frames with per-frame complexity below this

    // Spatial patch encoding with embedding cache
    bool  spatial_cache         = false;
    int   patch_grid            = 4;
    float spatial_threshold     = 0.05f;
    float skip_vit_threshold    = 0.05f;  // active ratio below this → skip ViT entirely
    int   crop_padding          = 16;
    int   min_crop_dim          = 112;
    float patch_reuse_motion    = 0.0f;
};

static std::vector<char *> filter_custom_args(int argc, char ** argv, gen_params & gp) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--sintel-dir" && i + 1 < argc) {
            gp.sintel_dir = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            gp.video_name = argv[++i];
        } else if (arg == "--n-frames" && i + 1 < argc) {
            gp.n_frames = std::stoi(argv[++i]);
        } else if (arg == "--output-json" && i + 1 < argc) {
            gp.output_json = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            gp.prompt = argv[++i];
        } else if (arg == "--baseline") {
            gp.baseline = true;
        } else if (arg == "--ignore-eos") {
            gp.ignore_eos = true;
        } else if (arg == "--n-predict" && i + 1 < argc) {
            gp.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--scout-layer" && i + 1 < argc) {
            gp.scout_layer = std::stoi(argv[++i]);
        } else if (arg == "--pyramid-clie-layer" && i + 1 < argc) {
            gp.pyramid.clie_layer = std::stoi(argv[++i]);
        } else if (arg == "--pyramid-pvc-min" && i + 1 < argc) {
            gp.pyramid.pvc_min_ratio = std::stof(argv[++i]);
        } else if (arg == "--pyramid-temporal-thresh" && i + 1 < argc) {
            gp.pyramid.temporal_sim_thresh = std::stof(argv[++i]);
        } else if (arg == "--no-pyramid-protect-frame0") {
            gp.pyramid.first_frame_protect = false;
        } else if (arg == "--no-pyramid") {
            gp.pyramid.enabled = false;
        } else if (arg == "--no-prellm") {
            gp.no_prellm = true;
        } else if (arg == "--adaptive") {
            gp.adaptive_frames = true;
        } else if (arg == "--base-frames" && i + 1 < argc) {
            gp.base_frames = std::stoi(argv[++i]);
        } else if (arg == "--min-selected-frames" && i + 1 < argc) {
            gp.min_selected_frames = std::stoi(argv[++i]);
        } else if (arg == "--max-selected-frames" && i + 1 < argc) {
            gp.max_selected_frames = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            gp.fps = std::stof(argv[++i]);
        } else if (arg == "--motion-threshold" && i + 1 < argc) {
            gp.motion_threshold = std::stof(argv[++i]);
        } else if (arg == "--motion-percentile" && i + 1 < argc) {
            gp.motion_percentile = std::stof(argv[++i]);
        } else if (arg == "--selection-percentile" && i + 1 < argc) {
            gp.selection_percentile = std::stof(argv[++i]);
        } else if (arg == "--complexity-floor" && i + 1 < argc) {
            gp.complexity_floor = std::stof(argv[++i]);
        } else if (arg == "--spatial-cache") {
            gp.spatial_cache = true;
        } else if (arg == "--patch-grid" && i + 1 < argc) {
            gp.patch_grid = std::stoi(argv[++i]);
        } else if (arg == "--spatial-threshold" && i + 1 < argc) {
            gp.spatial_threshold = std::stof(argv[++i]);
        } else if (arg == "--skip-vit-threshold" && i + 1 < argc) {
            gp.skip_vit_threshold = std::stof(argv[++i]);
        } else if (arg == "--crop-padding" && i + 1 < argc) {
            gp.crop_padding = std::stoi(argv[++i]);
        } else if (arg == "--min-crop-dim" && i + 1 < argc) {
            gp.min_crop_dim = std::stoi(argv[++i]);
        } else if (arg == "--patch-reuse-motion" && i + 1 < argc) {
            gp.patch_reuse_motion = std::stof(argv[++i]);
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

// ============================================================================
// Section 2: Visual token tracking
// ============================================================================

struct visual_token_info {
    int     frame_idx;
    int     patch_row;
    int     patch_col;
    llama_pos kv_pos;      // M-RoPE temporal position
    llama_pos ext_x;       // M-RoPE spatial X
    llama_pos ext_y;       // M-RoPE spatial Y
    int     kv_cell_idx;   // actual KV cache cell index
    float   importance;    // attention-based importance score
    bool    evicted;
};

struct visual_token_tracker {
    std::vector<visual_token_info> tokens;
    std::vector<std::pair<int,int>> frame_ranges; // (start_idx, count) per frame
    int n_frames       = 0;
    int tokens_per_frame = 0;

    void add_token(int frame_idx, int row, int col,
                   llama_pos pos, llama_pos ex, llama_pos ey, int cell_idx) {
        visual_token_info ti = {};
        ti.frame_idx   = frame_idx;
        ti.patch_row   = row;
        ti.patch_col   = col;
        ti.kv_pos      = pos;
        ti.ext_x       = ex;
        ti.ext_y       = ey;
        ti.kv_cell_idx = cell_idx;
        ti.importance  = 0.0f;
        ti.evicted     = false;
        tokens.push_back(ti);
    }

    int count_active() const {
        int c = 0;
        for (const auto & t : tokens) if (!t.evicted) c++;
        return c;
    }
};

// ============================================================================
// Section 3: Flash-compatible Qcur capture for three-stage scoring
//
// Instead of capturing kq_soft_max (requires flash attn OFF), we capture Qcur
// (query projections) from layers 7, 19, 27. Then compute Q@K^T on CPU using
// the K cache. This works with flash attention ENABLED.
// ============================================================================

struct multi_qcur_capture_state {
    // Single-layer capture: only capture from one layer (default: last layer, 27)
    // All layers produce identical scores (validated), so one Q@K^T pass suffices.
    int capture_layer = 27;

    struct qcur_data {
        std::vector<float> q;
        int n_embd_head = 0;
        int n_head      = 0;
        int n_q         = 0;
        bool captured   = false;
    };
    qcur_data captured;

    // Keep legacy 3-stage interface for compatibility
    static constexpr int N_STAGES = 3;
    int stage_layers[N_STAGES] = {7, 19, 27};
    qcur_data stages[N_STAGES];  // stages[2] aliases captured

    bool should_capture = false;
    bool disabled       = false;
};

static bool multi_qcur_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (multi_qcur_capture_state *)user_data;
    if (!state || !state->should_capture || state->disabled) return !ask;

    std::string name(t->name);
    std::string target = "Qcur-" + std::to_string(state->capture_layer);

    if (ask) {
        return name == target;
    }

    if (name != target) return true;

    // Qcur shape: [n_embd_head, n_head, n_tokens]
    // The tensor is named "Qcur" twice per layer (pre-RoPE and post-RoPE).
    // The LAST write wins (post-RoPE), which is the one we want.
    auto & sd = state->captured;
    sd.n_embd_head = (int)t->ne[0];
    sd.n_head      = (int)t->ne[1];
    sd.n_q         = (int)t->ne[2];
    sd.q.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, sd.q.data(), 0, ggml_nbytes(t));
    sd.captured = true;

    return true;
}

// ============================================================================
// Section 4: CPU Q@K^T computation (flash-attn compatible)
//
// Computes attention scores for visual tokens using captured Qcur and K cache.
// Returns per-visual-token importance scores.
// ============================================================================

static void compute_qk_attention_scores(
    const multi_qcur_capture_state::qcur_data & qstate,
    int layer_idx,
    llama_memory_t mem,
    const visual_token_tracker & tracker,
    int n_kv_cells_used,
    int n_head_kv,
    std::vector<float> & out_scores)  // [n_visual_tokens]
{
    if (!qstate.captured || qstate.q.empty()) {
        printf("  Warning: Qcur not captured for layer %d\n", layer_idx);
        return;
    }

    const int d_k     = qstate.n_embd_head;
    const int n_head  = qstate.n_head;
    const int n_q     = qstate.n_q;
    const int n_kv    = n_kv_cells_used;
    const int gqa     = n_head / n_head_kv;

    // Read K cache from GPU
    const ggml_tensor * k_tensor = llama_memory_k_tensor(mem, layer_idx);
    if (!k_tensor) {
        printf("  Warning: K tensor null for layer %d\n", layer_idx);
        return;
    }

    size_t row_bytes = ggml_row_size(k_tensor->type, k_tensor->ne[0]);
    std::vector<uint8_t> k_raw(n_kv * row_bytes);
    ggml_backend_tensor_get(k_tensor, k_raw.data(), 0, k_raw.size());

    // Convert K to f32
    int n_embd_k_gqa = n_head_kv * d_k;
    std::vector<float> k_f32(n_kv * n_embd_k_gqa);
    if (k_tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)k_raw.data();
        for (size_t i = 0; i < k_f32.size(); i++) k_f32[i] = ggml_fp16_to_fp32(src[i]);
    } else if (k_tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t * src = (ggml_bf16_t *)k_raw.data();
        for (size_t i = 0; i < k_f32.size(); i++) k_f32[i] = ggml_bf16_to_fp32(src[i]);
    } else if (k_tensor->type == GGML_TYPE_F32) {
        memcpy(k_f32.data(), k_raw.data(), k_f32.size() * sizeof(float));
    } else {
        printf("  Warning: unsupported K type %d\n", (int)k_tensor->type);
        return;
    }

    float scale = 1.0f / sqrtf((float)d_k);

    // Compute total attention per KV position: sum over all queries and heads of softmax(Q@K^T)
    std::vector<double> kv_attn(n_kv, 0.0);
    std::vector<float> logits(n_kv);

    for (int qi = 0; qi < n_q; qi++) {
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / gqa;
            const float * q_ptr = qstate.q.data() + (qi * n_head + h) * d_k;

            for (int j = 0; j < n_kv; j++) {
                const float * k_ptr = k_f32.data() + j * n_embd_k_gqa + kv_h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) dot += q_ptr[d] * k_ptr[d];
                logits[j] = dot * scale;
            }

            // Softmax
            float max_val = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (int j = 0; j < n_kv; j++) {
                logits[j] = expf(logits[j] - max_val);
                sum_exp += logits[j];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < n_kv; j++) {
                kv_attn[j] += (double)(logits[j] * inv_sum);
            }
        }
    }

    // Normalize by head count
    for (int j = 0; j < n_kv; j++) kv_attn[j] /= (double)n_head;

    // Map to visual token scores
    out_scores.resize(tracker.tokens.size(), 0.0f);
    for (int ti = 0; ti < (int)tracker.tokens.size(); ti++) {
        int cell = tracker.tokens[ti].kv_cell_idx;
        if (cell >= 0 && cell < n_kv) {
            out_scores[ti] = (float)kv_attn[cell];
        }
    }
}

// ============================================================================
// Section 5: VLM context wrapper
// ============================================================================

struct pyramid_vlm_context {
    common_init_result_ptr llama_init;
    const llama_model * model  = nullptr;
    llama_context     * lctx   = nullptr;
    const llama_vocab * vocab  = nullptr;
    common_sampler    * smpl   = nullptr;
    llama_batch         batch;

    mtmd::context_ptr ctx_vision;
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;

    llama_pos n_past = 0;
    int       n_batch = 2048;

    pyramid_vlm_context(common_params & params)
        : llama_init(common_init_from_params(params)) {

        model = llama_init->model();
        lctx  = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl  = common_sampler_init(model, params.sampling);
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;

        if (!model || !lctx) {
            fprintf(stderr, "Failed to initialize model/context\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);

        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu         = params.mmproj_use_gpu;
        mparams.print_timings   = true;
        mparams.n_threads       = params.cpuparams.n_threads;
        mparams.warmup          = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        mparams.flash_attn_type = params.flash_attn_type;

        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            fprintf(stderr, "Failed to load vision model from %s\n", clip_path);
            exit(1);
        }

        printf("Model and vision context initialized\n");
    }

    ~pyramid_vlm_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }
};

// ============================================================================
// Section 6: Frame loading
// ============================================================================

static std::vector<std::string> load_frame_paths(
    const std::string & sintel_dir,
    const std::string & video_name,
    int n_frames,
    std::vector<int> & out_frame_numbers)
{
    std::string base = sintel_dir + "/training/final/" + video_name + "/";
    std::vector<std::string> paths;
    out_frame_numbers.clear();

    // Count total frames
    int total = 0;
    for (int i = 1; i <= 999; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "frame_%04d.png", i);
        std::string path = base + buf;
        FILE * f = fopen(path.c_str(), "rb");
        if (f) { fclose(f); total++; }
        else break;
    }

    if (total == 0) {
        fprintf(stderr, "No frames found in %s\n", base.c_str());
        return paths;
    }

    // Uniformly sample n_frames
    for (int i = 0; i < n_frames; i++) {
        int frame_num = 1 + (i * (total - 1)) / std::max(1, n_frames - 1);
        char buf[64];
        snprintf(buf, sizeof(buf), "frame_%04d.png", frame_num);
        paths.push_back(base + buf);
        out_frame_numbers.push_back(frame_num);
    }

    printf("Selected %d frames from %d total in %s\n", n_frames, total, video_name.c_str());
    for (size_t i = 0; i < paths.size(); i++) {
        printf("  [%d] %s\n", out_frame_numbers[i], paths[i].c_str());
    }

    return paths;
}

// ============================================================================
// Section 6b: Optical flow reader and per-token motion scoring
// ============================================================================

struct optical_flow {
    int width = 0, height = 0;
    std::vector<float> u, v;  // horizontal, vertical flow
};

static bool read_flo_file(const std::string & path, optical_flow & flow) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    float magic = 0.0f;
    if (fread(&magic, sizeof(float), 1, f) != 1 || magic != 202021.25f) { fclose(f); return false; }

    int32_t w = 0, h = 0;
    if (fread(&w, sizeof(int32_t), 1, f) != 1 || fread(&h, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

    flow.width = w; flow.height = h;
    int n = w * h;
    std::vector<float> raw(n * 2);
    if ((int)fread(raw.data(), sizeof(float), n * 2, f) != n * 2) { fclose(f); return false; }
    fclose(f);

    flow.u.resize(n); flow.v.resize(n);
    for (int i = 0; i < n; i++) { flow.u[i] = raw[2*i]; flow.v[i] = raw[2*i+1]; }
    return true;
}

// Compute per-ViT-token motion magnitude from optical flow.
// Maps each token's spatial patch to the corresponding flow region and averages the magnitude.
// Returns [chunk_nx * chunk_ny] motion scores.
static std::vector<float> compute_token_motion_scores(
    const optical_flow & flow, int chunk_nx, int chunk_ny)
{
    int n_tokens = chunk_nx * chunk_ny;
    std::vector<float> scores(n_tokens, 0.0f);

    if (flow.width == 0 || flow.height == 0) return scores;

    for (int ty = 0; ty < chunk_ny; ty++) {
        for (int tx = 0; tx < chunk_nx; tx++) {
            // Map token grid position to pixel region
            int px_start = tx * flow.width / chunk_nx;
            int px_end   = (tx + 1) * flow.width / chunk_nx;
            int py_start = ty * flow.height / chunk_ny;
            int py_end   = (ty + 1) * flow.height / chunk_ny;

            float sum_mag = 0.0f;
            int count = 0;
            for (int py = py_start; py < py_end; py++) {
                for (int px = px_start; px < px_end; px++) {
                    int idx = py * flow.width + px;
                    float mag = sqrtf(flow.u[idx]*flow.u[idx] + flow.v[idx]*flow.v[idx]);
                    sum_mag += mag;
                    count++;
                }
            }
            scores[ty * chunk_nx + tx] = (count > 0) ? sum_mag / count : 0.0f;
        }
    }
    return scores;
}

// Load optical flow files for the selected frame indices.
// flow[i] is the flow FROM frame i TO frame i+1. For the last frame, flow is empty.
static std::vector<optical_flow> load_optical_flows(
    const std::string & sintel_dir, const std::string & video_name,
    const std::vector<int> & frame_numbers)
{
    std::string base = sintel_dir + "/training/flow/" + video_name + "/";
    std::vector<optical_flow> flows(frame_numbers.size());

    for (size_t i = 0; i < frame_numbers.size(); i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "frame_%04d.flo", frame_numbers[i]);
        optical_flow flow;
        if (read_flo_file(base + buf, flow)) {
            flows[i] = std::move(flow);
        }
    }
    return flows;
}

// ============================================================================
// Section 6c: Motion statistics (for adaptive frame selection)
// ============================================================================

struct motion_stats {
    float mean_magnitude    = 0.0f;
    float p95_magnitude     = 0.0f;
    float max_magnitude     = 0.0f;
    float variance          = 0.0f;
    float high_motion_ratio = 0.0f;
    float complexity_score  = 0.0f;
};

static motion_stats compute_motion_stats(const optical_flow & flow,
                                         float motion_threshold,
                                         float percentile) {
    const int n = flow.width * flow.height;
    std::vector<float> mag(n);

    double sum = 0.0;
    int high_count = 0;
    for (int i = 0; i < n; i++) {
        mag[i] = std::sqrt(flow.u[i] * flow.u[i] + flow.v[i] * flow.v[i]);
        sum += mag[i];
        if (mag[i] > motion_threshold) high_count++;
    }

    motion_stats stats;
    stats.mean_magnitude = (float)(sum / n);
    stats.high_motion_ratio = (float)high_count / n;

    std::vector<float> sorted_mag = mag;
    std::sort(sorted_mag.begin(), sorted_mag.end());
    stats.max_magnitude = sorted_mag.back();

    int p_idx = std::min((int)(percentile * n), n - 1);
    stats.p95_magnitude = sorted_mag[p_idx];

    double var_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = mag[i] - stats.mean_magnitude;
        var_sum += diff * diff;
    }
    stats.variance = (float)(var_sum / n);

    return stats;
}

// Compute 5-component global complexity score from per-frame p95 values.
//
// Uses percentile-based normalization so the score is relative to the video's
// own motion distribution, not to hardcoded absolute thresholds. This makes
// it resolution-independent (works for 480p MSVD and 1024p Sintel alike).
//
// Each component is normalized to [0,1] using the video's own statistics:
//   - Mean:     percentile rank of the mean within the sorted p95 distribution
//   - Variance: coefficient of variation (std/mean), capped at 2.0
//   - Range:    (max-min)/median, capped at 10.0
//   - Temporal: fraction of consecutive frame-pairs where p95 changes by >50%
//               (measures how "bursty" the motion is, independent of magnitude)
static float compute_global_complexity(const std::vector<float> & p95_values) {
    if (p95_values.empty()) return 0.0f;
    int n = (int)p95_values.size();

    std::vector<float> sorted_p95 = p95_values;
    std::sort(sorted_p95.begin(), sorted_p95.end());

    // Component 1 (40%): Percentile rank of the mean within the distribution.
    // A video where the mean is near the top of its own distribution has
    // consistently high motion; one where it's near the bottom is mostly static
    // with a few spikes.
    double p95_mean = 0.0;
    for (float v : p95_values) p95_mean += v;
    p95_mean /= n;
    int rank = 0;
    for (int i = 0; i < n; i++) { if (sorted_p95[i] <= (float)p95_mean) rank = i; }
    float mean_norm = (float)(rank + 1) / n;  // 0..1

    // Component 2 (30%): Coefficient of variation (std / mean).
    // High CV means motion is unevenly distributed across frames.
    double p95_var = 0.0;
    for (float v : p95_values) {
        double d = v - p95_mean;
        p95_var += d * d;
    }
    p95_var /= n;
    float cv = (p95_mean > 1e-6) ? (float)(sqrt(p95_var) / p95_mean) : 0.0f;
    float var_norm = std::min(cv / 2.0f, 1.0f);  // CV of 2.0 → 1.0

    // Component 3 (20%): Relative range (max-min)/median.
    // Large relative range means diverse motion intensities.
    float p95_min = sorted_p95.front();
    float p95_max = sorted_p95.back();
    float median  = sorted_p95[n / 2];
    float rel_range = (median > 1e-6f) ? (p95_max - p95_min) / median : 0.0f;
    float range_norm = std::min(rel_range / 10.0f, 1.0f);  // rel_range of 10 → 1.0

    // Component 4 (10%): Temporal burstiness — fraction of consecutive frames
    // where p95 changes by more than 50%. Captures scene cuts and sudden motion.
    int n_bursts = 0;
    for (int i = 1; i < n; i++) {
        float prev = p95_values[i - 1];
        float curr = p95_values[i];
        float denom = std::max(prev, curr);
        if (denom > 1e-6f && std::abs(curr - prev) / denom > 0.5f) {
            n_bursts++;
        }
    }
    float burst_ratio = (n > 1) ? (float)n_bursts / (n - 1) : 0.0f;

    float score = 0.4f * mean_norm + 0.3f * var_norm + 0.2f * range_norm + 0.1f * burst_ratio;
    return std::min(score, 1.0f);
}

// Compute per-frame complexity scores using min-max normalization across all frames.
// Matches Python per-frame complexity from build_motion_profile().
static void compute_per_frame_complexity(std::vector<motion_stats> & all_stats) {
    if (all_stats.empty()) return;

    float mean_min = 1e9f, mean_max = 0.0f;
    float var_min  = 1e9f, var_max  = 0.0f;
    float range_min = 1e9f, range_max = 0.0f;

    for (const auto & s : all_stats) {
        mean_min  = std::min(mean_min,  s.mean_magnitude);
        mean_max  = std::max(mean_max,  s.mean_magnitude);
        var_min   = std::min(var_min,   s.variance);
        var_max   = std::max(var_max,   s.variance);
        range_min = std::min(range_min, s.max_magnitude);
        range_max = std::max(range_max, s.max_magnitude);
    }

    for (auto & s : all_stats) {
        float mn = (mean_max > mean_min) ? (s.mean_magnitude - mean_min) / (mean_max - mean_min) : 0.0f;
        float vn = (var_max  > var_min)  ? (s.variance - var_min) / (var_max - var_min) : 0.0f;
        float rn = (range_max > range_min) ? (s.max_magnitude - range_min) / (range_max - range_min) : 0.0f;
        mn = std::min(std::max(mn, 0.0f), 1.0f);
        vn = std::min(std::max(vn, 0.0f), 1.0f);
        rn = std::min(std::max(rn, 0.0f), 1.0f);
        s.complexity_score = std::min(0.4f * mn + 0.3f * vn + 0.2f * rn + 0.1f * s.high_motion_ratio, 1.0f);
    }
}

// Determine adaptive frame count from global complexity score.
// Thresholds are calibrated for the percentile-based complexity score (0.17–0.69
// typical range), producing 3–11 frames across both low-res (MSVD 480p) and
// high-res (Sintel 1024p) datasets.
static int adaptive_frame_count(float complexity, int base_frames, int min_frames, int max_frames) {
    int count;
    if (complexity > 0.55f)      count = base_frames + 3;
    else if (complexity > 0.45f) count = base_frames;
    else if (complexity > 0.35f) count = base_frames - 2;
    else if (complexity > 0.25f) count = base_frames - 4;
    else                         count = base_frames - 5;
    return std::max(min_frames, std::min(count, max_frames));
}

static int count_video_frames(const std::string & sintel_dir, const std::string & video_name) {
    std::string dir = sintel_dir + "/training/final/" + video_name;
    int count = 0;
    for (int i = 1; ; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", i);
        FILE * test = fopen((dir + buf).c_str(), "rb");
        if (!test) break;
        fclose(test);
        count++;
    }
    return count;
}

// ============================================================================
// Section 6d: Spatial patch analysis
// ============================================================================

struct patch_info {
    int   grid_x, grid_y;
    float motion_ratio;
    float mean_motion;
    float max_motion;
    bool  selected;
};

struct frame_spatial_analysis {
    int frame_index;
    std::vector<patch_info> patches;
    int n_selected_patches;
    int n_total_patches;
    float spatial_savings;
};

static frame_spatial_analysis analyze_spatial_patches(const optical_flow & flow,
                                                      int frame_index,
                                                      int grid_size,
                                                      float spatial_threshold,
                                                      float motion_threshold) {
    frame_spatial_analysis result;
    result.frame_index = frame_index;
    result.n_total_patches = grid_size * grid_size;

    int pw = flow.width / grid_size;
    int ph = flow.height / grid_size;

    for (int gy = 0; gy < grid_size; gy++) {
        for (int gx = 0; gx < grid_size; gx++) {
            int x0 = gx * pw;
            int y0 = gy * ph;
            int x1 = (gx == grid_size - 1) ? flow.width : x0 + pw;
            int y1 = (gy == grid_size - 1) ? flow.height : y0 + ph;

            float sum_mag = 0.0f;
            float max_mag = 0.0f;
            int high_count = 0;
            int total = 0;

            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int idx = y * flow.width + x;
                    float m = std::sqrt(flow.u[idx] * flow.u[idx] + flow.v[idx] * flow.v[idx]);
                    sum_mag += m;
                    max_mag = std::max(max_mag, m);
                    if (m > motion_threshold) high_count++;
                    total++;
                }
            }

            patch_info pi;
            pi.grid_x = gx;
            pi.grid_y = gy;
            pi.mean_motion = (total > 0) ? sum_mag / total : 0.0f;
            pi.max_motion = max_mag;
            pi.motion_ratio = (total > 0) ? (float)high_count / total : 0.0f;
            pi.selected = (pi.motion_ratio > spatial_threshold) ||
                          (pi.mean_motion > motion_threshold) ||
                          (pi.max_motion > motion_threshold * 2.0f);
            result.patches.push_back(pi);
        }
    }

    result.n_selected_patches = 0;
    for (const auto & p : result.patches) {
        if (p.selected) result.n_selected_patches++;
    }

    int min_patches = std::max(1, result.n_total_patches / 4);
    if (result.n_selected_patches < min_patches) {
        std::vector<int> indices(result.patches.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return result.patches[a].mean_motion > result.patches[b].mean_motion;
        });
        for (int idx : indices) {
            if (result.n_selected_patches >= min_patches) break;
            if (!result.patches[idx].selected) {
                result.patches[idx].selected = true;
                result.n_selected_patches++;
            }
        }
    }

    result.spatial_savings = 1.0f - (float)result.n_selected_patches / result.n_total_patches;
    return result;
}

static std::vector<bool> build_patch_active_mask(
    const frame_spatial_analysis & sa,
    int grid_size,
    int vit_nx, int vit_ny,
    float patch_reuse_motion = 0.0f) {

    int n_tokens = vit_nx * vit_ny;
    std::vector<bool> active(n_tokens, true);

    std::vector<bool> sa_active(grid_size * grid_size, false);
    for (const auto & p : sa.patches) {
        int idx = p.grid_y * grid_size + p.grid_x;
        if (patch_reuse_motion > 0.0f) {
            sa_active[idx] = (p.mean_motion >= patch_reuse_motion);
        } else {
            sa_active[idx] = p.selected;
        }
    }

    for (int ty = 0; ty < vit_ny; ty++) {
        for (int tx = 0; tx < vit_nx; tx++) {
            int sa_gx = tx * grid_size / vit_nx;
            int sa_gy = ty * grid_size / vit_ny;
            sa_gx = std::min(sa_gx, grid_size - 1);
            sa_gy = std::min(sa_gy, grid_size - 1);
            active[ty * vit_nx + tx] = sa_active[sa_gy * grid_size + sa_gx];
        }
    }

    return active;
}

// ============================================================================
// Section 6e: Embedding cache
// ============================================================================

struct embedding_cache {
    int nx_full  = 0;
    int ny_full  = 0;
    int n_embd   = 0;
    std::vector<float> embeddings;
    bool valid = false;

    void allocate(int nx, int ny, int embd) {
        nx_full = nx;
        ny_full = ny;
        n_embd  = embd;
        embeddings.resize(nx * ny * embd, 0.0f);
        valid = false;
    }

    float * at(int gy, int gx) {
        return embeddings.data() + (gy * nx_full + gx) * n_embd;
    }

    void store_full(const float * src) {
        memcpy(embeddings.data(), src, embeddings.size() * sizeof(float));
        valid = true;
    }

    void selective_update(const float * new_embd,
                          const std::vector<bool> & patch_active,
                          int & n_updated, int & n_kept) {
        n_updated = 0;
        n_kept = 0;
        int n_total = nx_full * ny_full;
        for (int i = 0; i < n_total; i++) {
            if (patch_active[i]) {
                memcpy(embeddings.data() + i * n_embd,
                       new_embd + i * n_embd,
                       n_embd * sizeof(float));
                n_updated++;
            } else {
                n_kept++;
            }
        }
        valid = true;
    }
};

// ============================================================================
// Section 6f: Crop helpers
// ============================================================================

struct frame_crop_info {
    int  frame_index;
    int  orig_w, orig_h;
    int  crop_x, crop_y, crop_w, crop_h;
    bool was_cropped;
    int  estimated_tokens_full;
    int  estimated_tokens_crop;
};

static int estimate_tokens(int w, int h, int align = 28) {
    int w_a = std::max(align, (int)(std::round((float)w / align) * align));
    int h_a = std::max(align, (int)(std::round((float)h / align) * align));
    return (w_a / align) * (h_a / align);
}

static frame_crop_info compute_crop_region(const frame_spatial_analysis & sa,
                                            int img_w, int img_h,
                                            int grid_size,
                                            int crop_padding,
                                            int min_crop_dim) {
    frame_crop_info ci;
    ci.frame_index = sa.frame_index;
    ci.orig_w = img_w;
    ci.orig_h = img_h;
    ci.estimated_tokens_full = estimate_tokens(img_w, img_h);

    int gx_min = grid_size, gx_max = -1;
    int gy_min = grid_size, gy_max = -1;

    for (const auto & p : sa.patches) {
        if (p.selected) {
            gx_min = std::min(gx_min, p.grid_x);
            gx_max = std::max(gx_max, p.grid_x);
            gy_min = std::min(gy_min, p.grid_y);
            gy_max = std::max(gy_max, p.grid_y);
        }
    }

    if (gx_max < 0) {
        ci.crop_x = 0; ci.crop_y = 0;
        ci.crop_w = img_w; ci.crop_h = img_h;
        ci.was_cropped = false;
        ci.estimated_tokens_crop = ci.estimated_tokens_full;
        return ci;
    }

    int pw = img_w / grid_size;
    int ph = img_h / grid_size;

    int px_min = gx_min * pw;
    int py_min = gy_min * ph;
    int px_max = (gx_max == grid_size - 1) ? img_w : (gx_max + 1) * pw;
    int py_max = (gy_max == grid_size - 1) ? img_h : (gy_max + 1) * ph;

    px_min = std::max(0, px_min - crop_padding);
    py_min = std::max(0, py_min - crop_padding);
    px_max = std::min(img_w, px_max + crop_padding);
    py_max = std::min(img_h, py_max + crop_padding);

    int cw = px_max - px_min;
    int ch = py_max - py_min;

    if (cw < min_crop_dim) {
        int expand = min_crop_dim - cw;
        int left  = expand / 2;
        int right = expand - left;
        px_min = std::max(0, px_min - left);
        px_max = std::min(img_w, px_max + right);
        cw = px_max - px_min;
        if (cw < min_crop_dim) {
            if (px_min > 0) px_min = std::max(0, px_max - min_crop_dim);
            else px_max = std::min(img_w, px_min + min_crop_dim);
            cw = px_max - px_min;
        }
    }
    if (ch < min_crop_dim) {
        int expand = min_crop_dim - ch;
        int top    = expand / 2;
        int bottom = expand - top;
        py_min = std::max(0, py_min - top);
        py_max = std::min(img_h, py_max + bottom);
        ch = py_max - py_min;
        if (ch < min_crop_dim) {
            if (py_min > 0) py_min = std::max(0, py_max - min_crop_dim);
            else py_max = std::min(img_h, py_min + min_crop_dim);
            ch = py_max - py_min;
        }
    }

    ci.crop_x = px_min;
    ci.crop_y = py_min;
    ci.crop_w = cw;
    ci.crop_h = ch;

    double crop_area = (double)cw * ch;
    double full_area = (double)img_w * img_h;
    if (crop_area >= 0.9 * full_area) {
        ci.crop_x = 0; ci.crop_y = 0;
        ci.crop_w = img_w; ci.crop_h = img_h;
        ci.was_cropped = false;
        ci.estimated_tokens_crop = ci.estimated_tokens_full;
    } else {
        ci.was_cropped = true;
        ci.estimated_tokens_crop = estimate_tokens(cw, ch);
    }

    return ci;
}

static mtmd::bitmap crop_bitmap(const mtmd::bitmap & full, const frame_crop_info & ci) {
    const unsigned char * src = full.data();
    uint32_t src_w = full.nx();
    std::vector<unsigned char> cropped(ci.crop_w * ci.crop_h * 3);
    for (int row = 0; row < ci.crop_h; row++) {
        memcpy(&cropped[row * ci.crop_w * 3],
               &src[((ci.crop_y + row) * src_w + ci.crop_x) * 3],
               ci.crop_w * 3);
    }
    return mtmd::bitmap((uint32_t)ci.crop_w, (uint32_t)ci.crop_h, cropped.data());
}

// ============================================================================
// Section 6g: Online frame selector (adaptive cumulative motion)
// ============================================================================

struct selected_frame {
    int   frame_index;
    float motion_weight;
    float complexity_score;
};

struct frame_decision {
    int   frame_index;
    float p95_magnitude;
    float complexity_score;
    float importance_weight;
    bool  selected;
};

// Complexity-weighted probabilistic frame selector.
// Matches Python SpatioTemporalOptimizer:
//   1. Read all flow files → per-frame motion_stats + p95 values
//   2. Compute global complexity → adaptive frame count via step function
//   3. Importance weights: frames above selection_percentile get weight=p95,
//      frames below get weight=0.1
//   4. Weighted sampling without replacement (std::discrete_distribution)
//   5. Post-selection filter: remove frames with per-frame complexity < floor
//   6. Always include first and last frame as temporal anchors
struct online_frame_selector {
    int   base_frames;
    int   min_frames;
    int   max_frames;
    float motion_threshold;
    float motion_percentile;
    float selection_percentile;  // importance weight threshold (0-100)
    float complexity_floor;      // remove selected frames below this

    float global_complexity;
    float motion_coverage;       // fraction of total motion captured

    std::vector<frame_decision> decisions;
    std::vector<selected_frame> selected_frames;

    void init(int base, int mn, int mx, float mt, float mp, float sel_pct, float cfloor) {
        base_frames = base;
        min_frames = mn;
        max_frames = mx;
        motion_threshold = mt;
        motion_percentile = mp;
        selection_percentile = sel_pct;
        complexity_floor = cfloor;
        global_complexity = 0.0f;
        motion_coverage = 0.0f;
        decisions.clear();
        selected_frames.clear();
    }

    void run(const std::string & flow_dir, const std::string & video_name, int n_total_frames) {
        // -------------------------------------------------------------------
        // Pass 1: Read per-frame motion stats
        //   First try cached stats file (fast, <1ms).
        //   Fall back to reading all .flo files (slow, ~700ms).
        // -------------------------------------------------------------------
        int64_t t_io_start = ggml_time_ms();
        std::vector<motion_stats> all_stats(n_total_frames);
        std::vector<float> p95_values;

        // Try loading cached stats
        std::string cache_path = flow_dir + "/" + video_name + "/flow_stats.json";
        bool cached = false;
        {
            FILE * cf = fopen(cache_path.c_str(), "rb");
            if (cf) {
                fseek(cf, 0, SEEK_END);
                long sz = ftell(cf);
                fseek(cf, 0, SEEK_SET);
                std::string buf(sz, '\0');
                fread(&buf[0], 1, sz, cf);
                fclose(cf);

                // Parse JSON array of {p95, mean, var, high_ratio, max}
                // Simple manual parse since we know the format
                try {
                    auto jdata = json::parse(buf);
                    if (jdata.is_array() && (int)jdata.size() == n_total_frames) {
                        for (int i = 0; i < n_total_frames; i++) {
                            auto & js = jdata[i];
                            all_stats[i].p95_magnitude     = js.value("p95", 0.0f);
                            all_stats[i].mean_magnitude    = js.value("mean", 0.0f);
                            all_stats[i].variance          = js.value("var", 0.0f);
                            all_stats[i].high_motion_ratio = js.value("high_ratio", 0.0f);
                            all_stats[i].max_magnitude     = js.value("max", 0.0f);
                        }
                        cached = true;
                    }
                } catch (...) {}
            }
        }

        if (!cached) {
            // Fall back: read all .flo files
            for (int i = 1; i <= n_total_frames; i++) {
                if (i > 1) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "/frame_%04d.flo", i - 1);
                    std::string flo_path = flow_dir + "/" + video_name + buf;

                    optical_flow flow;
                    if (read_flo_file(flo_path, flow)) {
                        all_stats[i - 1] = compute_motion_stats(flow, motion_threshold, motion_percentile);
                    }
                }
            }

            // Save cache for next time
            json jcache = json::array();
            for (int i = 0; i < n_total_frames; i++) {
                jcache.push_back({
                    {"p95",        all_stats[i].p95_magnitude},
                    {"mean",       all_stats[i].mean_magnitude},
                    {"var",        all_stats[i].variance},
                    {"high_ratio", all_stats[i].high_motion_ratio},
                    {"max",        all_stats[i].max_magnitude},
                });
            }
            FILE * wf = fopen(cache_path.c_str(), "wb");
            if (wf) {
                std::string s = jcache.dump();
                fwrite(s.data(), 1, s.size(), wf);
                fclose(wf);
            }
        }

        // Build decisions + p95 values
        for (int i = 1; i <= n_total_frames; i++) {
            frame_decision fd;
            fd.frame_index = i;
            fd.p95_magnitude = (i > 1) ? all_stats[i - 1].p95_magnitude : 0.0f;
            fd.complexity_score = 0.0f;
            fd.importance_weight = 0.0f;
            fd.selected = false;
            p95_values.push_back(fd.p95_magnitude);
            decisions.push_back(fd);
        }

        int64_t t_io_end = ggml_time_ms();
        printf("  [selection timing] Flow stats: %lld ms (%s)\n",
               (long long)(t_io_end - t_io_start), cached ? "cached" : "from .flo files");

        // -------------------------------------------------------------------
        // Pass 2: Global complexity + per-frame complexity
        // -------------------------------------------------------------------
        int64_t t_logic_start = ggml_time_ms();

        global_complexity = compute_global_complexity(p95_values);
        compute_per_frame_complexity(all_stats);

        for (int i = 0; i < n_total_frames; i++) {
            decisions[i].complexity_score = all_stats[i].complexity_score;
        }

        // -------------------------------------------------------------------
        // Pass 3: Adaptive frame count from complexity
        // -------------------------------------------------------------------
        int target_count = adaptive_frame_count(global_complexity, base_frames, min_frames, max_frames);

        printf("  Global complexity: %.3f → target %d frames (base=%d, range=[%d,%d])\n",
               global_complexity, target_count, base_frames, min_frames, max_frames);

        // -------------------------------------------------------------------
        // Pass 4: Importance weights via percentile threshold
        // -------------------------------------------------------------------
        // Collect all p95 values, find threshold at selection_percentile
        std::vector<float> sorted_p95 = p95_values;
        std::sort(sorted_p95.begin(), sorted_p95.end());
        int pct_idx = std::min((int)(selection_percentile / 100.0f * n_total_frames), n_total_frames - 1);
        float weight_threshold = sorted_p95[pct_idx];

        std::vector<double> weights(n_total_frames);
        for (int i = 0; i < n_total_frames; i++) {
            weights[i] = (p95_values[i] >= weight_threshold) ? (double)p95_values[i] : 0.1;
            decisions[i].importance_weight = (float)weights[i];
        }

        // Normalize to probabilities
        double w_sum = 0.0;
        for (double w : weights) w_sum += w;
        if (w_sum <= 0.0) w_sum = 1.0;
        std::vector<double> probs(n_total_frames);
        for (int i = 0; i < n_total_frames; i++) {
            probs[i] = weights[i] / w_sum;
        }

        // -------------------------------------------------------------------
        // Pass 5: Weighted sampling without replacement
        // -------------------------------------------------------------------
        // Always include frame 0 (keyframe) and last frame (temporal anchor)
        std::set<int> selected_set;
        selected_set.insert(0);                     // first frame
        selected_set.insert(n_total_frames - 1);    // last frame

        std::mt19937 rng(42);  // deterministic seed for reproducibility
        int remaining = target_count - (int)selected_set.size();

        // Zero out probabilities for already-selected frames
        std::vector<double> sample_probs = probs;
        for (int idx : selected_set) sample_probs[idx] = 0.0;

        for (int k = 0; k < remaining && k < n_total_frames - (int)selected_set.size(); k++) {
            // Re-normalize
            double sp_sum = 0.0;
            for (double p : sample_probs) sp_sum += p;
            if (sp_sum <= 0.0) break;

            std::discrete_distribution<int> dist(sample_probs.begin(), sample_probs.end());
            int idx = dist(rng);

            selected_set.insert(idx);
            sample_probs[idx] = 0.0;  // remove from future sampling
        }

        // -------------------------------------------------------------------
        // Pass 6: Post-selection complexity filter
        // -------------------------------------------------------------------
        std::vector<int> filtered;
        for (int idx : selected_set) {
            if (decisions[idx].complexity_score >= complexity_floor
                || idx == 0
                || idx == n_total_frames - 1) {
                filtered.push_back(idx);
            }
        }

        // Ensure minimum frame count after filtering
        if ((int)filtered.size() < min_frames) {
            // Re-add highest-complexity removed frames
            std::vector<std::pair<float, int>> by_complexity;
            for (int idx : selected_set) {
                if (std::find(filtered.begin(), filtered.end(), idx) == filtered.end()) {
                    by_complexity.push_back({decisions[idx].complexity_score, idx});
                }
            }
            std::sort(by_complexity.rbegin(), by_complexity.rend());
            for (auto & [score, idx] : by_complexity) {
                if ((int)filtered.size() >= min_frames) break;
                filtered.push_back(idx);
            }
            std::sort(filtered.begin(), filtered.end());
        }

        // -------------------------------------------------------------------
        // Build selected_frames (1-based frame indices, sorted temporally)
        // -------------------------------------------------------------------
        double total_motion = 0.0, selected_motion = 0.0;
        for (float v : p95_values) total_motion += v;

        for (int idx : filtered) {
            decisions[idx].selected = true;

            selected_frame sf;
            sf.frame_index = idx + 1;  // convert 0-based → 1-based
            sf.motion_weight = p95_values[idx];
            sf.complexity_score = decisions[idx].complexity_score;
            selected_frames.push_back(sf);

            selected_motion += p95_values[idx];
        }

        motion_coverage = (total_motion > 0.0) ? (float)(selected_motion / total_motion) : 0.0f;

        int64_t t_logic_end = ggml_time_ms();
        printf("  [selection timing] Selection logic: %lld ms\n",
               (long long)(t_logic_end - t_logic_start));

        printf("Adaptive selection: %zu frames from %d (complexity=%.3f, coverage=%.1f%%)\n",
               selected_frames.size(), n_total_frames, global_complexity, motion_coverage * 100.0f);
        printf("  Selected: ");
        for (const auto & sf : selected_frames) {
            printf("%d(c=%.2f) ", sf.frame_index, sf.complexity_score);
        }
        printf("\n");
    }
};

// ============================================================================
// Section 6h: Pruned token injection (M-RoPE aware)
// ============================================================================

struct scored_visual_token {
    int   token_idx;
    float score;
};

// ============================================================================
// Section 6h-1: MOTION+KF scoring (validated best strategy across video types)
//
// Scores each visual token by:
//   50% optical flow motion magnitude (captures dynamic content)
//   50% cosine distance from keyframe embedding (captures semantic change)
// With 4x4 spatial grid guarantee for minimum spatial diversity.
// ============================================================================

static std::vector<scored_visual_token> score_and_select_motion_kf(
    const float * embd,          // current frame embeddings [n_tok * n_embd]
    const float * kf_embd,       // keyframe (frame 0) embeddings, may be nullptr
    int n_tok, int n_embd,
    const optical_flow & flow,
    int nx, int ny,
    float keep_ratio)
{
    int n_keep = std::max(1, (int)(n_tok * keep_ratio));

    // Motion scores
    std::vector<float> motion(n_tok, 0.0f);
    bool has_flow = flow.width > 0;
    float m_max = 1e-8f;
    if (has_flow) {
        motion = compute_token_motion_scores(flow, nx, ny);
        for (float m : motion) m_max = std::max(m_max, m);
    }

    // Per-token cosine distance from keyframe
    std::vector<float> cos_kf(n_tok, 0.0f);
    float cos_kf_max = 1e-8f;
    if (kf_embd) {
        for (int t = 0; t < n_tok; t++) {
            const float * e = embd + t * n_embd;
            const float * kf = kf_embd + t * n_embd;
            float dot = 0.0f, en2 = 0.0f, kn2 = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                dot += e[d] * kf[d];
                en2 += e[d] * e[d];
                kn2 += kf[d] * kf[d];
            }
            float e_norm = sqrtf(en2 + 1e-8f);
            float k_norm = sqrtf(kn2 + 1e-8f);
            cos_kf[t] = 1.0f - dot / (e_norm * k_norm);
            cos_kf_max = std::max(cos_kf_max, cos_kf[t]);
        }
    }

    // Combined score: 50% motion + 50% keyframe distance
    std::vector<scored_visual_token> scores(n_tok);
    for (int t = 0; t < n_tok; t++) {
        float s_motion = has_flow ? motion[t] / m_max : 0.0f;
        float s_kf = kf_embd ? cos_kf[t] / cos_kf_max : 0.0f;
        scores[t] = {t, 0.5f * s_motion + 0.5f * s_kf};
    }

    // Spatial grid guarantee (4x4): keep best-scoring token per grid cell
    int grid = 4;
    std::vector<bool> force_keep(n_tok, false);
    for (int gy = 0; gy < grid; gy++)
        for (int gx = 0; gx < grid; gx++) {
            int best = -1; float best_s = -1.0f;
            int y0 = gy * ny / grid, y1 = (gy+1) * ny / grid;
            int x0 = gx * nx / grid, x1 = (gx+1) * nx / grid;
            for (int ty = y0; ty < y1; ty++)
                for (int tx = x0; tx < x1; tx++) {
                    int i = ty * nx + tx;
                    if (scores[i].score > best_s) { best_s = scores[i].score; best = i; }
                }
            if (best >= 0) force_keep[best] = true;
        }

    // Select: forced grid tokens + top remaining by score
    std::vector<scored_visual_token> forced, candidates;
    for (int t = 0; t < n_tok; t++) {
        if (force_keep[t]) forced.push_back(scores[t]);
        else candidates.push_back(scores[t]);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const scored_visual_token & a, const scored_visual_token & b) { return a.score > b.score; });

    std::vector<scored_visual_token> selected = forced;
    int remaining = n_keep - (int)forced.size();
    for (int i = 0; i < remaining && i < (int)candidates.size(); i++)
        selected.push_back(candidates[i]);

    // Sort by position for correct sequential injection
    std::sort(selected.begin(), selected.end(),
              [](const scored_visual_token & a, const scored_visual_token & b) { return a.token_idx < b.token_idx; });

    return selected;
}

// ============================================================================
// Section 6h-2: Token merging (PruMerge-style)
//
// For each dropped token, finds the nearest surviving token by spatial distance
// on the ViT grid. Merges the dropped embedding into the survivor using a
// weighted average (weight = 1/spatial_distance). This preserves information
// from dropped regions in fewer tokens instead of discarding it entirely.
//
// Returns a new embedding buffer of size [n_selected * n_embd] where each
// selected token's embedding has absorbed its spatial neighbors.
// ============================================================================

static std::vector<float> merge_dropped_tokens(
    const float * full_embd,                        // [n_tok * n_embd] all embeddings
    const std::vector<scored_visual_token> & selected,  // selected tokens (sorted by idx)
    int n_tok, int n_embd,
    int nx, int ny)
{
    int n_sel = (int)selected.size();
    if (n_sel == 0 || n_sel >= n_tok) {
        // Nothing to merge: return copy of selected embeddings
        std::vector<float> out(n_sel * n_embd);
        for (int i = 0; i < n_sel; i++) {
            memcpy(out.data() + i * n_embd,
                   full_embd + selected[i].token_idx * n_embd,
                   n_embd * sizeof(float));
        }
        return out;
    }

    // Build set of selected indices for fast lookup
    std::set<int> sel_set;
    for (const auto & s : selected) sel_set.insert(s.token_idx);

    // Pre-compute grid positions for selected tokens
    struct sel_info {
        int grid_x, grid_y;
        int sel_idx;  // index into selected[]
    };
    std::vector<sel_info> sel_grid(n_sel);
    for (int i = 0; i < n_sel; i++) {
        int idx = selected[i].token_idx;
        sel_grid[i] = {idx % nx, idx / nx, i};
    }

    // Initialize merged embeddings with selected tokens' embeddings
    std::vector<float> merged(n_sel * n_embd);
    std::vector<float> weights(n_sel, 1.0f);  // start with weight 1.0 for self

    for (int i = 0; i < n_sel; i++) {
        memcpy(merged.data() + i * n_embd,
               full_embd + selected[i].token_idx * n_embd,
               n_embd * sizeof(float));
    }

    // For each dropped token, find nearest selected token and accumulate
    int n_merged = 0;
    for (int t = 0; t < n_tok; t++) {
        if (sel_set.count(t)) continue;  // this token is selected, skip

        int tx = t % nx, ty = t / nx;

        // Find nearest selected token by L2 grid distance
        int best_i = 0;
        float best_dist = FLT_MAX;
        for (int i = 0; i < n_sel; i++) {
            float dx = (float)(tx - sel_grid[i].grid_x);
            float dy = (float)(ty - sel_grid[i].grid_y);
            float dist = dx * dx + dy * dy;
            if (dist < best_dist) {
                best_dist = dist;
                best_i = i;
            }
        }

        // Merge weight: inverse distance (closer dropped tokens contribute more)
        // Minimum distance of 1.0 to avoid division by zero for adjacent tokens
        float w = 1.0f / std::max(1.0f, sqrtf(best_dist));

        // Accumulate weighted embedding
        const float * drop_embd = full_embd + t * n_embd;
        float * target = merged.data() + best_i * n_embd;
        for (int d = 0; d < n_embd; d++) {
            target[d] += w * drop_embd[d];
        }
        weights[best_i] += w;
        n_merged++;
    }

    // Normalize by total weight
    for (int i = 0; i < n_sel; i++) {
        if (weights[i] > 1.0f + 1e-6f) {  // had merges
            float inv_w = 1.0f / weights[i];
            float * emb = merged.data() + i * n_embd;
            for (int d = 0; d < n_embd; d++) {
                emb[d] *= inv_w;
            }
        }
    }

    return merged;
}

static int32_t inject_pruned_visual_tokens(
    llama_context * lctx,
    mtmd_context * ctx_vision,
    const float * full_embeddings,
    const std::vector<scored_visual_token> & selected,
    int nx, int ny, int n_embd,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch_size,
    llama_pos * new_n_past,
    const float * merged_embd = nullptr) {  // if non-null, use this instead of copying from full_embeddings

    int n_keep = (int)selected.size();
    if (n_keep == 0) {
        *new_n_past = n_past + std::max(nx, ny);
        return 0;
    }

    // Build subset embedding buffer (or use pre-merged buffer)
    std::vector<float> pruned_embd;
    if (merged_embd) {
        pruned_embd.assign(merged_embd, merged_embd + n_keep * n_embd);
    } else {
        pruned_embd.resize(n_keep * n_embd);
        for (int i = 0; i < n_keep; i++) {
            memcpy(pruned_embd.data() + i * n_embd,
                   full_embeddings + selected[i].token_idx * n_embd,
                   n_embd * sizeof(float));
        }
    }

    // Build M-RoPE positions
    bool use_mrope = mtmd_decode_use_mrope(ctx_vision);
    int n_pos_per_embd = use_mrope ? 4 : 1;
    std::vector<llama_pos> pos(n_keep * n_pos_per_embd, 0);

    if (use_mrope) {
        for (int i = 0; i < n_keep; i++) {
            int idx = selected[i].token_idx;
            int grid_y = idx / nx;
            int grid_x = idx % nx;
            pos[i]                = n_past;            // temporal
            pos[i + n_keep]       = n_past + grid_y;   // y
            pos[i + n_keep * 2]   = n_past + grid_x;   // x
            pos[i + n_keep * 3]   = 0;                  // unused
        }
    } else {
        for (int i = 0; i < n_keep; i++) {
            pos[i] = n_past + i;
        }
    }

    std::vector<int32_t> n_seq_id_vec(n_keep, 1);
    std::vector<llama_seq_id> seq_id_0(1, seq_id);
    std::vector<llama_seq_id *> seq_ids(n_keep, seq_id_0.data());
    std::vector<int8_t> logits(n_keep, 0);

    // Non-causal attention for image tokens
    if (mtmd_decode_use_non_causal(ctx_vision)) {
        llama_set_causal_attn(lctx, false);
    }

    // Decode in sub-batches
    int n_img_batches = (n_keep + n_batch_size - 1) / n_batch_size;
    int32_t ret = 0;
    for (int ib = 0; ib < n_img_batches; ib++) {
        int offset = ib * n_batch_size;
        int n_tokens_batch = std::min(n_batch_size, n_keep - offset);

        std::vector<llama_pos> pos_view;
        llama_pos * pos_ptr;
        if (use_mrope) {
            pos_view.reserve(n_tokens_batch * 4);
            for (int d = 0; d < 4; d++) {
                size_t src_idx = d * n_keep + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens_batch);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + offset;
        }

        llama_batch sub_batch = {
            /*n_tokens       =*/ n_tokens_batch,
            /*tokens         =*/ nullptr,
            /*embd           =*/ pruned_embd.data() + offset * n_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ n_seq_id_vec.data() + offset,
            /*seq_id         =*/ seq_ids.data() + offset,
            /*logits         =*/ logits.data() + offset,
        };

        ret = llama_decode(lctx, sub_batch);
        if (ret != 0) {
            fprintf(stderr, "Failed to decode pruned image batch %d/%d\n", ib + 1, n_img_batches);
            break;
        }
    }

    if (mtmd_decode_use_non_causal(ctx_vision)) {
        llama_set_causal_attn(lctx, true);
    }

    // CRITICAL: n_past advances by max(nx, ny) regardless of n_keep (M-RoPE constraint)
    *new_n_past = n_past + std::max(nx, ny);
    return ret;
}

// ============================================================================
// Section 7: Benchmark results
// ============================================================================

struct benchmark_result {
    std::string video_name;
    int    n_frames             = 0;
    int    n_frames_total       = 0;   // total available in video
    int    n_visual_tokens      = 0;
    int    n_visual_tokens_kept = 0;
    int    n_tokens_generated   = 0;
    float  prefill_time_ms      = 0.0f;
    float  encode_time_ms       = 0.0f;
    float  eviction_time_ms     = 0.0f;
    float  generation_time_ms   = 0.0f;
    float  selection_time_ms    = 0.0f;
    float  total_time_ms        = 0.0f;
    float  tokens_per_sec       = 0.0f;
    float  attn_retained_pct    = 0.0f;
    std::string generated_text;
    bool   pyramid_enabled      = false;
    float  pvc_min_ratio        = 0.0f;

    // Adaptive + spatial cache metrics
    bool   adaptive_enabled     = false;
    bool   spatial_cache_enabled = false;
    int    n_vit_skipped        = 0;   // frames where ViT was entirely skipped
    int    n_vit_cropped        = 0;   // frames where only active region was crop-encoded
    int    n_vit_full           = 0;   // frames fully ViT-encoded
    float  vit_savings_pct      = 0.0f;
    float  global_complexity    = 0.0f;
    float  motion_coverage      = 0.0f;

    // Embedding diversity (SVD entropy of visual embeddings injected into LLM)
    float  embedding_entropy    = 0.0f;  // higher = more diverse visual information
    float  avg_cosine_dist      = 0.0f;  // avg pairwise cosine distance between tokens

    json to_json() const {
        json j;
        j["video_name"]           = video_name;
        j["n_frames"]             = n_frames;
        j["n_frames_total"]       = n_frames_total;
        j["n_visual_tokens"]      = n_visual_tokens;
        j["n_visual_tokens_kept"] = n_visual_tokens_kept;
        j["n_tokens_generated"]   = n_tokens_generated;
        j["prefill_time_ms"]      = prefill_time_ms;
        j["encode_time_ms"]       = encode_time_ms;
        j["eviction_time_ms"]     = eviction_time_ms;
        j["generation_time_ms"]   = generation_time_ms;
        j["selection_time_ms"]    = selection_time_ms;
        j["total_time_ms"]        = total_time_ms;
        j["tokens_per_sec"]       = tokens_per_sec;
        j["attn_retained_pct"]    = attn_retained_pct;
        j["generated_text"]       = generated_text;
        j["pyramid_enabled"]      = pyramid_enabled;
        j["pvc_min_ratio"]        = pvc_min_ratio;
        j["adaptive_enabled"]     = adaptive_enabled;
        j["spatial_cache_enabled"] = spatial_cache_enabled;
        j["n_vit_skipped"]        = n_vit_skipped;
        j["n_vit_cropped"]        = n_vit_cropped;
        j["n_vit_full"]           = n_vit_full;
        j["vit_savings_pct"]      = vit_savings_pct;
        j["global_complexity"]    = global_complexity;
        j["motion_coverage"]      = motion_coverage;
        j["spectral_entropy"]     = embedding_entropy;  // Shannon entropy of normalized singular values
        j["avg_cosine_dist"]      = avg_cosine_dist;
        return j;
    }
};

// ============================================================================
// Section 8: Main inference pipeline
// ============================================================================

// ============================================================================
// Section 7b: Embedding diversity metrics
// ============================================================================

// Compute diversity of visual embeddings:
//
// 1. Spectral entropy (Shannon entropy of normalized singular values).
//    Measures the effective dimensionality of the token set.
//    - High: tokens span many independent directions (diverse information)
//    - Low: tokens cluster on few directions (redundant)
//    - Token-count independent: 100 diverse tokens > 5000 redundant ones
//    - Max value = log(k) where k = min(n_sample, n_embd)
//
//    Computed via eigendecomposition of the Gram matrix G = X X^T [n×n]
//    on a subsample of tokens. The eigenvalues of G equal σ² (squared
//    singular values of X), so we normalize and compute entropy.
//
// 2. Avg pairwise cosine distance (sampled).
//    Mean (1 - cosine_similarity) between token pairs.
//
static void compute_embedding_diversity(
    const std::vector<float> & all_embd,  // [n_tok * n_embd]
    int n_tok, int n_embd,
    float & out_entropy, float & out_avg_cos_dist)
{
    out_entropy = 0.0f;
    out_avg_cos_dist = 0.0f;
    if (n_tok < 2 || n_embd < 1) return;

    // Subsample for efficiency
    int n_sample = std::min(n_tok, 200);
    std::vector<int> indices(n_tok);
    std::iota(indices.begin(), indices.end(), 0);
    if (n_tok > n_sample) {
        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    // Build centered subsampled matrix X [n_sample × n_embd]
    // Center by subtracting mean per dimension
    std::vector<double> mean(n_embd, 0.0);
    for (int i = 0; i < n_sample; i++) {
        const float * e = all_embd.data() + indices[i] * n_embd;
        for (int d = 0; d < n_embd; d++) mean[d] += e[d];
    }
    for (int d = 0; d < n_embd; d++) mean[d] /= n_sample;

    std::vector<double> X(n_sample * n_embd);
    for (int i = 0; i < n_sample; i++) {
        const float * e = all_embd.data() + indices[i] * n_embd;
        for (int d = 0; d < n_embd; d++) {
            X[i * n_embd + d] = e[d] - mean[d];
        }
    }

    // Compute Gram matrix G = X X^T [n_sample × n_sample]
    // G[i][j] = dot(X[i], X[j])
    std::vector<double> G(n_sample * n_sample, 0.0);
    for (int i = 0; i < n_sample; i++) {
        for (int j = i; j < n_sample; j++) {
            double dot = 0;
            for (int d = 0; d < n_embd; d++) {
                dot += X[i * n_embd + d] * X[j * n_embd + d];
            }
            G[i * n_sample + j] = dot;
            G[j * n_sample + i] = dot;
        }
    }

    // Power iteration to find top-k eigenvalues of G
    // Use iterative deflation: find eigenvalue, subtract, repeat
    // This gives us the dominant singular values for spectral entropy
    int k_eig = std::min(n_sample, 100);  // top 100 eigenvalues suffice
    std::vector<double> eigenvalues;
    std::vector<double> G_work = G;  // working copy for deflation

    for (int k = 0; k < k_eig; k++) {
        // Power iteration for largest eigenvalue of G_work
        std::vector<double> v(n_sample);
        std::mt19937 rng2(42 + k);
        std::normal_distribution<double> normal(0, 1);
        for (int i = 0; i < n_sample; i++) v[i] = normal(rng2);

        double eigenval = 0;
        for (int iter = 0; iter < 100; iter++) {
            // w = G_work @ v
            std::vector<double> w(n_sample, 0.0);
            for (int i = 0; i < n_sample; i++) {
                for (int j = 0; j < n_sample; j++) {
                    w[i] += G_work[i * n_sample + j] * v[j];
                }
            }

            // eigenvalue = ||w||
            double norm = 0;
            for (int i = 0; i < n_sample; i++) norm += w[i] * w[i];
            norm = sqrt(norm);
            if (norm < 1e-10) break;

            eigenval = norm;
            for (int i = 0; i < n_sample; i++) v[i] = w[i] / norm;
        }

        if (eigenval < 1e-10) break;
        eigenvalues.push_back(eigenval);

        // Deflate: G_work -= eigenval * v v^T
        for (int i = 0; i < n_sample; i++) {
            for (int j = 0; j < n_sample; j++) {
                G_work[i * n_sample + j] -= eigenval * v[i] * v[j];
            }
        }
    }

    // Spectral entropy: normalize eigenvalues to probabilities, compute Shannon entropy
    if (!eigenvalues.empty()) {
        double eig_sum = 0;
        for (double e : eigenvalues) eig_sum += e;

        if (eig_sum > 1e-10) {
            for (double e : eigenvalues) {
                double p = e / eig_sum;
                if (p > 1e-10) {
                    out_entropy -= (float)(p * log(p));
                }
            }
        }
    }

    // Metric 2: Avg pairwise cosine distance (sampled)
    int max_pairs = 10000;
    double cos_dist_sum = 0;
    int n_pairs = 0;
    for (int i = 0; i < n_sample && n_pairs < max_pairs; i++) {
        for (int j = i + 1; j < n_sample && n_pairs < max_pairs; j++) {
            const float * a = all_embd.data() + indices[i] * n_embd;
            const float * b = all_embd.data() + indices[j] * n_embd;
            float dot = 0, na2 = 0, nb2 = 0;
            for (int d = 0; d < n_embd; d++) {
                dot += a[d] * b[d];
                na2 += a[d] * a[d];
                nb2 += b[d] * b[d];
            }
            float cos_sim = dot / (sqrtf(na2 + 1e-8f) * sqrtf(nb2 + 1e-8f));
            cos_dist_sum += 1.0 - cos_sim;
            n_pairs++;
        }
    }
    if (n_pairs > 0) out_avg_cos_dist = (float)(cos_dist_sum / n_pairs);
}

static benchmark_result run_inference(
    pyramid_vlm_context & ctx,
    const gen_params & gp,
    const std::vector<std::string> & frame_paths,
    const std::vector<int> & frame_numbers,
    const std::vector<frame_spatial_analysis> & spatial_analyses,
    multi_qcur_capture_state & capture_state)
{
    benchmark_result result;
    result.video_name     = gp.video_name;
    result.n_frames       = (int)frame_paths.size();
    result.adaptive_enabled = gp.adaptive_frames;
    result.spatial_cache_enabled = gp.spatial_cache;
    result.pyramid_enabled = gp.pyramid.enabled && !gp.baseline;
    result.pvc_min_ratio  = gp.pyramid.pvc_min_ratio;

    int64_t t_total_start = ggml_time_ms();

    // -----------------------------------------------------------
    // Phase 1: Load images
    // -----------------------------------------------------------
    mtmd::bitmaps bitmaps;
    for (const auto & path : frame_paths) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), path.c_str()));
        if (!bmp.ptr) {
            fprintf(stderr, "Failed to load image: %s\n", path.c_str());
            return result;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }
    printf("Loaded %zu frames\n", bitmaps.entries.size());

    // Load optical flow for motion-based pre-LLM scoring
    auto flows = load_optical_flows(gp.sintel_dir, gp.video_name, frame_numbers);
    int n_flows_loaded = 0;
    for (const auto & f : flows) { if (f.width > 0) n_flows_loaded++; }
    printf("Loaded %d/%zu optical flow files\n", n_flows_loaded, flows.size());

    // -----------------------------------------------------------
    // Phase 2: Build prompt with media markers and tokenize
    // -----------------------------------------------------------
    std::string full_content;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) {
        full_content += mtmd_default_marker();
    }
    full_content += "\n" + gp.prompt;

    common_chat_msg msg;
    msg.role    = "user";
    msg.content = full_content;

    bool add_bos = ctx.chat_history.empty();
    auto formatted = common_chat_format_single(
        ctx.tmpls.get(), ctx.chat_history, msg, true, false);
    ctx.chat_history.push_back(msg);

    printf("Formatted prompt: %zu chars\n", formatted.size());

    mtmd_input_text text_input;
    text_input.text          = formatted.c_str();
    text_input.add_special   = add_bos;
    text_input.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tok_res = mtmd_tokenize(ctx.ctx_vision.get(),
                                     chunks.ptr.get(),
                                     &text_input,
                                     bitmaps_c_ptr.data(),
                                     bitmaps_c_ptr.size());
    if (tok_res != 0) {
        fprintf(stderr, "Failed to tokenize prompt (error %d)\n", tok_res);
        return result;
    }

    // -----------------------------------------------------------
    // Phase 3: Process chunks (encode images + decode into KV cache)
    // -----------------------------------------------------------
    int64_t t_prefill_start = ggml_time_ms();

    visual_token_tracker tracker;
    // Qcur capture doesn't need tracker reference — scores are computed post-prefill via CPU Q@K^T

    // Embedding cache for spatial redundancy (Section 6e)
    embedding_cache emb_cache;
    mtmd::input_chunk_ptr keyframe_chunk_ptr(nullptr);

    // Keyframe embeddings for MOTION+KF scoring (Section 6h-1)
    std::vector<float> keyframe_embeddings;
    int keyframe_n_tok = 0;

    // Accumulate visual embeddings for diversity metrics
    std::vector<float> all_visual_embd;

    size_t n_chunks = chunks.size();
    int image_idx = 0;
    int n_kv_cells_used = 0;

    const llama_model * model = llama_get_model(ctx.lctx);
    int n_embd = llama_model_n_embd_inp(model);

    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        // ---- TEXT chunk ----
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            bool is_last_chunk = (ci == n_chunks - 1);

            // Single-pass eviction: capture Qcur from one layer during last text chunk prefill
            if (is_last_chunk && result.pyramid_enabled && !tracker.tokens.empty()) {
                capture_state.should_capture = true;
                printf("Capturing Qcur from layer %d for post-prefill eviction (flash attn ON)\n",
                       capture_state.capture_layer);
            }

            llama_pos new_n_past = ctx.n_past;
            int32_t ret = mtmd_helper_eval_chunk_single(
                ctx.ctx_vision.get(), ctx.lctx, chunk,
                ctx.n_past, 0, ctx.n_batch, is_last_chunk, &new_n_past);

            if (is_last_chunk) {
                capture_state.should_capture = false;
            }

            if (ret != 0) {
                fprintf(stderr, "Failed to evaluate text chunk %zu\n", ci);
                return result;
            }
            n_kv_cells_used += (int)(new_n_past - ctx.n_past);
            ctx.n_past = new_n_past;
            continue;
        }

        // ---- IMAGE chunk ----
        const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        if (!image_tokens) {
            fprintf(stderr, "Image tokens null for chunk %zu\n", ci);
            return result;
        }

        int chunk_nx = (int)mtmd_image_tokens_get_nx(image_tokens);
        int chunk_ny = (int)mtmd_image_tokens_get_ny(image_tokens);
        int n_tokens_chunk = chunk_nx * chunk_ny;

        // Record frame range
        int frame_token_start = (int)tracker.tokens.size();

        llama_pos frame_pos_start = ctx.n_past;
        llama_pos new_n_past = ctx.n_past;
        int actual_tokens_injected = n_tokens_chunk;

        // Determine spatial cache path for this frame
        bool use_spatial_cache = gp.spatial_cache && !gp.baseline;
        bool can_use_cache = use_spatial_cache && emb_cache.valid
                             && (chunk_nx == emb_cache.nx_full) && (chunk_ny == emb_cache.ny_full);
        bool has_sa = (image_idx < (int)spatial_analyses.size()
                       && !spatial_analyses[image_idx].patches.empty());

        if (image_idx == 0 || !can_use_cache) {
            // ====== PATH A: Full ViT encode (keyframe or no cache available) ======
            int64_t t_enc = ggml_time_ms();
            int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
            if (ret != 0) {
                fprintf(stderr, "Failed to encode frame %d\n", image_idx);
                return result;
            }
            result.encode_time_ms += (float)(ggml_time_ms() - t_enc);

            float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

            // Initialize / update embedding cache
            if (use_spatial_cache) {
                if (image_idx == 0) {
                    emb_cache.allocate(chunk_nx, chunk_ny, n_embd);
                    emb_cache.store_full(embd);
                    // Save keyframe chunk for re-use in decode_image_chunk
                    keyframe_chunk_ptr.reset(mtmd_input_chunk_copy(chunk));
                    printf("  Frame %d: KEYFRAME (%dx%d grid, %d tokens, encode %.1f ms) [cache initialized]\n",
                           image_idx, chunk_nx, chunk_ny, n_tokens_chunk,
                           (float)(ggml_time_ms() - t_enc));
                } else {
                    emb_cache.store_full(embd);
                    printf("  Frame %d: FULL ENCODE (no cache match, %dx%d, %d tokens, encode %.1f ms)\n",
                           image_idx, chunk_nx, chunk_ny, n_tokens_chunk,
                           (float)(ggml_time_ms() - t_enc));
                }
            }
            result.n_vit_full++;

            // Store keyframe embeddings for MOTION+KF scoring
            if (image_idx == 0) {
                keyframe_embeddings.assign(embd, embd + n_tokens_chunk * n_embd);
                keyframe_n_tok = n_tokens_chunk;
            }

            // Determine the embedding source for LLM decode
            const float * embd_src = use_spatial_cache ? emb_cache.embeddings.data() : embd;

            // Pre-LLM pruning for non-keyframe, non-baseline (MOTION+KF strategy)
            if (result.pyramid_enabled && image_idx > 0 && !gp.no_prellm) {
                float keep_ratio = 0.40f;

                const float * kf_ptr = (!keyframe_embeddings.empty() && keyframe_n_tok == n_tokens_chunk)
                                       ? keyframe_embeddings.data() : nullptr;
                const optical_flow & flow = (image_idx < (int)flows.size()) ? flows[image_idx] : flows[0];

                auto svt = score_and_select_motion_kf(
                    embd_src, kf_ptr, n_tokens_chunk, n_embd,
                    flow, chunk_nx, chunk_ny, keep_ratio);
                int n_keep = (int)svt.size();

                // Merge dropped tokens into nearest survivors
                auto merged = merge_dropped_tokens(
                    embd_src, svt, n_tokens_chunk, n_embd, chunk_nx, chunk_ny);

                int32_t dec_ret = inject_pruned_visual_tokens(
                    ctx.lctx, ctx.ctx_vision.get(),
                    embd_src, svt, chunk_nx, chunk_ny, n_embd,
                    frame_pos_start, 0, ctx.n_batch, &new_n_past,
                    merged.data());
                if (dec_ret != 0) {
                    fprintf(stderr, "Failed to decode pruned frame %d\n", image_idx);
                    return result;
                }

                actual_tokens_injected = n_keep;

                int frame_cell_start = n_kv_cells_used;
                for (int i = 0; i < n_keep; i++) {
                    int idx = svt[i].token_idx;
                    int ty = idx / chunk_nx, tx = idx % chunk_nx;
                    tracker.add_token(image_idx, ty, tx,
                                      frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                      frame_cell_start + i);
                }

                tracker.frame_ranges.push_back({frame_token_start, n_keep});
                tracker.n_frames++;
                n_kv_cells_used += n_keep;
                ctx.n_past = new_n_past;

                printf("  Frame %d: %dx%d grid, %d/%d tokens (MOTION+KF keep %.0f%%), n_past=%d\n",
                       image_idx, chunk_nx, chunk_ny, n_keep, n_tokens_chunk,
                       keep_ratio * 100, (int)ctx.n_past);
            } else {
                // Full injection (frame 0 or baseline or no-prellm)
                int32_t ret2;
                if (use_spatial_cache && keyframe_chunk_ptr) {
                    ret2 = mtmd_helper_decode_image_chunk(
                        ctx.ctx_vision.get(), ctx.lctx, keyframe_chunk_ptr.get(),
                        const_cast<float*>(embd_src), ctx.n_past, 0, ctx.n_batch, &new_n_past);
                } else {
                    ret2 = mtmd_helper_decode_image_chunk(
                        ctx.ctx_vision.get(), ctx.lctx, chunk,
                        embd, ctx.n_past, 0, ctx.n_batch, &new_n_past);
                }
                if (ret2 != 0) {
                    fprintf(stderr, "Failed to decode frame %d embeddings (n_past=%d)\n",
                            image_idx, (int)ctx.n_past);
                    return result;
                }

                int frame_cell_start = n_kv_cells_used;
                for (int ty = 0; ty < chunk_ny; ty++) {
                    for (int tx = 0; tx < chunk_nx; tx++) {
                        int local_idx = ty * chunk_nx + tx;
                        tracker.add_token(image_idx, ty, tx,
                                          frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                          frame_cell_start + local_idx);
                    }
                }

                tracker.frame_ranges.push_back({frame_token_start, n_tokens_chunk});
                if (image_idx == 0) tracker.tokens_per_frame = n_tokens_chunk;
                tracker.n_frames++;
                n_kv_cells_used += n_tokens_chunk;
                ctx.n_past = new_n_past;

                printf("  Frame %d: %dx%d grid, %d tokens (full), n_past=%d\n",
                       image_idx, chunk_nx, chunk_ny, n_tokens_chunk, (int)ctx.n_past);
            }

        } else if (can_use_cache && has_sa) {
            // ====== SPATIAL CACHE PATHS (frame > 0, cache valid, spatial analysis available) ======
            auto patch_active = build_patch_active_mask(
                spatial_analyses[image_idx], gp.patch_grid, chunk_nx, chunk_ny,
                gp.patch_reuse_motion);

            int n_active = 0;
            for (bool a : patch_active) { if (a) n_active++; }
            float active_ratio = (float)n_active / n_tokens_chunk;

            if (active_ratio < gp.skip_vit_threshold) {
                // ====== PATH B: Skip ViT entirely, reuse cached embeddings ======
                result.n_vit_skipped++;

                // Embeddings come from cache (unchanged)
                const float * embd_src = emb_cache.embeddings.data();

                if (result.pyramid_enabled && !gp.no_prellm) {
                    // Pre-LLM pruning on cached embeddings (MOTION+KF strategy)
                    float keep_ratio = 0.40f;

                    const float * kf_ptr = (!keyframe_embeddings.empty() && keyframe_n_tok == n_tokens_chunk)
                                           ? keyframe_embeddings.data() : nullptr;
                    const optical_flow & flow = (image_idx < (int)flows.size()) ? flows[image_idx] : flows[0];

                    auto all_scores = score_and_select_motion_kf(
                        embd_src, kf_ptr, n_tokens_chunk, n_embd,
                        flow, chunk_nx, chunk_ny, keep_ratio);
                    int n_keep = (int)all_scores.size();

                    // Merge dropped tokens into nearest survivors
                    auto merged = merge_dropped_tokens(
                        embd_src, all_scores, n_tokens_chunk, n_embd, chunk_nx, chunk_ny);

                    int32_t ret = inject_pruned_visual_tokens(
                        ctx.lctx, ctx.ctx_vision.get(),
                        embd_src, all_scores, chunk_nx, chunk_ny, n_embd,
                        frame_pos_start, 0, ctx.n_batch, &new_n_past,
                        merged.data());
                    if (ret != 0) {
                        fprintf(stderr, "Failed to decode skip-ViT pruned frame %d\n", image_idx);
                        return result;
                    }

                    actual_tokens_injected = n_keep;
                    int frame_cell_start = n_kv_cells_used;
                    for (int i = 0; i < n_keep; i++) {
                        int idx = all_scores[i].token_idx;
                        int ty = idx / chunk_nx, tx = idx % chunk_nx;
                        tracker.add_token(image_idx, ty, tx,
                                          frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                          frame_cell_start + i);
                    }
                    tracker.frame_ranges.push_back({frame_token_start, n_keep});
                    tracker.n_frames++;
                    n_kv_cells_used += n_keep;
                    ctx.n_past = new_n_past;

                    printf("  Frame %d: SKIP ViT (%.0f%% active < %.0f%% threshold), %d/%d tokens pruned, n_past=%d\n",
                           image_idx, active_ratio * 100.0f, gp.skip_vit_threshold * 100.0f,
                           n_keep, n_tokens_chunk, (int)ctx.n_past);
                } else {
                    // Decode all cached embeddings (no pruning)
                    int32_t ret = mtmd_helper_decode_image_chunk(
                        ctx.ctx_vision.get(), ctx.lctx, keyframe_chunk_ptr.get(),
                        const_cast<float*>(embd_src), ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    if (ret != 0) {
                        fprintf(stderr, "Failed to decode cached frame %d\n", image_idx);
                        return result;
                    }

                    int frame_cell_start = n_kv_cells_used;
                    for (int ty = 0; ty < chunk_ny; ty++)
                        for (int tx = 0; tx < chunk_nx; tx++)
                            tracker.add_token(image_idx, ty, tx,
                                              frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                              frame_cell_start + ty * chunk_nx + tx);

                    tracker.frame_ranges.push_back({frame_token_start, n_tokens_chunk});
                    tracker.n_frames++;
                    n_kv_cells_used += n_tokens_chunk;
                    ctx.n_past = new_n_past;

                    printf("  Frame %d: SKIP ViT (%.0f%% active < %.0f%% threshold), %d tokens cached, n_past=%d\n",
                           image_idx, active_ratio * 100.0f, gp.skip_vit_threshold * 100.0f,
                           n_tokens_chunk, (int)ctx.n_past);
                }

            } else {
                // ====== PATH C: Crop-encode active region, splice into cache ======
                result.n_vit_cropped++;

                // Find bounding box of active tokens
                int tx_min = chunk_nx, tx_max = -1;
                int ty_min = chunk_ny, ty_max = -1;
                for (int ty = 0; ty < chunk_ny; ty++) {
                    for (int tx = 0; tx < chunk_nx; tx++) {
                        if (patch_active[ty * chunk_nx + tx]) {
                            tx_min = std::min(tx_min, tx);
                            tx_max = std::max(tx_max, tx);
                            ty_min = std::min(ty_min, ty);
                            ty_max = std::max(ty_max, ty);
                        }
                    }
                }

                int bb_nx = tx_max - tx_min + 1;
                int bb_ny = ty_max - ty_min + 1;

                int img_w = (int)bitmaps.entries[image_idx].nx();
                int img_h = (int)bitmaps.entries[image_idx].ny();

                // Compute pixel crop from token bounding box
                float inv_sx = (float)img_w / (chunk_nx * 28);
                float inv_sy = (float)img_h / (chunk_ny * 28);
                int raw_cx = (int)(tx_min * 28 * inv_sx);
                int raw_cy = (int)(ty_min * 28 * inv_sy);
                int raw_cw = (int)(bb_nx * 28 * inv_sx);
                int raw_ch = (int)(bb_ny * 28 * inv_sy);

                raw_cx = std::max(0, std::min(raw_cx, img_w - 1));
                raw_cy = std::max(0, std::min(raw_cy, img_h - 1));
                raw_cw = std::min(raw_cw, img_w - raw_cx);
                raw_ch = std::min(raw_ch, img_h - raw_cy);

                frame_crop_info tmp_ci;
                tmp_ci.crop_x = raw_cx; tmp_ci.crop_y = raw_cy;
                tmp_ci.crop_w = raw_cw; tmp_ci.crop_h = raw_ch;
                mtmd::bitmap cropped_bmp = crop_bitmap(bitmaps.entries[image_idx], tmp_ci);

                // Tokenize the cropped image
                std::string crop_marker(mtmd_default_marker());
                mtmd_input_text crop_text;
                crop_text.text          = crop_marker.c_str();
                crop_text.add_special   = false;
                crop_text.parse_special = true;

                mtmd::input_chunks crop_chunks(mtmd_input_chunks_init());
                const mtmd_bitmap * crop_bmp_ptr = cropped_bmp.ptr.get();
                int32_t crop_tok_res = mtmd_tokenize(ctx.ctx_vision.get(),
                                                      crop_chunks.ptr.get(),
                                                      &crop_text,
                                                      &crop_bmp_ptr, 1);

                const mtmd_input_chunk * crop_chunk = nullptr;
                int crop_enc_nx = 0, crop_enc_ny = 0;
                if (crop_tok_res == 0) {
                    for (size_t k = 0; k < crop_chunks.size(); k++) {
                        if (mtmd_input_chunk_get_type(crop_chunks[k]) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                            crop_chunk = crop_chunks[k];
                            break;
                        }
                    }
                    if (crop_chunk) {
                        const auto cit = mtmd_input_chunk_get_tokens_image(crop_chunk);
                        crop_enc_nx = (int)mtmd_image_tokens_get_nx(cit);
                        crop_enc_ny = (int)mtmd_image_tokens_get_ny(cit);
                    }
                }

                bool crop_ok = crop_chunk && (crop_enc_nx == bb_nx) && (crop_enc_ny == bb_ny);

                if (crop_ok) {
                    // Crop-encode the active region
                    int64_t t_enc = ggml_time_ms();
                    int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), crop_chunk);
                    if (ret != 0) {
                        fprintf(stderr, "Failed to encode crop for frame %d\n", image_idx);
                        return result;
                    }
                    result.encode_time_ms += (float)(ggml_time_ms() - t_enc);

                    float * crop_embd = mtmd_get_output_embd(ctx.ctx_vision.get());

                    // Splice: update active patches in cache, keep static ones
                    int n_updated = 0, n_kept_in_bb = 0;
                    for (int cy = 0; cy < bb_ny; cy++) {
                        for (int cx = 0; cx < bb_nx; cx++) {
                            int full_tx = tx_min + cx;
                            int full_ty = ty_min + cy;
                            int full_idx = full_ty * chunk_nx + full_tx;
                            if (patch_active[full_idx]) {
                                memcpy(emb_cache.at(full_ty, full_tx),
                                       crop_embd + (cy * bb_nx + cx) * n_embd,
                                       n_embd * sizeof(float));
                                n_updated++;
                            } else {
                                n_kept_in_bb++;
                            }
                        }
                    }
                    emb_cache.valid = true;

                    printf("  Frame %d: CROP+SPLICE (%d/%d updated, crop %dx%d, %.0f%% active, encode %.1f ms)\n",
                           image_idx, n_updated, n_tokens_chunk, bb_nx, bb_ny,
                           active_ratio * 100.0f, (float)(ggml_time_ms() - t_enc));
                } else {
                    // Fallback: full encode when crop dimensions don't match
                    int64_t t_enc = ggml_time_ms();
                    int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
                    if (ret != 0) {
                        fprintf(stderr, "Failed to encode frame %d (crop fallback)\n", image_idx);
                        return result;
                    }
                    result.encode_time_ms += (float)(ggml_time_ms() - t_enc);

                    float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
                    int n_updated = 0, n_kept = 0;
                    emb_cache.selective_update(embd, patch_active, n_updated, n_kept);

                    printf("  Frame %d: CROP FALLBACK (dim mismatch %dx%d vs %dx%d, full encode, %d/%d updated)\n",
                           image_idx, crop_enc_nx, crop_enc_ny, bb_nx, bb_ny,
                           n_updated, n_tokens_chunk);
                }

                // Decode from cache (spliced embeddings) with optional pre-LLM pruning
                const float * embd_src = emb_cache.embeddings.data();

                if (result.pyramid_enabled && !gp.no_prellm) {
                    float keep_ratio = 0.40f;

                    const float * kf_ptr = (!keyframe_embeddings.empty() && keyframe_n_tok == n_tokens_chunk)
                                           ? keyframe_embeddings.data() : nullptr;
                    const optical_flow & flow = (image_idx < (int)flows.size()) ? flows[image_idx] : flows[0];

                    auto all_scores = score_and_select_motion_kf(
                        embd_src, kf_ptr, n_tokens_chunk, n_embd,
                        flow, chunk_nx, chunk_ny, keep_ratio);
                    int n_keep = (int)all_scores.size();

                    // Merge dropped tokens into nearest survivors
                    auto merged = merge_dropped_tokens(
                        embd_src, all_scores, n_tokens_chunk, n_embd, chunk_nx, chunk_ny);

                    int32_t ret = inject_pruned_visual_tokens(
                        ctx.lctx, ctx.ctx_vision.get(),
                        embd_src, all_scores, chunk_nx, chunk_ny, n_embd,
                        frame_pos_start, 0, ctx.n_batch, &new_n_past,
                        merged.data());
                    if (ret != 0) {
                        fprintf(stderr, "Failed to decode crop-spliced pruned frame %d\n", image_idx);
                        return result;
                    }

                    actual_tokens_injected = n_keep;
                    int frame_cell_start = n_kv_cells_used;
                    for (int i = 0; i < n_keep; i++) {
                        int idx = all_scores[i].token_idx;
                        int ty = idx / chunk_nx, tx = idx % chunk_nx;
                        tracker.add_token(image_idx, ty, tx,
                                          frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                          frame_cell_start + i);
                    }
                    tracker.frame_ranges.push_back({frame_token_start, n_keep});
                    tracker.n_frames++;
                    n_kv_cells_used += n_keep;
                    ctx.n_past = new_n_past;

                    printf("    -> %d/%d tokens after pre-LLM pruning, n_past=%d\n",
                           n_keep, n_tokens_chunk, (int)ctx.n_past);
                } else {
                    int32_t ret = mtmd_helper_decode_image_chunk(
                        ctx.ctx_vision.get(), ctx.lctx, keyframe_chunk_ptr.get(),
                        const_cast<float*>(embd_src), ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    if (ret != 0) {
                        fprintf(stderr, "Failed to decode spliced frame %d\n", image_idx);
                        return result;
                    }

                    int frame_cell_start = n_kv_cells_used;
                    for (int ty = 0; ty < chunk_ny; ty++)
                        for (int tx = 0; tx < chunk_nx; tx++)
                            tracker.add_token(image_idx, ty, tx,
                                              frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                              frame_cell_start + ty * chunk_nx + tx);

                    tracker.frame_ranges.push_back({frame_token_start, n_tokens_chunk});
                    tracker.n_frames++;
                    n_kv_cells_used += n_tokens_chunk;
                    ctx.n_past = new_n_past;

                    printf("    -> %d tokens (full cache decode), n_past=%d\n",
                           n_tokens_chunk, (int)ctx.n_past);
                }
            }
        } else {
            // ====== PATH D: Cache valid but no spatial analysis — full encode ======
            int64_t t_enc = ggml_time_ms();
            int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
            if (ret != 0) {
                fprintf(stderr, "Failed to encode frame %d (path D)\n", image_idx);
                return result;
            }
            result.encode_time_ms += (float)(ggml_time_ms() - t_enc);
            result.n_vit_full++;

            float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
            if (use_spatial_cache) emb_cache.store_full(embd);

            ret = mtmd_helper_decode_image_chunk(
                ctx.ctx_vision.get(), ctx.lctx,
                (use_spatial_cache && keyframe_chunk_ptr) ? keyframe_chunk_ptr.get() : chunk,
                (use_spatial_cache ? emb_cache.embeddings.data() : embd),
                ctx.n_past, 0, ctx.n_batch, &new_n_past);
            if (ret != 0) {
                fprintf(stderr, "Failed to decode frame %d (path D)\n", image_idx);
                return result;
            }

            int frame_cell_start = n_kv_cells_used;
            for (int ty = 0; ty < chunk_ny; ty++)
                for (int tx = 0; tx < chunk_nx; tx++)
                    tracker.add_token(image_idx, ty, tx,
                                      frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                      frame_cell_start + ty * chunk_nx + tx);

            tracker.frame_ranges.push_back({frame_token_start, n_tokens_chunk});
            tracker.n_frames++;
            n_kv_cells_used += n_tokens_chunk;
            ctx.n_past = new_n_past;

            printf("  Frame %d: %dx%d grid, %d tokens (full, no SA), n_past=%d\n",
                   image_idx, chunk_nx, chunk_ny, n_tokens_chunk, (int)ctx.n_past);
        }

        // Accumulate ViT embeddings for diversity metrics
        // Use the ViT output (embd) for frames that were encoded,
        // or the embedding cache for frames that skipped ViT
        {
            float * frame_embd = mtmd_get_output_embd(ctx.ctx_vision.get());
            const float * src = nullptr;
            int n_to_store = 0;

            if (use_spatial_cache && emb_cache.valid) {
                // Spatial cache path: use cache contents (includes spliced updates)
                src = emb_cache.embeddings.data();
                n_to_store = emb_cache.nx_full * emb_cache.ny_full;
            } else if (frame_embd) {
                // Direct ViT output
                src = frame_embd;
                n_to_store = n_tokens_chunk;
            }

            if (src && n_to_store > 0) {
                size_t old_size = all_visual_embd.size();
                all_visual_embd.resize(old_size + n_to_store * n_embd);
                memcpy(all_visual_embd.data() + old_size, src, n_to_store * n_embd * sizeof(float));
            }
        }

        image_idx++;
    }

    result.n_visual_tokens = (int)tracker.tokens.size();
    result.prefill_time_ms = (float)(ggml_time_ms() - t_prefill_start);

    // Save visual embeddings to binary file for exact SVD computation in Python
    if (!all_visual_embd.empty() && n_embd > 0 && !gp.output_json.empty()) {
        int n_vis = (int)(all_visual_embd.size() / n_embd);
        std::string embd_path = gp.output_json;
        // Replace .json with .embd
        size_t dot_pos = embd_path.rfind('.');
        if (dot_pos != std::string::npos) embd_path = embd_path.substr(0, dot_pos);
        embd_path += ".embd";

        FILE * ef = fopen(embd_path.c_str(), "wb");
        if (ef) {
            // Header: n_tokens (int32), n_embd (int32)
            int32_t header[2] = {(int32_t)n_vis, (int32_t)n_embd};
            fwrite(header, sizeof(int32_t), 2, ef);
            // Data: float32 [n_vis * n_embd]
            fwrite(all_visual_embd.data(), sizeof(float), all_visual_embd.size(), ef);
            fclose(ef);
            printf("  Saved %d visual embeddings (%d dim) to %s (%.1f MB)\n",
                   n_vis, n_embd, embd_path.c_str(),
                   (float)(all_visual_embd.size() * sizeof(float)) / 1e6f);
        }
    }

    // Compute ViT savings
    int n_total_enc = result.n_vit_full + result.n_vit_skipped + result.n_vit_cropped;
    if (n_total_enc > 0) {
        result.vit_savings_pct = 100.0f * (result.n_vit_skipped + result.n_vit_cropped * 0.5f) / n_total_enc;
    }

    printf("\nPrefill complete: %d visual tokens, %d KV cells, %.1f ms\n",
           result.n_visual_tokens, n_kv_cells_used, result.prefill_time_ms);
    if (gp.spatial_cache) {
        printf("  ViT: %d full, %d skipped, %d cropped (%.0f%% savings)\n",
               result.n_vit_full, result.n_vit_skipped, result.n_vit_cropped, result.vit_savings_pct);
    }

    // -----------------------------------------------------------
    // Phase 4: Single-pass post-prefill KV eviction
    //
    // Captures Qcur from one layer (default: 27) during last text chunk prefill.
    // Computes Q@K^T once on CPU, evicts to 21% retention in a single pass.
    // Validated: all layers produce identical token rankings (Spearman=1.0),
    // so one Q@K^T pass is equivalent to the three-stage cascade.
    // -----------------------------------------------------------
    if (result.pyramid_enabled && !tracker.tokens.empty()) {
        int64_t t_evict_start = ggml_time_ms();

        auto * mem = llama_get_memory(ctx.lctx);
        int n_total = (int)tracker.tokens.size();
        int tpf = tracker.tokens_per_frame;
        int n_head_kv = llama_model_n_head_kv(model);

        // Single Q@K^T computation using captured Qcur
        float keep_ratio = gp.pyramid.pvc_min_ratio;  // default 0.21 (same net as 0.7*0.6*0.5)

        printf("\n=== Single-Pass KV Eviction (layer %d Q@K^T, flash attn ON) ===\n",
               capture_state.capture_layer);
        printf("Initial: %d visual tokens, target: keep %.0f%% (frame 0 protected: %d)\n",
               n_total, keep_ratio * 100, tpf);

        std::vector<float> scores;
        int64_t t_qk = ggml_time_ms();
        compute_qk_attention_scores(
            capture_state.captured, capture_state.capture_layer, mem,
            tracker, n_kv_cells_used, n_head_kv, scores);
        float qk_ms = (float)(ggml_time_ms() - t_qk);

        if (!scores.empty()) {
            // Collect all tokens with scores
            struct scored_tok { int idx; float score; };
            std::vector<scored_tok> active;
            for (int i = 0; i < n_total; i++) {
                if (!tracker.tokens[i].evicted) {
                    float score = (i < (int)scores.size()) ? scores[i] : 0.0f;
                    if (gp.pyramid.first_frame_protect && i < tpf) score = FLT_MAX;
                    active.push_back({i, score});
                }
            }

            int n_active = (int)active.size();
            int n_keep = std::max(1, (int)(n_active * keep_ratio));
            int n_to_evict = n_active - n_keep;

            if (n_to_evict > 0) {
                std::sort(active.begin(), active.end(),
                          [](const scored_tok & a, const scored_tok & b) { return a.score < b.score; });

                int n_evicted = 0;
                for (int i = 0; i < n_to_evict; i++) {
                    auto & tok = tracker.tokens[active[i].idx];
                    if (tok.evicted) continue;
                    bool ok = llama_memory_seq_rm_pos_ext(mem, 0, tok.kv_pos, tok.ext_x, tok.ext_y);
                    if (ok) { tok.evicted = true; n_evicted++; }
                }

                printf("  %d -> evict %d -> %d remain (Q@K: %.0f ms)\n",
                       n_active, n_evicted, n_active - n_evicted, qk_ms);
            } else {
                printf("  Nothing to evict (Q@K: %.0f ms)\n", qk_ms);
            }
        } else {
            printf("  Qcur not captured, skipping eviction\n");
        }

        result.n_visual_tokens_kept = tracker.count_active();
        result.eviction_time_ms = (float)(ggml_time_ms() - t_evict_start);

        printf("  Result: %d -> %d visual tokens (%.1f%% retained, %.0f ms total)\n",
               n_total, result.n_visual_tokens_kept,
               100.0f * result.n_visual_tokens_kept / n_total,
               result.eviction_time_ms);
        printf("============================================================\n\n");
    } else {
        result.n_visual_tokens_kept = result.n_visual_tokens;
    }

    // -----------------------------------------------------------
    // Phase 5: Text generation
    // -----------------------------------------------------------
    capture_state.should_capture = false;
    capture_state.disabled = true;

    int64_t t_gen_start = ggml_time_ms();
    std::string generated_text;
    int n_generated = 0;
    // Disable defrag: M-RoPE extended positions can be corrupted during cell compaction,
    // causing backend_res == nullptr assertion failures in llama_decode.
    // KV cache holes from eviction are tolerable — they waste some memory bandwidth
    // but do not affect correctness.
    bool needs_defrag = false;

    printf("--- Generation (max %d tokens) ---\n", gp.n_predict);

    for (int i = 0; i < gp.n_predict; i++) {
        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (!gp.ignore_eos && llama_vocab_is_eog(ctx.vocab, token_id)) {
            break;
        }

        std::string piece = common_token_to_piece(ctx.lctx, token_id);
        generated_text += piece;
        printf("%s", piece.c_str());
        fflush(stdout);

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            fprintf(stderr, "\nFailed to decode token %d\n", i);
            break;
        }

        // Deferred defrag: compact KV cache after first decode
        // (doing it between prefill and first sample invalidates cached logits)
        if (i == 0 && needs_defrag) {
            // Defrag: compact KV cache to eliminate holes from eviction.
            // The mv() function in llama-kv-cells correctly copies pos + ext (M-RoPE x,y).
            printf("[defrag] Compacting KV cache after first decode...\n");
            int64_t t_defrag = ggml_time_ms();
            llama_memory_defragment(llama_get_memory(ctx.lctx), ctx.lctx);
            printf("[defrag] Done in %.1f ms\n", (float)(ggml_time_ms() - t_defrag));
            needs_defrag = false;
        }

        n_generated++;
    }
    printf("\n");

    result.n_tokens_generated = n_generated;
    result.generation_time_ms = (float)(ggml_time_ms() - t_gen_start);
    result.generated_text     = generated_text;
    result.total_time_ms      = (float)(ggml_time_ms() - t_total_start);
    result.tokens_per_sec     = n_generated / (result.generation_time_ms / 1000.0f);

    printf("\n=== Timing Summary ===\n");
    if (result.selection_time_ms > 0) {
        printf("Frame select:  %8.1f ms\n", result.selection_time_ms);
    }
    printf("ViT encode:    %8.1f ms\n", result.encode_time_ms);
    if (result.spatial_cache_enabled) {
        printf("  ViT: %d full, %d skip, %d crop (%.0f%% savings)\n",
               result.n_vit_full, result.n_vit_skipped, result.n_vit_cropped, result.vit_savings_pct);
    }
    printf("Prefill:       %8.1f ms\n", result.prefill_time_ms);
    printf("KV eviction:   %8.1f ms\n", result.eviction_time_ms);
    printf("Generation:    %8.1f ms (%d tokens, %.1f t/s)\n",
           result.generation_time_ms, n_generated, result.tokens_per_sec);
    printf("Total:         %8.1f ms\n", result.total_time_ms);
    printf("======================\n");

    return result;
}

// ============================================================================
// Section 9: Main
// ============================================================================

int main(int argc, char ** argv) {
    gen_params gp;
    auto filtered_args = filter_custom_args(argc, argv, gp);

    // Re-create argc/argv for common_params_parse
    int new_argc = (int)filtered_args.size();
    char ** new_argv = filtered_args.data();

    common_params params;
    if (!common_params_parse(new_argc, new_argv, params, LLAMA_EXAMPLE_MTMD)) {
        fprintf(stderr, "Failed to parse arguments\n");
        return 1;
    }

    common_init();

    // Default model paths if not specified
    if (params.model.path.empty()) {
        params.model.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf";
    }
    if (params.mmproj.path.empty()) {
        params.mmproj.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf";
    }

    // Override n_predict if specified via our custom arg
    if (params.n_predict < 0) {
        params.n_predict = gp.n_predict;
    }

    // Set up flash-compatible Qcur capture for three-stage scoring
    multi_qcur_capture_state capture_state = {};

    // Register cb_eval callback — captures Qcur (NOT kq_soft_max), works with flash attn ON
    if (gp.pyramid.enabled && !gp.baseline) {
        params.cb_eval = multi_qcur_callback;
        params.cb_eval_user_data = &capture_state;
        // Flash attention stays ENABLED — Qcur capture is compatible
        printf("Pyramid enabled (single-pass Qcur capture at layer %d, flash attn ON)\n",
               capture_state.capture_layer);
    } else {
        printf("Pyramid disabled (baseline mode)\n");
    }

    // Initialize VLM context
    pyramid_vlm_context ctx(params);

    // -----------------------------------------------------------
    // Frame selection: adaptive (cumulative motion) or uniform
    // -----------------------------------------------------------
    std::vector<std::string> frame_paths;
    std::vector<int> frame_numbers;
    std::vector<frame_spatial_analysis> spatial_analyses;
    int n_total_video_frames = 0;
    float selection_time_ms = 0.0f;
    float sel_global_complexity = 0.0f;
    float sel_motion_coverage = 0.0f;

    if (gp.adaptive_frames && !gp.baseline) {
        // Adaptive frame selection via cumulative motion thresholding
        n_total_video_frames = count_video_frames(gp.sintel_dir, gp.video_name);
        if (n_total_video_frames == 0) {
            fprintf(stderr, "No frames found for video '%s'\n", gp.video_name.c_str());
            return 1;
        }

        int64_t t_sel = ggml_time_ms();
        std::string flow_dir = gp.sintel_dir + "/training/flow";

        online_frame_selector selector;
        selector.init(gp.base_frames, gp.min_selected_frames, gp.max_selected_frames,
                      gp.motion_threshold, gp.motion_percentile,
                      gp.selection_percentile, gp.complexity_floor);
        selector.run(flow_dir, gp.video_name, n_total_video_frames);

        selection_time_ms = (float)(ggml_time_ms() - t_sel);
        sel_global_complexity = selector.global_complexity;
        sel_motion_coverage = selector.motion_coverage;

        if (selector.selected_frames.empty()) {
            fprintf(stderr, "No frames selected by adaptive selector\n");
            return 1;
        }

        // Build frame paths from selected frames
        std::string frame_dir = gp.sintel_dir + "/training/final/" + gp.video_name;
        for (const auto & sf : selector.selected_frames) {
            char buf[64];
            snprintf(buf, sizeof(buf), "/frame_%04d.png", sf.frame_index);
            frame_paths.push_back(frame_dir + buf);
            frame_numbers.push_back(sf.frame_index);
        }

        // Compute spatial analysis for each selected frame (if spatial cache enabled)
        if (gp.spatial_cache) {
            for (const auto & sf : selector.selected_frames) {
                char buf[64];
                snprintf(buf, sizeof(buf), "/frame_%04d.flo", sf.frame_index);
                std::string flo_path = flow_dir + "/" + gp.video_name + buf;

                optical_flow flow;
                if (read_flo_file(flo_path, flow) && flow.width > 0) {
                    auto sa = analyze_spatial_patches(flow, sf.frame_index,
                                                      gp.patch_grid, gp.spatial_threshold,
                                                      gp.motion_threshold);
                    spatial_analyses.push_back(sa);
                } else {
                    // No flow for this frame (e.g., first frame)
                    frame_spatial_analysis empty_sa;
                    empty_sa.frame_index = sf.frame_index;
                    empty_sa.n_total_patches = gp.patch_grid * gp.patch_grid;
                    empty_sa.n_selected_patches = empty_sa.n_total_patches;
                    empty_sa.spatial_savings = 0.0f;
                    spatial_analyses.push_back(empty_sa);
                }
            }
        }

        printf("Adaptive selection: %zu frames from %d total (%.1f ms)\n",
               frame_paths.size(), n_total_video_frames, selection_time_ms);
    } else {
        // Uniform frame selection (existing behavior)
        frame_paths = load_frame_paths(gp.sintel_dir, gp.video_name, gp.n_frames, frame_numbers);
        if (frame_paths.empty()) {
            fprintf(stderr, "No frames found\n");
            return 1;
        }
        n_total_video_frames = gp.n_frames;

        // Compute spatial analysis if spatial cache enabled (even with uniform selection)
        if (gp.spatial_cache && !gp.baseline) {
            std::string flow_dir = gp.sintel_dir + "/training/flow";
            for (int fn : frame_numbers) {
                char buf[64];
                snprintf(buf, sizeof(buf), "/frame_%04d.flo", fn);
                std::string flo_path = flow_dir + "/" + gp.video_name + buf;

                optical_flow flow;
                if (read_flo_file(flo_path, flow) && flow.width > 0) {
                    spatial_analyses.push_back(analyze_spatial_patches(
                        flow, fn, gp.patch_grid, gp.spatial_threshold, gp.motion_threshold));
                } else {
                    frame_spatial_analysis empty_sa;
                    empty_sa.frame_index = fn;
                    empty_sa.n_total_patches = gp.patch_grid * gp.patch_grid;
                    empty_sa.n_selected_patches = empty_sa.n_total_patches;
                    empty_sa.spatial_savings = 0.0f;
                    spatial_analyses.push_back(empty_sa);
                }
            }
        }
    }

    // Print configuration
    printf("\n=== Configuration ===\n");
    printf("Video: %s (%d total, %zu selected)\n",
           gp.video_name.c_str(), n_total_video_frames, frame_paths.size());
    printf("Model: %s\n", params.model.path.c_str());
    printf("Frame selection: %s\n", (gp.adaptive_frames && !gp.baseline) ? "ADAPTIVE" : "UNIFORM");
    if (gp.adaptive_frames && !gp.baseline) {
        printf("  Base frames: %d, range: [%d, %d], selection pct: %.0f%%\n",
               gp.base_frames, gp.min_selected_frames, gp.max_selected_frames,
               gp.selection_percentile);
        printf("  Complexity floor: %.2f (post-selection filter)\n", gp.complexity_floor);
    }
    printf("Spatial cache: %s\n", (gp.spatial_cache && !gp.baseline) ? "ENABLED" : "DISABLED");
    if (gp.spatial_cache && !gp.baseline) {
        printf("  Patch grid: %dx%d, skip threshold: %.0f%%, spatial threshold: %.2f\n",
               gp.patch_grid, gp.patch_grid, gp.skip_vit_threshold * 100, gp.spatial_threshold);
    }
    printf("Pyramid: %s\n", (gp.pyramid.enabled && !gp.baseline) ? "ENABLED" : "DISABLED");
    if (gp.pyramid.enabled && !gp.baseline) {
        printf("  Single-pass eviction: layer %d Q@K^T, keep %.0f%%\n",
               capture_state.capture_layer, gp.pyramid.pvc_min_ratio * 100);
        printf("  Pre-LLM pruning: %s\n", gp.no_prellm ? "DISABLED" : "enabled (motion+cosine+norm)");
        printf("  Frame 0 protect: %s\n", gp.pyramid.first_frame_protect ? "yes" : "no");
    }
    printf("Max generation: %d tokens\n", gp.n_predict);
    printf("=====================\n\n");

    // Run inference
    benchmark_result result = run_inference(ctx, gp, frame_paths, frame_numbers, spatial_analyses, capture_state);
    result.n_frames_total = n_total_video_frames;
    result.selection_time_ms = selection_time_ms;
    result.global_complexity = sel_global_complexity;
    result.motion_coverage = sel_motion_coverage;

    // Save results
    if (!gp.output_json.empty()) {
        json j = result.to_json();
        std::ofstream ofs(gp.output_json);
        ofs << j.dump(2) << std::endl;
        printf("Results saved to %s\n", gp.output_json.c_str());
    }

    return 0;
}
