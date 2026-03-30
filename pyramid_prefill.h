#pragma once

// Pyramid KV Prefill — RTX 4090 Architecture (Architecture B)
//
// Layer-wise visual token dropping for Qwen2.5-VL video inference.
// Uses attention weights from "scout" layers (0-1) to score visual tokens,
// then drops low-importance tokens via -INF injection into KQ attention mask.
// Retention follows a cosine schedule: 100% at layer 0 → PVC_MIN at last layer.

#include <cmath>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Configuration constants (runtime-overridable via CLI)
// ============================================================================

struct pyramid_config {
    int   clie_layer            = 2;      // scout uses layers 0..clie_layer-1
    float pvc_min_ratio         = 0.20f;  // deepest layer keeps 20%
    float pvc_max_ratio         = 1.00f;  // shallowest keeps 100%
    bool  first_frame_protect   = true;   // always keep frame 0 (PureKV anchor)
    float temporal_sim_thresh   = 0.95f;  // inter-frame cosine similarity threshold
    bool  enabled               = true;   // master switch
};

// ============================================================================
// Visual token position tracking
// ============================================================================

struct visual_token_range {
    int seq_id          = 0;
    int start_pos       = 0;    // llama position of first visual token
    int n_tokens        = 0;    // total visual token count
    int n_frames        = 0;    // number of video frames
    int tokens_per_frame = 0;   // n_tokens / n_frames
    int kv_cell_start   = 0;    // KV cache cell index of first visual token
};

// ============================================================================
// Pyramid GPU state (persistent across calls)
// ============================================================================

struct pyramid_gpu_state {
    float * d_scores          = nullptr;  // [n_visual], device
    float * d_scores_sorted   = nullptr;  // [n_visual], scratch for CUB sort
    float * d_thresholds      = nullptr;  // [n_layers], device
    void  * d_cub_temp        = nullptr;
    size_t  cub_temp_bytes    = 0;
    int     visual_start_pos  = 0;
    int     n_visual          = 0;
    int     n_layers          = 0;
    bool    initialized       = false;

    void cleanup() {
        // Cleaned up by caller with cudaFree if CUDA is available
        d_scores = nullptr;
        d_scores_sorted = nullptr;
        d_thresholds = nullptr;
        d_cub_temp = nullptr;
        initialized = false;
    }
};

// ============================================================================
// Cosine retention schedule
// ============================================================================

// Returns the fraction of visual tokens to retain at layer `il` out of `n_layers` total.
// Layers 0..clie_layer-1 always return 1.0 (scout layers, no dropping).
static inline float pvc_ratio(int il, int n_layers, const pyramid_config & cfg) {
    if (il < cfg.clie_layer) return 1.0f;
    return cfg.pvc_min_ratio + (cfg.pvc_max_ratio - cfg.pvc_min_ratio)
        * 0.5f * (1.0f + cosf((float)M_PI * (float)il / (float)(n_layers - 1)));
}

// ============================================================================
// CPU-side importance scoring and threshold computation (Architecture A path)
// ============================================================================

// Reduce attention weights [n_kv, n_heads_kv, n_q] to per-visual-token importance [n_visual]
static inline void compute_importance_scores_cpu(
    const float * attn_data,    // [n_kv, n_heads_kv, n_q] row-major (ggml layout)
    int n_kv, int n_heads_kv, int n_q,
    const visual_token_range & vrange,
    const pyramid_config & cfg,
    std::vector<float> & importance)   // output: [n_visual]
{
    importance.assign(vrange.n_tokens, 0.0f);

    for (int q = 0; q < n_q; q++) {
        for (int h = 0; h < n_heads_kv; h++) {
            for (int v = 0; v < vrange.n_tokens; v++) {
                int kv_pos = vrange.kv_cell_start + v;
                // ggml softmax output layout: [n_kv, n_heads, n_q, 1]
                // Index: h * n_q * n_kv + q * n_kv + kv_pos
                // But actual layout depends on permutation. After ggml_soft_max_ext,
                // the tensor is [n_kv, n_heads_kv, n_q] with strides [1, n_kv, n_kv*n_heads_kv]
                importance[v] += attn_data[(h * n_q + q) * n_kv + kv_pos];
            }
        }
    }

    // Protect frame 0: set importance to FLT_MAX so they're never dropped
    if (cfg.first_frame_protect && vrange.tokens_per_frame > 0) {
        for (int v = 0; v < vrange.tokens_per_frame && v < vrange.n_tokens; v++) {
            importance[v] = FLT_MAX;
        }
    }
}

// Compute per-layer thresholds using std::nth_element (O(n) average)
static inline void compute_layer_thresholds_cpu(
    const std::vector<float> & importance,
    int n_layers,
    const pyramid_config & cfg,
    std::vector<float> & thresholds)  // output: [n_layers]
{
    int n_visual = (int)importance.size();
    thresholds.resize(n_layers);

    std::vector<float> scratch = importance;

    for (int il = 0; il < n_layers; il++) {
        if (il < cfg.clie_layer) {
            thresholds[il] = -FLT_MAX;  // no dropping in scout layers
            continue;
        }

        float ratio = pvc_ratio(il, n_layers, cfg);
        int k = std::max(1, (int)(ratio * n_visual));

        // Partition so that scratch[n_visual - k] is the kth-largest
        scratch = importance;  // reset
        std::nth_element(scratch.begin(),
                         scratch.begin() + (n_visual - k),
                         scratch.end());
        thresholds[il] = scratch[n_visual - k];
    }
}

// Build per-layer drop masks: 0.0 for kept, -INF for dropped
// mask layout: [n_kv * n_q] per layer (flattened row-major)
static inline void build_drop_masks_cpu(
    const std::vector<float> & importance,
    const std::vector<float> & thresholds,
    const visual_token_range & vrange,
    int n_kv, int n_q, int n_layers,
    const pyramid_config & cfg,
    std::vector<std::vector<float>> & drop_masks)  // output: [n_layers][n_kv * n_q]
{
    drop_masks.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        if (il < cfg.clie_layer) {
            drop_masks[il].clear();  // no mask for scout layers
            continue;
        }

        drop_masks[il].assign((size_t)n_kv * n_q, 0.0f);

        for (int v = 0; v < vrange.n_tokens; v++) {
            if (importance[v] < thresholds[il]) {
                int kv_pos = vrange.kv_cell_start + v;
                for (int q = 0; q < n_q; q++) {
                    drop_masks[il][(size_t)q * n_kv + kv_pos] = -INFINITY;
                }
            }
        }
    }
}

// ============================================================================
// Post-prefill KV eviction (pyramid-guided)
// ============================================================================

// Returns indices of visual tokens to evict (importance below threshold at deepest layer)
static inline std::vector<int> compute_eviction_set(
    const std::vector<float> & importance,
    const visual_token_range & vrange,
    const pyramid_config & cfg,
    int n_layers)
{
    // Use the deepest layer's retention ratio for eviction
    float ratio = cfg.pvc_min_ratio;
    int n_visual = vrange.n_tokens;
    int k = std::max(1, (int)(ratio * n_visual));

    std::vector<float> scratch = importance;
    std::nth_element(scratch.begin(),
                     scratch.begin() + (n_visual - k),
                     scratch.end());
    float threshold = scratch[n_visual - k];

    std::vector<int> evict_indices;
    for (int v = 0; v < n_visual; v++) {
        if (importance[v] < threshold) {
            evict_indices.push_back(v);
        }
    }
    return evict_indices;
}

// ============================================================================
// CLI argument parsing
// ============================================================================

static inline void parse_pyramid_args(int argc, char ** argv, pyramid_config & cfg) {
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--pyramid-clie-layer" && i + 1 < argc) {
            cfg.clie_layer = std::stoi(argv[++i]);
        } else if (arg == "--pyramid-pvc-min" && i + 1 < argc) {
            cfg.pvc_min_ratio = std::stof(argv[++i]);
        } else if (arg == "--pyramid-temporal-thresh" && i + 1 < argc) {
            cfg.temporal_sim_thresh = std::stof(argv[++i]);
        } else if (arg == "--no-pyramid-protect-frame0") {
            cfg.first_frame_protect = false;
        } else if (arg == "--no-pyramid") {
            cfg.enabled = false;
        }
    }
}

// ============================================================================
// Logging helpers
// ============================================================================

static inline void print_pyramid_schedule(int n_layers, const pyramid_config & cfg) {
    printf("\n=== Pyramid KV Prefill Schedule ===\n");
    printf("Scout layers: 0-%d (full attention)\n", cfg.clie_layer - 1);
    printf("Min retention: %.0f%%  Max retention: %.0f%%\n",
           cfg.pvc_min_ratio * 100, cfg.pvc_max_ratio * 100);
    printf("Frame 0 protected: %s\n", cfg.first_frame_protect ? "yes" : "no");
    printf("\nLayer retention ratios:\n");
    for (int il = 0; il < n_layers; il++) {
        float r = pvc_ratio(il, n_layers, cfg);
        printf("  Layer %2d: %5.1f%%", il, r * 100);
        if (il < cfg.clie_layer) printf(" (scout)");
        printf("\n");
    }
    printf("===================================\n\n");
}
