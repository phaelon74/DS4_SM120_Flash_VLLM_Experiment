#pragma once

#include <torch/python.h>

namespace deep_gemm {

void sm120_fp8_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim);

void sm120_fp4_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv,
    const torch::Tensor& kv_sf, const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim);

void sm120_fp8_paged_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& kv_cache,
    const torch::Tensor& kv_cache_sf, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& logits,
    const torch::Tensor& block_table, const at::ScalarType& logits_dtype,
    int batch_size, int next_n, int num_heads, int head_dim, int block_kv,
    bool is_context_lens_2d, int logits_stride, int block_table_stride,
    int max_context_len);

void sm120_fp4_paged_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& q_sf,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const at::ScalarType& logits_dtype, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, bool is_context_lens_2d,
    int logits_stride, int block_table_stride, int max_context_len);

} // namespace deep_gemm
