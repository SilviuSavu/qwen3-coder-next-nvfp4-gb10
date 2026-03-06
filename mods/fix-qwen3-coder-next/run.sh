#!/bin/bash
set -e

# Revert PR #34279 that removed tl.int64 stride annotations (causes ~12% slowdown)
# The diff captures #34279's changes; applying in reverse restores int64 strides
echo "Reverting PR #34279 that causes slowness"
patch -p1 -R -d /usr/local/lib/python3.12/dist-packages < fix_slowness.diff || echo "Can't revert PR #34279, skipping as it was reverted in recent commits"

# Triton allocator workaround for DGX Spark (GB10/sm121)
# Patches Triton's NullAllocator to use PyTorch's CUDA caching allocator
# Tracking: https://github.com/vllm-project/vllm/issues/33857
echo "Fixing Triton allocator bug"
cp _triton* /usr/local/lib/python3.12/dist-packages/

# Fix PR #35156 which hardcodes mlp.gate quant_config=None, breaking AutoRound
# Applies conditional fix from PR #35261
echo "Fixing gate quantization for AutoRound (PR #35261)"
python3 fix_gate_quant.py
