#!/usr/bin/env python3
"""
Fix PR #35156 which hardcodes mlp.gate quant_config=None, breaking AutoRound.
Applies the fix from PR #35261: only disable gate quantization for ModelOpt checkpoints.
"""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"

with open(TARGET) as f:
    content = f.read()

# Add the import if not present
import_line = "from vllm.model_executor.layers.quantization.modelopt import (\n    is_modelopt_quant_config,\n)"
if "is_modelopt_quant_config" not in content:
    # Insert after the last quantization import
    anchor = "from vllm.model_executor.layers.quantization import QuantizationConfig"
    if anchor in content:
        content = content.replace(anchor, anchor + "\n" + import_line)
    else:
        # Fallback: insert at top of imports
        content = import_line + "\n" + content

# Fix the gate quant_config
old = """self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )"""

new = """# ModelOpt checkpoints never quantize mlp.gate layers (PR #35261)
        gate_quant_config = (
            None if is_modelopt_quant_config(quant_config) else quant_config
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=gate_quant_config,
            prefix=f"{prefix}.gate",
        )"""

if old in content:
    content = content.replace(old, new, 1)
    print("  qwen3_next.py: gate quant_config fix applied")
else:
    print("  qwen3_next.py: pattern not found, may already be fixed")

with open(TARGET, "w") as f:
    f.write(content)
