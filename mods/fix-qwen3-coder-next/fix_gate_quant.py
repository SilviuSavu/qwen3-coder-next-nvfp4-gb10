#!/usr/bin/env python3
"""
Fix PR #35156 which hardcodes mlp.gate quant_config=None, breaking AutoRound.
Simply restores quant_config passthrough so quantized gate weights load correctly.
"""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"

with open(TARGET) as f:
    content = f.read()

old = """self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )"""

new = """self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )"""

if old in content:
    content = content.replace(old, new, 1)
    print("  qwen3_next.py: gate quant_config fix applied")
else:
    print("  qwen3_next.py: pattern not found, may already be fixed")

with open(TARGET, "w") as f:
    f.write(content)
