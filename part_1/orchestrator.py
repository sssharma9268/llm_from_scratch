# Repository layout (Part 1)
#
#   part_1/
#     orchestrator.py               # runs demos/tests/visualizations for Part 1
#     pos_encoding.py               # 1.1 positional encodings (learned + sinusoidal)
#     attn_numpy_demo.py            # 1.2 self-attention math with tiny numbers (NumPy)
#     single_head.py                # 1.3 single attention head (PyTorch)
#     multi_head.py                 # 1.4 multi-head attention (with shape tracing)
#     ffn.py                        # 1.5 feed-forward network (GELU, width = mult*d_model)
#     block.py                      # 1.6 Transformer block (residuals + LayerNorm)
#     attn_mask.py                  # causal mask helpers
#     vis_utils.py                  # plotting helpers (matrices & attention maps)
#     demo_mha_shapes.py            # prints explicit matrix multiplications & shapes step-by-step
#     demo_visualize_multi_head.py  # saves attention heatmaps per head (grid)
#     out/                          # (created at runtime) images & logs live here
#     tests/
#       test_attn_math.py           # correctness: tiny example vs PyTorch single-head
#       test_causal_mask.py         # verifies masking behavior
#
# NOTE ON IMPORTS
# ----------------
# All imports are LOCAL. Run from inside `part_1/`.
# Example quickstart (CPU ok):
#   cd part_1
#   python orchestrator.py --visualize

