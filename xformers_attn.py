# import torch
# import torch.nn.functional as F
# from xformers.ops import memory_efficient_attention, LowerTriangularMask

# # Your function
# def attn_xformers(query, key, value, attention_mask=None, is_causal=False, dropout_p=0.0):
#     attn_bias = None

#     if is_causal:
#         attn_bias = LowerTriangularMask()

#     if attention_mask is not None:
#         mask_bool = attention_mask.bool()
#         bias = torch.zeros(mask_bool.shape, dtype=query.dtype, device=query.device)
#         bias = bias.masked_fill(~mask_bool, float("-inf"))
#         attn_bias = bias if attn_bias is None else (attn_bias + bias)
#         print (attn_bias.shape)

#     return memory_efficient_attention(query, key, value, attn_bias=attn_bias, p=dropout_p)

# if __name__ == "__main__":
    
#     # Debug script
#     def debug_attention():
#         torch.manual_seed(0)

#         # toy inputs
#         B, H, N, M, D = 2, 4, 8, 8, 16  # batch, heads, query_len, key_len, dim
#         query = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
#         key   = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
#         value = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

#         # padding mask example (keep first 6 tokens, mask last 2)
#         attention_mask = torch.ones(B, 1, N, M, device="cuda", dtype=torch.bool)
#         attention_mask[:, :, :, -2:] = False
#         # attention_mask = None

#         # xformers
#         out_xf = attn_xformers(query, key, value, attention_mask=attention_mask, is_causal=False)

#         # torch baseline
#         out_torch = F.scaled_dot_product_attention(
#             query, key, value,
#             attn_mask=attention_mask,  # bool mask is fine
#             dropout_p=0.0,
#             is_causal=False
#         )

#         # checks
#         print("xformers out shape:", out_xf.shape)
#         print("torch out shape   :", out_torch.shape)

#         # compare numerically
#         diff = (out_xf - out_torch).abs().max().item()
#         print(f"max abs diff: {diff:.6f}")

#         if diff > 1e-2:
#             print("⚠️ Significant difference, check masking logic / dtype")
#         else:
#             print("✅ Looks good!")

#     debug_attention()
