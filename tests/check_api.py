"""Quick API compatibility check for pytorch/2.8.0 on Perlmutter."""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaConfig, LlamaForCausalLM

print(f"torch: {torch.__version__}")
print(f"transformer_auto_wrap_policy: OK")
print(f"register_comm_hook in FSDP: {hasattr(FSDP, 'register_comm_hook')}")

ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
print(f"autocast: OK")

# Build a tiny Llama model to test meta-device init
cfg = LlamaConfig(
    hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=1024,
    max_position_embeddings=512,
)
with torch.device('meta'):
    m = LlamaForCausalLM(cfg)
m2 = m.to_empty(device='cpu')

has_iw = hasattr(m2, 'init_weights')
has_pi = hasattr(m2, 'post_init')
print(f"init_weights method: {has_iw}")
print(f"post_init method: {has_pi}")

# Try running init
if has_iw:
    m2.init_weights()
    print("init_weights() ran OK")

# Tiny forward pass (CPU)
inp = torch.randint(0, 1024, (1, 10))
with torch.no_grad():
    out = m2(input_ids=inp)
print(f"forward pass OK — loss: {out.loss}")
print("ALL CHECKS PASSED")
