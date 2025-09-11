from src.datasets.re10k_wds import build_re10k_wds
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

ho = build_re10k_wds()
# sampler = DistributedSampler(ho, shuffle=True, drop_last=False)

dl = DataLoader(
    ho,
    batch_size=2,
    num_workers=4,
    
)

# Prepare with accelerator
# dl = accelerator.prepare(dl)

print(f"Process {accelerator.process_index}/{accelerator.num_processes} starting...")
rank = accelerator.device
for i, batch in enumerate(dl):
    # if accelerator.is_main_process:
    print(f"Rank:{rank}, Batch {i}: keys = {list(batch.keys())}")
    print("__key__:", batch['__key__'])

    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape = {value.shape}")
        else:
            print(f"  {key}: type = {type(value)}")

    if i >= 5:  # Just check first few batches
        break

print(f"Process {accelerator.process_index} finished.")

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 data_check.py