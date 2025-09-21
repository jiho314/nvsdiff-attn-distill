# just for coding
import torch

from my_diffusers.models.attention_processor import AttnProcessor2_0

# unet layers: total 16
# down_blocks(6): 0.0, 0.1, 1.0, 1.1, 2.0, 2.1
# mid_block(1): 0.0
# up_blocks(9): 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2

# sorted attn processor  = ['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor']
# Attn Head Num: [5,5,10,10,20,20] [20] [20,20,20, 10,10,10, 5,5,5]

SDXL_ATTN_DIM = [320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
SDXL_FEAT_DIM = []
SDXL_ATTN_HEAD_NUM = [5,5,10,10,20,20, 20, 20,20,20,10,10,10,5,5,5]

def print_attn_cache_setting(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        print(f"Layer {i}: {name}, cache_attn: {unet.attn_processors[name].cache_attn}")

def set_attn_cache( unet,  cache_layers = [] , print_=True):
    ''' cache_layers: list of int layer index to cache attention (0~15)
    '''
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        if i in cache_layers:
            # print(f"Cache attn layer: {i}={name}")
            unet.attn_processors[name].cache_attn = True
        else:
            unet.attn_processors[name].cache_attn = False
    if print_:
        print_attn_cache_setting(unet)

def unset_attn_cache(unet, print_=False):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache_attn = False
    if print_:
        print_attn_cache_setting(unet)

def pop_cached_attn(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    block_attn_cache = {}
    for i, name in enumerate(attn_proc_names):
        if unet.attn_processors[name].cache_attn:
            cache_list = unet.attn_processors[name].cache['attn']
            assert len(cache_list) == 1, f"Expected single cached attention tensor, got {len(cache_list)}"
            block_attn_cache[str(i)] = cache_list.pop()
    return block_attn_cache

def clear_attn_cache(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache['attn'].clear()
    torch.cuda.empty_cache()

# Feat
def print_feat_cache_setting(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        print(f"Layer {i}: {name}, cache_feat: {unet.attn_processors[name].cache_feat}")

def set_feat_cache( unet,  cache_layers = [] ):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        if i in cache_layers:
            # print(f"Cache feat layer: {name}")
            unet.attn_processors[name].cache_feat = True
        else:
            unet.attn_processors[name].cache_feat = False

    print_feat_cache_setting(unet)

def unset_feat_cache(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache_feat = False

    print_feat_cache_setting(unet)

def pop_cached_feat(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    block_feat_cache = {}
    for i, name in enumerate(attn_proc_names):
        if unet.attn_processors[name].cache_feat:
            cache_list = unet.attn_processors[name].cache['feat']
            assert len(cache_list) == 1, f"Expected single cached feature tensor, got {len(cache_list)}"
            block_feat_cache[str(i)] = cache_list.pop()
    return block_feat_cache

def clear_feat_cache(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache['feat'].clear()
    torch.cuda.empty_cache()