
from cgi import print_environ
import os 
import json
import sys
import webdataset as wds
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import IterableDataset
from pathlib import Path
from io import BytesIO
from PIL import Image



class Re10k_torch(IterableDataset):
    """
    RE10K Partial Val IterableDataset.
    - Input: `.torch` bundle files (traverse all files under root)
    - Index: use context/target from evaluation_index_re10k.json (Interpolate),
              or heuristic indexing (Extrapolate)
    - VGGT to estimate pose (depth/points included)
    - Output keys:
        images: [V, 3, 512, 512] (첫 항목이 target)
        (optional)
        points: [V, 3, 512, 512]
        intrinsics: [V, 3, 3]
        extrinsics: [V, 3, 4]
    """

    def __init__(
        self,
        *,
        root: str = "/mnt/data2/minkyung/re10k/test_partial/test",
        index_json: str = "/mnt/data1/minseop/multiview-gen/evaluation_index_re10k.json",
        # num_ref_viewpoints: int = 2,
        setting: str = "extrapolate",
        # device: str = "cuda",
        # amp_dtype: torch.dtype = torch.bfloat16,
        use_vggt: bool = False,
        validation = False, # whether to use all 3 idx among extrapolate # TODO fix code
    ) -> None:
        super().__init__()
        self.validation = validation
        # Match eval_utils: it scans real_root (..../test_partial) not the nested 'test' dir
        _root_path = Path(root)
        if _root_path.name.lower() == "test":
            _root_path = _root_path.parent
        self.root = str(_root_path)
        # Deterministic, path-sorted order
        self.files = sorted([str(p) for p in Path(self.root).rglob("*.torch")])
        assert len(self.files) > 0, f"No .torch files found in {root}"
        # 평가 인덱스(JSON)는 원본 코드와 동일하게 항상 로드
        with open(index_json, "r") as f:
            self.index = json.load(f)
        # self.num_ref = int(num_ref_viewpoints)
        self.setting = str(setting).lower()
        assert self.setting in ("interpolate", "extrapolate"), "setting must be 'interpolate' or 'extrapolate'"
        # self.device = torch.device(device)
        # self.amp_dtype = amp_dtype
        self.use_vggt = bool(use_vggt)
        # # interpolate 평가 시 사용할 타겟 뷰 개수 (컨텍스트 다음에 배치)
        # self.num_target_views = 3

        # VGGT 준비 (옵션)
        assert use_vggt == False, "VGGT is not used in dataset"
        self.model = None
        # if self.use_vggt:
        #     self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        #     self.model.eval()

        # Build key -> (file_path, index_in_file) map to strictly match eval_utils key order
        self.key_to_loc: dict[str, tuple[str, int]] = {}
        self.key_to_loc_norm: dict[str, tuple[str, int]] = {}
        def _norm_key(s: str) -> str:
            # normalize: take last path segment, drop extension, lowercase
            import os as _os
            base = _os.path.basename(str(s))
            if "." in base:
                base = base.split(".")[0]
            return base.lower()
        for fp in self.files:
            try:
                examples = torch.load(fp, map_location="cpu")
            except Exception:
                continue
            if isinstance(examples, dict):
                examples = [examples]
            for idx, ex in enumerate(examples):
                k = ex.get("key") or ex.get("scene")
                if isinstance(k, str):
                    if k not in self.key_to_loc:
                        self.key_to_loc[k] = (fp, idx)
                    nk = _norm_key(k)
                    if nk not in self.key_to_loc_norm:
                        self.key_to_loc_norm[nk] = (fp, idx)

        # eval_utils와 동일하게 파일별 예제 개수를 미리 저장
        self.chunk_overview: list[int] = []
        for fp in self.files:
            try:
                examples = torch.load(fp, map_location="cpu")
            except Exception:
                self.chunk_overview.append(0)
                continue
            if isinstance(examples, dict):
                self.chunk_overview.append(1)
            else:
                try:
                    self.chunk_overview.append(len(examples))
                except Exception:
                    self.chunk_overview.append(0)

    @staticmethod
    def _decode_images_from_bytes(objs):
        to_tensor = transforms.ToTensor()
        out = []
        for o in objs:
            img = Image.open(BytesIO(o.cpu().numpy().tobytes())).convert("RGB")
            out.append(to_tensor(img))
        return torch.stack(out)  # [V,3,H,W], float in [0,1]

    @staticmethod
    def _resize_center_crop_batch(x: torch.Tensor, size: int = 512) -> torch.Tensor:
        # x: [V, C, H, W]
        _, _, h, w = x.shape
        if w < h:
            new_w, new_h = size, int(size * h / w)
        else:
            new_h, new_w = size, int(size * w / h)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return x[:, :, top:top + size, left:left + size]

    def _select_indices(self, key: str, num_imgs: int) -> list:
        if self.setting == "interpolate":
            # Follow genwarp EvalBatch: concat(context, target) then pick positions [0,1,2,3,4]
            rec = self.index.get(key, None)
            assert rec is not None and ("context" in rec and "target" in rec), f"Missing index entry for key={key}"
            ctx_list = list(rec.get("context", []))
            tgt_list = list(rec.get("target", []))
            return ctx_list, tgt_list
            # idx_full = ctx_list + tgt_list
            # used_pos = [0, 1, 2, 3, 4]
            # used_pos = [0, 1, 2]
            # used_idx = [idx_full[p] for p in used_pos if p < len(idx_full)]
            # return used_idx
        else:
            # extrapolate: heuristic indices exactly as EvalBatch
            ctx_list = [
                int(num_imgs * 2 / 3),
                int(num_imgs * 4 / 5),
                # int(num_imgs * 1 / 2), # not used, 
                # int(num_imgs * 3 / 5), # not used
            ]
            tgt_list = [
                int(num_imgs * 1 / 3),
                int(num_imgs * 2 / 5),
                int(num_imgs * 1 / 4),
            ]
            # TODO: fix code
            if self.validation:
                tgt_list = [int(num_imgs * 1 / 4)]
            return ctx_list, tgt_list
    

    def __iter__(self):
        # eval_utils의 순회를 그대로 재현: (chunk_num, example_num)을 유지하면서
        # 각 스텝에서 먼저 example_num을 증가시킨 뒤 로드하고, index에 없는 키는
        # 다음 예제로 이동하며 계속 진행
        chunk_num = 0
        example_num = -1

        while True:
            # step
            example_num += 1
            if chunk_num < len(self.chunk_overview):
                if self.chunk_overview[chunk_num] == example_num:
                    chunk_num += 1
                    example_num = 0
                    if chunk_num == len(self.chunk_overview): # every chunk used
                        break
            
            f = self.files[chunk_num]
            examples = torch.load(f, map_location="cpu")
            # try:
            #     examples = torch.load(f, map_location="cpu")
            # except Exception:
            #     # 손상 파일은 통째로 스킵
            #     chunk_num += 1
            #     example_num = 0
            #     continue
            if isinstance(examples, dict):
                examples = [examples]

            example = examples[example_num]
            # 인덱스에 없는 키는 다음 예제로 계속 이동 (eval_utils while-loop 동작)
            key = example.get("key") if isinstance(example, dict) else None
            entry = self.index.get(key) if isinstance(key, str) else None
            if entry is None:
                print(f"scene '{key}' not found in eval_idx set")
                continue

            imgs_all = example.get("images")
            # if imgs_all is None:
            #     continue
            num_imgs = len(imgs_all)

            # # used_idx 이름 구성 (디버그/동일성 유지 목적)
            # if self.setting == "interpolate":
            #     ctx_list = list(entry.get("context", []))
            #     tgt_list = list(entry.get("target", []))
            # else:
            #     ctx_list = [
            #         int(num_imgs * 2 / 3),
            #         int(num_imgs * 4 / 5),
            #         int(num_imgs * 1 / 2),
            #         int(num_imgs * 3 / 5),
            #     ]
            #     tgt_list = [
            #         int(num_imgs * 1 / 3),
            #         int(num_imgs * 2 / 5),
            #         int(num_imgs * 1 / 4),
            #     ]
            # idx_full = ctx_list + tgt_list
            # used_pos = [0, 1, 2, 3, 4] if self.setting == "interpolate" else [0, 1, 4, 5, 6]
            # used_idx_vals = [idx_full[p] for p in used_pos if p < len(idx_full)]
            # name = key + "_idx_" + "_".join(map(str, used_idx_vals))

            # EvalBatch와 동일 선택
            ctx_idxs, tgt_idxs = self._select_indices(key, num_imgs)
            assert len(ctx_idxs) == 2, "Only refernce num 2 is supported"
            for tgt_idx in tgt_idxs:
                name = key + "_idx_" + "_".join(map(str,[tgt_idx] + ctx_idxs))
                sel = [tgt_idx] + ctx_idxs # target at first
                img_objs = [imgs_all[i] for i in sel]
                images = self._decode_images_from_bytes(img_objs)  # [V,3,H,W]
                images = self._resize_center_crop_batch(images, size=512)

                images_out = torch.nn.functional.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False).to(dtype=torch.float32)
                # pts_out = None
                # if pts is not None:
                #     # pts currently at 518x518; downscale to 512 exactly like eval_utils.transform_pts
                #     pts_out = torch.nn.functional.interpolate(pts, size=(512, 512), mode="bilinear", align_corners=False).to(dtype=torch.float32)

                sample = {"image": images_out, "key": key, "instance_name": name, "sel_indices": sel}
                # if pts_out is not None:
                #     sample["points"] = pts_out
                # if intrinsic is not None:
                #     sample["intrinsics"] = intrinsic[0].to(torch.float32).cpu()
                # if extrinsic is not None:
                #     sample["extrinsics"] = extrinsic[0].to(torch.float32).cpu()
                yield sample

# debug
if __name__ == "__main__":
    d = Re10k_torch(
            root="/mnt/data2/minkyung/re10k/test_partial/test",
            index_json='/mnt/data1/minseop/multiview-gen/evaluation_index_re10k.json',
            setting="extrapolate",
        )
    from torch.utils.data import DataLoader

    d_loader = DataLoader(
            d,
            batch_size=1,
            # persistent_workers=True,
            num_workers=0)
    ho =[]
    for i, batch in enumerate(d_loader):
        ho.append(batch)

    print('data len :',  len(ho))

    import pdb ; pdb.set_trace()