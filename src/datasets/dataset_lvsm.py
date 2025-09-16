# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F



class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            with open(self.config.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
        
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.dataset_path}'")
            raise e
        

        self.use_idx_file = self.config.use_idx_file
        self.target_view_indices = None
        self.single_target_eval = self.use_idx_file and getattr(self.config, "num_tgt_views", 1) == 1
        # Load file that specifies the input and target view indices to use for inference
        if self.use_idx_file:
            self.view_idx_list = dict()
            if self.config.idx_file_path is not None:
                if os.path.exists(self.config.idx_file_path):
                    with open(self.config.idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                    # filter out scenes without specified input/target pairs
                    filtered_scene_paths = []
                    filtered_targets = []
                    for scene in self.all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        view_indices = self.view_idx_list.get(scene_name)
                        if view_indices is None:
                            continue
                        targets = view_indices.get("target")
                        if targets is None:
                            continue
                        if not isinstance(targets, (list, tuple)):
                            targets = [targets]
                        if self.single_target_eval:
                            for target_idx in targets:
                                filtered_scene_paths.append(scene)
                                filtered_targets.append(int(target_idx))
                        else:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths
                    if self.single_target_eval:
                        self.target_view_indices = filtered_targets


    def __len__(self):
        return len(self.all_scene_paths)


    def build_intrinsic_matrix(self, fxfycxcy):
        """
        Build 3x3 intrinsic matrix from fxfycxcy parameters
        Args:
            fxfycxcy: tensor of shape [..., 4] containing [fx, fy, cx, cy]
        Returns:
            intrinsic matrix of shape [..., 3, 3]
        """
        batch_shape = fxfycxcy.shape[:-1]
        intrinsic = torch.zeros(*batch_shape, 3, 3, dtype=fxfycxcy.dtype, device=fxfycxcy.device)
        intrinsic[..., 0, 0] = fxfycxcy[..., 0]  # fx
        intrinsic[..., 1, 1] = fxfycxcy[..., 1]  # fy
        intrinsic[..., 0, 2] = fxfycxcy[..., 2]  # cx
        intrinsic[..., 1, 2] = fxfycxcy[..., 3]  # cy
        intrinsic[..., 2, 2] = 1.0
        return intrinsic

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        # resize_h = self.config.model.image_tokenizer.image_size
        # patch_size = self.config.model.image_tokenizer.patch_size
        # square_crop = self.config.training.get("square_crop", False)
        square_crop = True
        resize_h = self.config.image_size

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            # resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)    
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        # 
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        N = self.config.num_ref_views + self.config.num_tgt_views
        if len(frames) < N:
            return None
        # sample view candidates
        min_frame_dist = self.config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, self.config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        sampled_frames = random.sample(range(start_frame + 1, end_frame), N-2)
        image_indices = [start_frame, end_frame] + sampled_frames 
        image_indices = sorted(image_indices) 
        if self.config.shuffle_prob > random.random():
            random.shuffle(image_indices)
        return image_indices

    def __getitem__(self, idx):
        # try:
        scene_path = self.all_scene_paths[idx].strip()
        data_json = json.load(open(scene_path, 'r'))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.use_idx_file and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            context_indices = current_view_idx.get("context", [])
            if not isinstance(context_indices, (list, tuple)):
                context_indices = [context_indices]
            context_indices = [int(i) for i in context_indices]

            if self.single_target_eval and self.target_view_indices is not None:
                target_idx = int(self.target_view_indices[idx])
                image_indices = context_indices + [target_idx]
            else:
                target_indices = current_view_idx.get("target", [])
                if not isinstance(target_indices, (list, tuple)):
                    target_indices = [target_indices]
                target_indices = [int(i) for i in target_indices]
                image_indices = context_indices + target_indices
        else:
            # sample input and target views
            image_indices = self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
    
        # except:
        #     traceback.print_exc()
        #     print(f"error loading")
        #     print(image_indices)
        #     print(image_paths_chosen)
        #     return self.__getitem__(random.randint(0, len(self) - 1))

        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.config.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)
        input_w2cs = torch.linalg.inv(input_c2ws)
        intrinsic_matrices = self.build_intrinsic_matrix(input_intrinsics)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        
        return {
            "image": input_images,
            "extrinsic": input_w2cs,
            "intrinsic": intrinsic_matrices,
            # "c2w": input_c2ws,
            # "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene_name
        }

        # return {
        #     "image": input_images,
        #     "c2w": input_c2ws,
        #     "fxfycxcy": input_intrinsics,
        #     "index": indices,
        #     "scene_name": scene_name
        # }
