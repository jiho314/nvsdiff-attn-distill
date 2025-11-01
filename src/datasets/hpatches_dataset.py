import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageDraw
import math
import numpy as np


# Default absolute path to the HPatches sequences inside this workspace
DEFAULT_HPATCHES_PATH = "/mnt/data1/seonghu/d2-net/hpatches_sequences/hpatches-sequences-release"
DEFAULT_KPTS_PATH = "/mnt/data1/seonghu/superpoint-1k"


class HPatchesDataset(Dataset):
    """PyTorch Dataset for HPatches sequences.

    Expects the HPatches sequences directory structure where each sequence
    is a folder containing 1.ppm .. 6.ppm and homography files H_1_2 etc.

    Args:
        root (str): path to hpatches-sequences-release directory.
        seqs (list|None): optional list of sequence folder names to use.
        transforms (callable|None): optional transform applied to PIL image.
    """

    def __init__(self, root=None, seqs=None, transforms=None):
        # default to repository hpatches location when not provided
        if root is None:
            root = DEFAULT_HPATCHES_PATH
        self.root = os.path.abspath(root)
        self.transforms = transforms

        # collect all sequence names
        all_seqs = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        if seqs is not None:
            seqs = [s for s in seqs if s in all_seqs]
        else:
            seqs = all_seqs

        # Filter sequences to match eval_hpatches expectations: require images 1..6, GT H_1_2..H_1_6 and superpoint kps
        valid_seqs = []
        for seq in seqs:
            seq_dir = os.path.join(self.root, seq)
            imgs_ok = all(os.path.exists(os.path.join(seq_dir, f"{i}.ppm")) for i in range(1, 7))
            H_ok = all(os.path.exists(os.path.join(seq_dir, f'H_1_{i}')) for i in range(2, 7))
            kps_ok = all(os.path.exists(os.path.join(DEFAULT_KPTS_PATH, f"{seq}-{i}.kp")) for i in range(1, 7))
            if imgs_ok and H_ok and kps_ok:
                valid_seqs.append(seq)

        if len(valid_seqs) == 0:
            raise RuntimeError(f"No valid HPatches sequences found in {self.root} with superpoint keypoints at {DEFAULT_KPTS_PATH}")

        self.items = []  # list of tuples (seq_name, img_idx, img_path)
        for seq in valid_seqs:
            seq_dir = os.path.join(self.root, seq)
            for idx in range(1, 7):
                img_name = f"{idx}.ppm"
                img_path = os.path.join(seq_dir, img_name)
                self.items.append((seq, idx, img_path))

        # expose valid sequences and simple grouping (All / Viewpoint / Illumination)
        self.valid_seqs = list(valid_seqs)
        self.groups = {
            "All": self.valid_seqs,
            "Viewpoint Change": [s for s in self.valid_seqs if os.path.basename(s).startswith('v')],
            "Illumination Change": [s for s in self.valid_seqs if os.path.basename(s).startswith('i')],
        }

    # ----------------------------
    # Resize / crop / homography utilities moved from eval_hpatches.py
    # ----------------------------
    @staticmethod
    def build_resize_crop_matrix(w: int, h: int, crop_w: int, crop_h: int, out_w: int = 512, out_h: int = 512) -> np.ndarray:
        """Return 3x3 matrix that maps original-image coordinates -> cropped+resized coordinates.

        Mapping: p_orig -> p_cropped = p_orig - [left, top]
                 p_resized = (out / crop) * p_cropped
        So total S = T_resize @ T_crop where T_crop translates by -left,-top.
        """
        left = int(max(round((w - crop_w) / 2.0), 0))
        top  = int(max(round((h - crop_h) / 2.0), 0))
        T_crop = np.array([[1.0, 0.0, -float(left)], [0.0, 1.0, -float(top)], [0.0, 0.0, 1.0]], dtype=np.float64)
        T_resize = np.array([[float(out_w) / float(crop_w), 0.0, 0.0], [0.0, float(out_h) / float(crop_h), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return T_resize @ T_crop

    @staticmethod
    def build_resize_then_crop_matrix(w: int, h: int, out_size: int = 512) -> np.ndarray:
        """Build 3x3 matrix mapping original image coords -> resized-then-center-cropped coords of size out_size x out_size.

        Steps:
        1) scale uniformly so the smaller side becomes out_size
        2) center-crop a square of size out_size from the resized image
        Returns matrix C @ R where R scales and C translates by -left,-top.
        """
        if w <= 0 or h <= 0:
            raise ValueError("Invalid image size for build_resize_then_crop_matrix")
        s = float(out_size) / float(min(w, h))
        w_r = int(round(w * s))
        h_r = int(round(h * s))
        left = int(max(round((w_r - out_size) / 2.0), 0))
        top = int(max(round((h_r - out_size) / 2.0), 0))
        R = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        C = np.array([[1.0, 0.0, -float(left)], [0.0, 1.0, -float(top)], [0.0, 0.0, 1.0]], dtype=np.float64)
        return C @ R

    @staticmethod
    def map_keypoints_with_matrix(S: np.ndarray, kps: np.ndarray, out_size: int = 512) -> np.ndarray:
        """Apply 3x3 map S to keypoints (N,2). Returns filtered (inside 0..out_size) (M,2) float32 array."""
        if kps is None or kps.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        kps_xy = kps.reshape(-1, 2).astype(np.float64)
        ones = np.ones((kps_xy.shape[0], 1), dtype=np.float64)
        P = np.concatenate([kps_xy, ones], axis=1)  # (N,3)
        Pp = (S @ P.T).T
        eps = 1e-8
        Pp[:, 0] = Pp[:, 0] / (Pp[:, 2] + eps)
        Pp[:, 1] = Pp[:, 1] / (Pp[:, 2] + eps)
        mapped = Pp[:, :2]
        inside_mask = (mapped[:, 0] >= 0) & (mapped[:, 0] < out_size) & (mapped[:, 1] >= 0) & (mapped[:, 1] < out_size)
        if not inside_mask.any():
            return np.zeros((0, 2), dtype=np.float32)
        return mapped[inside_mask].astype(np.float32)

    @staticmethod
    def scale_homography_to_resize(H: np.ndarray, w_i: int, h_i: int, w_j: int, h_j: int, out_w: int = 512, out_h: int = 512) -> np.ndarray:
        """Scale homography H (from original coords) to operate on resized images (out_w x out_h).

        H: 3x3 homography mapping points in image i (original) to image j (original).
        Returns H_resized = S_j @ H @ inv(S_i), where S maps original->resized.
        """
        S_i = np.array([[out_w / float(w_i), 0, 0], [0, out_h / float(h_i), 0], [0, 0, 1]], dtype=np.float64)
        S_j = np.array([[out_w / float(w_j), 0, 0], [0, out_h / float(h_j), 0], [0, 0, 1]], dtype=np.float64)
        return S_j @ H @ np.linalg.inv(S_i)

    @staticmethod
    def filter_and_map_keypoints_to_512(kps: np.ndarray, w: int, h: int, crop_w: int = 1200, crop_h: int = 1200, out_size: int = 512) -> np.ndarray:
        """Filter keypoints to those inside the centered crop (crop_w x crop_h) and map to out_size coords.

        Returns an (M,2) float32 array in out_size coordinate frame.
        """
        if kps is None or kps.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # compute centered crop top-left
        left = int(max(round((w - crop_w) / 2.0), 0))
        top  = int(max(round((h - crop_h) / 2.0), 0))
        right = left + crop_w
        bottom = top + crop_h

        # ensure kps shape
        kps_xy = kps.reshape(-1, 2).astype(np.float64)

        # boolean mask for points inside crop bounds (inclusive left/top, exclusive right/bottom)
        inside_mask = (kps_xy[:, 0] >= left) & (kps_xy[:, 0] < right) & (kps_xy[:, 1] >= top) & (kps_xy[:, 1] < bottom)
        if not inside_mask.any():
            return np.zeros((0, 2), dtype=np.float32)

        kps_in = kps_xy[inside_mask]
        # shift to crop-local coordinates
        kps_shifted = kps_in - np.array([[left, top]], dtype=np.float64)
        # map to out_size
        sx = float(out_size) / float(crop_w)
        sy = float(out_size) / float(crop_h)
        kps_out = np.stack([kps_shifted[:, 0] * sx, kps_shifted[:, 1] * sy], axis=1).astype(np.float32)
        return kps_out

    @staticmethod
    def read_superpoint_kpts(seq: str, img_idx: int, kpts_root: str = DEFAULT_KPTS_PATH) -> np.ndarray:
        kp_path = os.path.join(kpts_root, f"{seq}-{img_idx}.kp")
        if not os.path.exists(kp_path):
            raise FileNotFoundError(f"SuperPoint keypoints not found: {kp_path}")
        kps = np.loadtxt(kp_path).astype(np.float32)
        return kps

    @staticmethod
    def load_homography(root: str, seq: str, i: int, j: int) -> np.ndarray:
        H_path = os.path.join(root, seq, f"H_{i}_{j}")
        if not os.path.exists(H_path):
            raise FileNotFoundError(f"Homography not found: {H_path}")
        H = np.loadtxt(H_path).astype(np.float64)
        return H

    @staticmethod
    def warp_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """Apply homography H (3x3) to points (N,2) -> (N,2)"""
        if pts_xy is None or pts_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        ones = np.ones((pts_xy.shape[0], 1), dtype=np.float64)
        P = np.concatenate([pts_xy.astype(np.float64), ones], axis=1)  # (N,3)
        Pp = (H @ P.T).T
        eps = 1e-8
        Pp[:, 0] = Pp[:, 0] / (Pp[:, 2] + eps)
        Pp[:, 1] = Pp[:, 1] / (Pp[:, 2] + eps)
        return Pp[:, :2].astype(np.float32)

    @staticmethod
    def preprocess_pair(root: str, seq: str, i_img: int, j_img: int, out_size: int = 512) -> dict:
        """Load a pair, center-crop to common square, resize to out_size, map keypoints and scale homography.

        Returns dict with keys:
          'ref': PIL Image resized, 'tgt': PIL Image resized,
          'kps_i_resized': (N,2) float32, 'H_resized': (3,3) float64, 'gt_xy_j_resized': (N,2) float32
        """
        img1_path = os.path.join(root, seq, f"{i_img}.ppm")
        img2_path = os.path.join(root, seq, f"{j_img}.ppm")
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            raise FileNotFoundError(f"Image pair not found: {img1_path} or {img2_path}")

        img1_pil = Image.open(img1_path).convert('RGB')
        img2_pil = Image.open(img2_path).convert('RGB')

        w1, h1 = img1_pil.size
        w2, h2 = img2_pil.size

        # Step 1: resize each image so its smaller side becomes out_size
        def compute_resize_size(w: int, h: int, out_s: int) -> tuple:
            s = float(out_s) / float(min(w, h))
            return int(round(w * s)), int(round(h * s)), s

        w1_r, h1_r, s1 = compute_resize_size(w1, h1, out_size)
        w2_r, h2_r, s2 = compute_resize_size(w2, h2, out_size)

        img1_resized = img1_pil.resize((w1_r, h1_r), resample=Image.BILINEAR)
        img2_resized = img2_pil.resize((w2_r, h2_r), resample=Image.BILINEAR)

        # Step 2: center-crop a  out_size x out_size patch from each resized image
        def center_crop_resized(pil_resized: Image.Image, out_s: int) -> Image.Image:
            w, h = pil_resized.size
            left = int(max(round((w - out_s) / 2.0), 0))
            top = int(max(round((h - out_s) / 2.0), 0))
            return pil_resized.crop((left, top, left + out_s, top + out_s))

        ref_resized = center_crop_resized(img1_resized, out_size)
        tgt_resized = center_crop_resized(img2_resized, out_size)

        # Build mapping matrices from original -> resized-then-cropped coords
        Si = HPatchesDataset.build_resize_then_crop_matrix(w1, h1, out_size)
        Sj = HPatchesDataset.build_resize_then_crop_matrix(w2, h2, out_size)

        # Map keypoints from original image coords -> final out_size coords
        kps_i = HPatchesDataset.read_superpoint_kpts(seq, i_img)
        kps_i_resized = HPatchesDataset.map_keypoints_with_matrix(Si, kps_i, out_size=out_size)

        # Scale homography to operate on resized-then-cropped images: H_resized = Sj @ H @ inv(Si)
        H = HPatchesDataset.load_homography(root, seq, i_img, j_img)
        H_resized = Sj @ H @ np.linalg.inv(Si)

        # warp keypoints (mapped to out_size) using resized homography
        gt_xy_j_resized = HPatchesDataset.warp_points(H_resized, kps_i_resized)

        # Filter correspondences to those that are finite, reasonably sized and INSIDE the target out_size frame.
        # This removes points that warp outside the visible 0..out_size range (common with independent center crops)
        if gt_xy_j_resized is None or gt_xy_j_resized.size == 0 or kps_i_resized is None or kps_i_resized.size == 0:
            # nothing to filter
            return {
                'ref': ref_resized,
                'tgt': tgt_resized,
                'kps_i_resized': kps_i_resized,
                'H_resized': H_resized,
                'gt_xy_j_resized': gt_xy_j_resized,
            }

        # numerical sanity: finite and not extremely large
        finite_mask = np.isfinite(gt_xy_j_resized).all(axis=1)
        large_mask = (np.abs(gt_xy_j_resized) < 1e6).all(axis=1)

        # inside target frame mask
        inside_tgt = (gt_xy_j_resized[:, 0] >= 0) & (gt_xy_j_resized[:, 0] < out_size) & (
            gt_xy_j_resized[:, 1] >= 0) & (gt_xy_j_resized[:, 1] < out_size)

        keep_mask = finite_mask & large_mask & inside_tgt

        if not keep_mask.any():
            # return empty arrays to signal no valid correspondences
            return {
                'ref': ref_resized,
                'tgt': tgt_resized,
                'kps_i_resized': np.zeros((0, 2), dtype=np.float32),
                'H_resized': H_resized,
                'gt_xy_j_resized': np.zeros((0, 2), dtype=np.float32),
            }

        kps_i_resized = kps_i_resized[keep_mask]
        gt_xy_j_resized = gt_xy_j_resized[keep_mask]

        return {
            'ref': ref_resized,
            'tgt': tgt_resized,
            'kps_i_resized': kps_i_resized,
            'H_resized': H_resized,
            'gt_xy_j_resized': gt_xy_j_resized,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, im_idx, img_path = self.items[idx]
        # By default, return the full preprocessed pair for this sequence.
        # We use image 1 as reference and pair it with im_idx (if im_idx==1, pair with 2).
        ref_idx = 1
        tgt_idx = im_idx if int(im_idx) != 1 else 2
        out = HPatchesDataset.preprocess_pair(self.root, seq, ref_idx, tgt_idx, out_size=512)
        # convert PIL images to numpy arrays or apply user transforms so DataLoader.collate can batch
        ref_img = out.get('ref')
        tgt_img = out.get('tgt')
        if self.transforms is not None:
            try:
                out['ref'] = self.transforms(ref_img)
            except Exception:
                # fallback to numpy array
                out['ref'] = np.array(ref_img)
            try:
                out['tgt'] = self.transforms(tgt_img)
            except Exception:
                out['tgt'] = np.array(tgt_img)
        else:
            # default: return HWC uint8 numpy arrays
            out['ref'] = np.array(ref_img)
            out['tgt'] = np.array(tgt_img)

        # include identity metadata
        out['seq'] = seq
        out['im_idx'] = im_idx
        out['img_path'] = img_path
        return out




if __name__ == '__main__':
    """Quick test: instantiate dataset with DEFAULT_HPATCHES_PATH and iterate a few items."""
    from torch.utils.data import DataLoader

    print(f"DEFAULT_HPATCHES_PATH={DEFAULT_HPATCHES_PATH}")
    assert os.path.exists(DEFAULT_HPATCHES_PATH), f"HPatches path not found: {DEFAULT_HPATCHES_PATH}"

    dataset = HPatchesDataset(DEFAULT_HPATCHES_PATH)
    print(f"Found {len(dataset)} images across {len(set([s for s,_,_ in dataset.items]))} sequences")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in enumerate(dataloader):
        print(f"[{i}] seq={sample['seq'][0]} im_idx={sample['im_idx'][0]} img_path={sample['img_path'][0]}")
        if i >= 5:
            break

    # Visualize correspondences for multiple sequences (use same GT and kps as eval_hpatches)
    seqs = sorted(set([s for s, _, _ in dataset.items]))
    NUM_VIS = 5
    to_vis = seqs[:NUM_VIS]
    for seq0 in to_vis:
        print(f"Preprocessing and visualizing sequence {seq0}")
        try:
            out = HPatchesDataset.preprocess_pair(dataset.root, seq0, 1, 2, out_size=512)
        except FileNotFoundError as e:
            print(f"Skipping {seq0}: {e}")
            continue

        ref = out['ref']
        tgt = out['tgt']
        kps_i_resized = out['kps_i_resized']
        gt_xy_j_resized = out['gt_xy_j_resized']

        # build side-by-side visualization
        vis_img = Image.new('RGB', (512 * 2, 512), (255, 255, 255))
        vis_img.paste(ref, (0, 0))
        vis_img.paste(tgt, (512, 0))
        draw = ImageDraw.Draw(vis_img)

        for (x1, y1), (x2, y2) in zip(kps_i_resized, gt_xy_j_resized):
            try:
                p1 = (int(round(x1)), int(round(y1)))
                p2 = (int(round(x2)) + 512, int(round(y2)))
                draw.ellipse([p1[0]-2, p1[1]-2, p1[0]+2, p1[1]+2], outline='red')
                draw.ellipse([p2[0]-2, p2[1]-2, p2[0]+2, p2[1]+2], outline='red')
                draw.line([p1, p2], fill='yellow')
            except Exception:
                continue

        out_dir = os.path.join(os.getcwd(), 'hpatches_vis_preprocessed')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{seq0}_1_2_preprocessed.jpg')
        vis_img.save(out_path)
        print(f"Saved preprocessed visualization to {out_path}")


# Example usage (run in interactive script or notebook):
# from dift.hpatches_dataset import HPatchesDataset
# from torch.utils.data import DataLoader
# from torchvision import transforms
#
# transform = transforms.Compose([
#     transforms.Resize((768, 768)),
#     transforms.ToTensor(),
# ])
# dataset = HPatchesDataset('/path/to/hpatches-sequences-release', transforms=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
#
# for sample in dataloader:
#     # sample['image']: tensor (B,C,H,W)
#     # sample['img_path'], sample['seq'], sample['im_idx'] available
#     pass
