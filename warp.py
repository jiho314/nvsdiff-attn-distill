
import torch
from pytorch3d.renderer import (
    PointsRasterizer, PointsRenderer, AlphaCompositor,
    PointsRasterizationSettings, PerspectiveCameras
)
from pytorch3d.structures import Pointclouds

# modified from Met3r (https://github.com/mohammadasim98/met3r)
def render_points_pytorch3d(
        points_xyz_w,
        feats, 
        intrinsic,
        extrinsic= None,
        render_resolution = (512, 512),
        ndc_radius = 0.01,
        # radius_scale = 1.0, # not used
        points_per_pixel=10,
        background=0.):
    """
    points_xyz_w: (B,N,3) world coords
    feats:       (B,N,C) 
    K:            (B,3,3) intrinsics (fx,fy,cx,cy)
    Rt:           (B,3,4) or (B 4 4) world->cam extrinsics (if None: identity)
    H,W: rendering size
    Returns:      (B,C,H,W) float image
    """
    H, W = render_resolution
    assert H == W, "Non-Square rendering size is not supported"
    device, original_dtype = points_xyz_w.device, points_xyz_w.dtype
    B, N, _ = points_xyz_w.shape
    K = intrinsic.clone()
    Rt = extrinsic.clone()


    if Rt is None:
        R = torch.eye(3, device=device).expand(B,3,3)
        T = torch.zeros(B,3, device=device)
    else:
        R = Rt[..., :3, :3]              # (B,3,3)
        T = Rt[..., :3, 3]              # (B,3)

    # dtype float32 
    _points_xyz_w, _feats, _R, _T, _K = points_xyz_w.to(torch.float32), feats.to(torch.float32), R.to(torch.float32), T.to(torch.float32), K.to(torch.float32)
    


    # # Extrinsic Convention: OpenCV -> OpenGL 
    _R = _R.clone().permute(0, 2, 1)
    _T = _T.clone()
    _R[:, :, :2] *= -1
    _T[:, :2] *= -1
    # # _R[...,0,0] *= -1
    # # _R[...,1,1] *= -1
    # _R[...,:,:2] *= -1
    # _T[...,0] *= -1 # TODO: check if this is right
    # _T[...,1] *= -1

    fx, fy = _K[...,0,0], _K[...,1,1]
    cx, cy = _K[...,0,2], _K[...,1,2]
    original_H, original_W = cy *2 , cx *2

    # Adjust intrinsics for rendering resolution
    for i in range(B):
        if original_H[i].item() != H or original_W[i].item() != W:
            fx[i] = fx[i] * (H / original_H[i])
            fy[i] = fy[i] * (W / original_W[i])
            cx[i] = cx[i] * (H / original_H[i])
            cy[i] = cy[i] * (W / original_W[i])


    # PyTorch3D cameras can take pixel intrinsics if you also pass image_size:
    cameras = PerspectiveCameras(
        focal_length=torch.stack([fx, fy], dim=-1),          # (B,2)
        principal_point=torch.stack([cx, cy], dim=-1),       # (B,2)
        image_size=torch.as_tensor([(H, W)], device=device).expand(B, -1),  # (B,2)
        in_ndc= False,
        R=_R, T=_T, device=device
    )

    if ndc_radius is None: 
        ndc_radius =  2 / max(H,W) # ndc range [-1, 1] 

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=ndc_radius, # ncd radius required
        points_per_pixel=points_per_pixel,
        bin_size = 0
    )

    # Build batched pointcloud
    pcls = Pointclouds(points=list(_points_xyz_w.unbind(dim=0)), features=list(_feats.unbind(dim=0)))


    rasterizer=PointsRasterizer(raster_settings=raster_settings)
    compositor=AlphaCompositor()

    # render
    def render(point_clouds, **kwargs):
        with torch.autocast(str(device), enabled=False):
            fragments = rasterizer(point_clouds,**kwargs)

        r = rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf

    images, zbuf = render(pcls,cameras=cameras, background_color=[background] * _feats.shape[-1])
    return images.permute(0, 3, 1, 2).to(original_dtype), zbuf.to(original_dtype)
