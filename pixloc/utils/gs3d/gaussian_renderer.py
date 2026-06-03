"""Minimal 3DGS render function for inference.

Adapted from Feature 3DGS.
Only the default rendering path (SH computed by CUDA, no pre-computed
covariance) is kept, which matches the pipe settings used by GS3DRenderer.

Requires ``diff-gaussian-rasterization`` (Feature 3DGS version) to be
installed — see README for instructions.

Original: https://github.com/graphdeco-inria/gaussian-splatting
License: see https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
"""

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    ones = torch.ones((means3D.shape[0], 1), dtype=means3D.dtype, device="cuda")
    means_h = torch.cat((means3D, ones), dim=1)
    view_xyz = means_h @ viewpoint_camera.world_view_transform
    z = view_xyz[:, 2:3].clamp_min(0)
    aux_colors = torch.cat((z, ones, torch.zeros_like(ones)), dim=1)
    aux_image, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=aux_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    alpha = aux_image[1:2].clamp_min(1e-6)
    depth = aux_image[0:1] / alpha

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "feature_map": None,
        "depth": depth,
    }
