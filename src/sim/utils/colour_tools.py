"""
colour_tools.py

Utilities for matching sim asset colours to physical colours.

Usage:
    # measure physical colours once
    python colour_tools.py --sample --serial 032522250421

    # in game_sim.py
    from colour_tools import set_sim_colours, randomise_asset_colours
"""

import numpy as np


def _find_and_set_colour(stage, prim_path: str, r: float, g: float, b: float):
        from pxr import UsdShade, Gf, Usd

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"prim not found: {prim_path}")
            return False

        found = False
        for child in Usd.PrimRange(prim):
            shader = UsdShade.Shader(child)
            if not shader.GetPrim().IsValid():
                continue

            # check both naming conventions
            for input_name in ["diffuse_color_constant", "diffuseColor"]:
                inp = shader.GetInput(input_name)
                if inp:
                    inp.Set(Gf.Vec3f(r, g, b))
                    found = True

        if not found:
            print(f"no diffuse input found under {prim_path}")
        return found

def set_robot_colour(stage, env_index: int, r: float, g: float, b: float):
    """Change only the 3D printed parts of the SO101, keep servos dark."""
    from pxr import UsdShade, Gf

    shader_path = f"/World/envs/env_{env_index}/Robot/Looks/material_a_d_printed/Shader"
    shader      = UsdShade.Shader.Get(stage, shader_path)
    if shader.GetPrim().IsValid():
        shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(r, g, b))

def tint_arcade_stick(stage, env_index: int, scale: float = 1.0):
    from pxr import UsdShade, Gf, Sdf

    base_path    = f"/World/envs/env_{env_index}/object/Looks/UV_Grid"
    texture_path = f"{base_path}/diffuseTex"
    surface_path = f"{base_path}/UV_Grid"

    texture = UsdShade.Shader.Get(stage, texture_path)
    surface = UsdShade.Shader.Get(stage, surface_path)

    if not texture.GetPrim().IsValid():
        print(f"texture shader not found: {texture_path}")
        return
    if not surface.GetPrim().IsValid():
        print(f"surface shader not found: {surface_path}")
        return

    # get or create the scale input — Set() fails if input doesn't exist
    scale_inp = texture.GetInput("scale")
    if not scale_inp:
        scale_inp = texture.CreateInput("scale", Sdf.ValueTypeNames.Float4)
    scale_inp.Set(Gf.Vec4f(scale, scale, scale, 1.0))

    # ensure diffuseColor is connected to the texture rgb output
    diffuse    = surface.GetInput("diffuseColor")
    rgb_output = texture.GetOutput("rgb")
    if not rgb_output:
        rgb_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    diffuse.ConnectToSource(rgb_output)
    
def set_sim_colours(
    stage,
    num_envs:   int,
    table_rgb:  tuple = (0.72, 0.58, 0.42),
    table_prim: str = None,
):
    """
    Apply physically-measured colours to all env instances.
    Call once after the scene loads.

    Replace the default RGB tuples with values from:
        python colour_tools.py --sample

    Args:
        table_prim: full path if table is shared e.g. "/World/Table",
                    or relative name if per-env e.g. "Table"
    """
    for i in range(num_envs):
        if table_prim:
            # handle both absolute and relative table paths
            path = table_prim if table_prim.startswith("/") \
                   else f"/World/envs/env_{i}/{table_prim}"
            _find_and_set_colour(stage, path, *table_rgb)

    print(f"colours applied to {num_envs} envs")


def randomise_asset_colours(
    stage,
    env_index:   int,
    arm_range:   tuple = (0.75, 0.95),
    stick_range: tuple = (0.05, 0.20),
    table_range: tuple = (0.60, 0.85),
    arm_prim:    str = "Robot",
    stick_prim:  str = "arcade_stick",
    table_prim:  str = None,
):
    """
    Randomise colours each episode to bridge the sim-to-real gap.
    Call at episode reset in game_sim.py.
    """
    import random

    v = random.uniform(*arm_range)
    _find_and_set_colour(
        stage, f"/World/envs/env_{env_index}/{arm_prim}",
        v, v * 0.90, v * 0.78
    )

    v = random.uniform(*stick_range)
    _find_and_set_colour(
        stage, f"/World/envs/env_{env_index}/{stick_prim}",
        v, v * 0.95, v * 1.05
    )

    if table_prim:
        v    = random.uniform(*table_range)
        path = table_prim if table_prim.startswith("/") \
               else f"/World/envs/env_{env_index}/{table_prim}"
        _find_and_set_colour(stage, path, v, v * 0.82, v * 0.62)

def set_robot_colour(stage, env_index: int, r: float, g: float, b: float):
    """Change only the 3D printed parts of the SO101, keep servos dark."""
    from pxr import UsdShade, Gf

    shader_path = f"/World/envs/env_{env_index}/Robot/Looks/material_a_d_printed/Shader"
    shader      = UsdShade.Shader.Get(stage, shader_path)
    if shader.GetPrim().IsValid():
        shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(r, g, b))
        
def sample_physical_colours(serial: str = "032522250421"):
    """
    Open RealSense stream and click to sample RGB values.
    Prints values ready to paste into set_sim_colours().
    """
    import pyrealsense2 as rs
    import cv2

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    frame_data = [None]

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and frame_data[0] is not None:
            bgr = frame_data[0][y, x]
            rgb = bgr[::-1] / 255.0
            print(f"({x},{y}) → RGB: {rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}")

    cv2.namedWindow("click to sample — q to quit")
    cv2.setMouseCallback("click to sample — q to quit", click)
    print("click on arm, arcade stick, table. Press q when done.")

    try:
        while True:
            frames       = pipeline.wait_for_frames()
            frame_data[0] = np.asanyarray(
                frames.get_color_frame().get_data()
            )
            cv2.imshow("click to sample — q to quit", frame_data[0])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--serial", type=str, default="032522250421")
    args = parser.parse_args()
    if args.sample:
        sample_physical_colours(serial=args.serial)