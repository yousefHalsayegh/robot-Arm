"""
pong_display.py

Displays N ALE Pong environments in a single omni.ui window inside
Isaac Sim, avoiding the SDL/GLX conflict from render_mode="human".

Usage:
    from pong_display import PongDisplay

    display = PongDisplay(num_envs=4)

    # in the loop after ale_envs[i].step()
    frame = ale_envs[i].render()   # [210, 160, 3] uint8
    display.update(i, frame)

    # flush after all envs updated each step
    simulation_app.update()
"""

import numpy as np


class PongDisplay:
    """
    Single omni.ui window showing all N Pong envs side by side.
    Uses DynamicTextureProvider — same GLX context as Isaac Sim,
    no SDL conflict.
    """

    def __init__(
        self,
        num_envs:   int,
        frame_w:    int = 160,   # native ALE Pong width
        frame_h:    int = 210,   # native ALE Pong height
        scale:      int = 2,     # display scale factor
    ):
        self.num_envs  = num_envs
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        self.disp_w    = frame_w * scale
        self.disp_h    = frame_h * scale
        self.providers = []
        self._build()

    def _build(self):
        import omni.ui as ui

        total_w = self.disp_w * self.num_envs

        self._window = ui.Window(
            "Pong Environments",
            width=total_w,
            height=self.disp_h,
        )

        with self._window.frame:
            with ui.HStack(spacing=0):
                for i in range(self.num_envs):
                    name     = f"pong_display_env{i}"
                    provider = ui.DynamicTextureProvider(name)
                    self.providers.append(provider)

                    with ui.VStack(width=self.disp_w):
                        ui.Label(
                            f"Env {i}",
                            height=16,
                            alignment=ui.Alignment.CENTER,
                        )
                        ui.ImageWithProvider(
                            provider,
                            width=self.disp_w,
                            height=self.disp_h - 16,
                        )

        # push blank black frames so all windows show immediately
        blank = np.zeros(
            (self.frame_h, self.frame_w, 4), dtype=np.uint8
        )
        for i in range(self.num_envs):
            self.providers[i].set_data_array(
                blank, [self.frame_w, self.frame_h]
            )

    def update(self, env_index: int, rgb_frame: np.ndarray):
        """
        Push a new frame for one env.

        Args:
            env_index: which env (0 to num_envs-1)
            rgb_frame: [H, W, 3] uint8 RGB from ale_env.render()
        """
        if rgb_frame is None or rgb_frame.max() == 0:
            return

        h, w, _ = rgb_frame.shape
        rgba     = np.concatenate(
            [rgb_frame, np.full((h, w, 1), 255, dtype=np.uint8)],
            axis=-1,
        )
        self.providers[env_index].set_data_array(
            np.ascontiguousarray(rgba), [w, h]
        )

    def update_all(self, ale_envs: list):
        """
        Render and push frames for all envs at once.
        Call after all ale_envs have been stepped.

        Args:
            ale_envs: list of N gymnasium ALE envs
        """
        for i, env in enumerate(ale_envs):
            frame = env.render()
            self.update(i, frame)