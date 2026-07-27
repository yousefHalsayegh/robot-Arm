"""
Debug visualization for ball_position()'s detection.

Shows the actual 84x84 preprocessed frame that ball_position() operates
on (not the native rendered Pong frame PongDisplay shows), with the
detected court/player crop boxes outlined and colored dots drawn at the
detected ball/paddle positions -- lets you visually verify the crop
bounds and intensity thresholds are picking up the right pixels, live,
directly in the Isaac Sim UI.

Uses the same omni.ui DynamicTextureProvider approach as PongDisplay, so
there's no separate SDL window / GLX context conflict.
"""
import numpy as np


# Same crop bounds/thresholds as the production ball_position() --
# duplicated here (not imported) so this module has no dependency on
# wherever ball_position() lives; keep these in sync if you change the
# crops/thresholds in the production function.
COURT_BOUNDS = (15, 77, 12, 71)    # (row_start, row_end, col_start, col_end)
PLAYER_BOUNDS = (15, 77, 72, 76)
BALL_THRESHOLD = (0.4, 0.9)
PADDLE_THRESHOLD = (0.5, 0.7)


def ball_position_debug(obs):
    """
    Same detection logic as the production ball_position(), but also
    returns the x (column) coordinate and converts both y and x back to
    full-frame coordinates (production ball_position()'s y values are
    each relative to their own crop's top row -- fine for a diff-based
    proportional controller, but not directly usable for drawing an
    accurate marker on the full frame without this correction).

    Returns: ball_y, ball_x, paddle_y, paddle_x (all None if not
    detected, all in full-frame pixel coordinates).
    """
    r0, r1, c0, c1 = COURT_BOUNDS
    court = obs[r0:r1, c0:c1]
    pr0, pr1, pc0, pc1 = PLAYER_BOUNDS
    player = obs[pr0:pr1, pc0:pc1]

    lo, hi = BALL_THRESHOLD
    ball_pixels = np.argwhere((court > lo) & (court < hi))
    if len(ball_pixels) > 0:
        ball_y = float(np.mean(ball_pixels[:, 0])) + r0
        ball_x = float(np.mean(ball_pixels[:, 1])) + c0
    else:
        ball_y = None
        ball_x = None

    lo, hi = PADDLE_THRESHOLD
    paddle_pixels = np.argwhere((player > lo) & (player < hi))
    if len(paddle_pixels) > 0:
        paddle_y = float(np.mean(paddle_pixels[:, 0])) + pr0
        paddle_x = float(np.mean(paddle_pixels[:, 1])) + pc0
    else:
        paddle_y = None
        paddle_x = None

    return ball_y, ball_x, paddle_y, paddle_x


class PongDebugDisplay:
    """
    Separate small window (one panel per env) showing the actual
    preprocessed frame with the court/player crop boxes outlined and the
    detected ball (red) / paddle (yellow) positions marked.

    Docks as a tab alongside an existing panel (default: "Property")
    instead of opening as a separate floating window.

    update_all() is throttled to update_every_n_calls (default 4) --
    pushing a new texture to the GPU on every single physics step is
    unnecessary for a debug view and adds continuous GPU memory/bandwidth
    pressure; throttling this is a cheap way to reduce that load if it's
    contributing to instability.
    """

    def __init__(self, num_envs: int, frame_size: int = 84, scale: int = 4,
                 update_every_n_calls: int = 4, dock_next_to: str = "Property"):
        self.num_envs = num_envs
        self.frame_size = frame_size
        self.scale = scale
        self.disp_size = frame_size * scale
        self.update_every_n_calls = max(1, update_every_n_calls)
        self.dock_next_to = dock_next_to
        self._call_count = 0
        self.providers = []
        self._build()

    def _build(self):
        import omni.ui as ui

        total_w = self.disp_size * self.num_envs

        self._window = ui.Window(
            "Pong Ball/Paddle Debug",
            width=total_w,
            height=self.disp_size + 16,
        )

        # dock as a tab next to an existing panel rather than floating
        self._window.deferred_dock_in(
            self.dock_next_to, ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE
        )

        with self._window.frame:
            with ui.HStack(spacing=0):
                for i in range(self.num_envs):
                    name = f"pong_debug_env{i}"
                    provider = ui.DynamicTextureProvider(name)
                    self.providers.append(provider)

                    with ui.VStack(width=self.disp_size):
                        ui.Label(
                            f"Env {i} debug",
                            height=16,
                            alignment=ui.Alignment.CENTER,
                        )
                        ui.ImageWithProvider(
                            provider,
                            width=self.disp_size,
                            height=self.disp_size,
                        )

        blank = np.zeros((self.disp_size, self.disp_size, 4), dtype=np.uint8)
        for i in range(self.num_envs):
            self.providers[i].set_data_array(blank, [self.disp_size, self.disp_size])

    def _draw_marker(self, rgb_big, y, x, color, radius=4):
        if y is None or x is None:
            return
        cy, cx = int(round(y * self.scale)), int(round(x * self.scale))
        y0, y1 = max(cy - radius, 0), min(cy + radius, rgb_big.shape[0])
        x0, x1 = max(cx - radius, 0), min(cx + radius, rgb_big.shape[1])
        rgb_big[y0:y1, x0:x1] = color

    def _draw_box(self, rgb_big, bounds, color, thickness=2):
        r0, r1, c0, c1 = bounds
        r0s, r1s = r0 * self.scale, r1 * self.scale
        c0s, c1s = c0 * self.scale, c1 * self.scale
        r0s, r1s = max(r0s, 0), min(r1s, rgb_big.shape[0])
        c0s, c1s = max(c0s, 0), min(c1s, rgb_big.shape[1])
        rgb_big[r0s:r0s + thickness, c0s:c1s] = color
        rgb_big[max(r1s - thickness, 0):r1s, c0s:c1s] = color
        rgb_big[r0s:r1s, c0s:c0s + thickness] = color
        rgb_big[r0s:r1s, max(c1s - thickness, 0):c1s] = color

    def update(self, env_index: int, obs_frame: np.ndarray,
               ball_y=None, ball_x=None, paddle_y=None, paddle_x=None):
        """
        Draws the current 84x84 grayscale obs_frame (values 0-1),
        upscaled by `scale`, with the court/player crop boxes outlined
        and markers at the detected ball/paddle positions.

        Args:
            env_index: which env panel to update
            obs_frame: [84, 84] float array (the same frame passed to
                ball_position()), values in [0, 1]
            ball_y, ball_x, paddle_y, paddle_x: full-frame pixel
                coordinates, e.g. from ball_position_debug(obs_frame).
                Any of these can be None if not detected -- that marker
                is simply skipped.
        """
        gray = np.clip(obs_frame, 0.0, 1.0)
        rgb = (np.stack([gray, gray, gray], axis=-1) * 255).astype(np.uint8)

        # nearest-neighbor upscale -- fine for a debug view
        rgb_big = np.repeat(np.repeat(rgb, self.scale, axis=0), self.scale, axis=1)
        rgb_big = np.ascontiguousarray(rgb_big)

        self._draw_box(rgb_big, COURT_BOUNDS, np.array([0, 255, 0], dtype=np.uint8))
        self._draw_box(rgb_big, PLAYER_BOUNDS, np.array([0, 150, 255], dtype=np.uint8))

        self._draw_marker(rgb_big, ball_y, ball_x, np.array([255, 0, 0], dtype=np.uint8))
        self._draw_marker(rgb_big, paddle_y, paddle_x, np.array([255, 255, 0], dtype=np.uint8))

        h, w, _ = rgb_big.shape
        rgba = np.concatenate([rgb_big, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
        self.providers[env_index].set_data_array(np.ascontiguousarray(rgba), [w, h])

    def update_all(self, obs_batch):
        """
        Convenience wrapper: given a batch of obs frames (one per env,
        same [4, 84, 84] stacked format your main loop already uses),
        runs ball_position_debug() and updates every panel -- throttled
        to self.update_every_n_calls (skips most calls entirely, doing
        no work and no GPU texture upload on skipped calls).

        Args:
            obs_batch: list/array of N frames, each [84, 84] (already
                the latest frame from the stack, e.g. states[i][-1])
        """
        self._call_count += 1
        if self._call_count % self.update_every_n_calls != 0:
            return

        for i, obs_frame in enumerate(obs_batch):
            ball_y, ball_x, paddle_y, paddle_x = ball_position_debug(obs_frame)
            self.update(i, obs_frame, ball_y, ball_x, paddle_y, paddle_x)