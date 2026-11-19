"""Model input and targets, exactly as the C++ client (DigitalCurling4-CNN) builds and reads them.

Input: (N, 18, 32, 16) float, see `GameStateToInput` in `src/engine/utility.hpp`:
  - channel 0: stones of the team without the hammer, channel 1: the hammer team's stones,
    one pixel per stone on a 16-inch grid (`position_to_pixel`);
  - channels 2..17: shot-number planes, channel l + 2 is all ones for l <= shot (0..15 in the end).

Outputs: a tuple (policy logits (N, 1024), score logits (N, 17)):
  - policy index = rotation * 512 + speed * 32 + angle (`PixelToShot`). Rotation 0 is
    angular velocity +1 in the client, but the client sends `-angular_velocity` to the server
    (`client_base.cpp`), so in the server's logs rotation 0 = angular velocity -1;
  - only legal shots are scored (`LEGAL_MASK`, `createFilter()` in `search_policy.hpp`);
  - score class = (hammer team's points - other team's points in this end) + 8, clipped to 0..16.

Stones are stored compactly as pixel coordinates (`stone_pixels`, -1 = no stone) and expanded
into the input planes on the GPU per batch (`encode_batch`).
"""

from __future__ import annotations

import math

import numpy as np

N_CHANNEL = 18
HEIGHT = 32
WIDTH = 16
N_SHOTS = 16  # shots per end

HOG_Y = 32.004
TEE_Y = 38.405
CELL = 0.0254 * 16  # one pixel = 16 inches

POLICY_ROTATION = 2
POLICY_SPEED = 16
POLICY_ANGLE = 32
N_ACTIONS = POLICY_ROTATION * POLICY_SPEED * POLICY_ANGLE
SPEEDS = np.array([2.23, 2.27, 2.31, 2.35, 2.37, 2.385, 2.4, 2.415, 2.43, 2.45, 2.485, 2.535, 2.6, 3.0, 3.4, 3.8])
ANGLES = np.array([math.atan(-(k - POLICY_ANGLE // 2) * 0.0254 * 5 / TEE_Y) + math.pi / 2 for k in range(POLICY_ANGLE)])

# createFilter() in search_policy.hpp: rotation 1 only for angle index < 14, rotation 0 only for
# angle index >= 18, either rotation for speed index > 10 (takeouts).
LEGAL_MASK = np.zeros((POLICY_ROTATION, POLICY_SPEED, POLICY_ANGLE), dtype=bool)
LEGAL_MASK[1, :, :14] = True
LEGAL_MASK[0, :, 18:] = True
LEGAL_MASK[:, 11:, :] = True
LEGAL_MASK = LEGAL_MASK.reshape(-1)

SCORE_OFFSET = 8
N_SCORE = 2 * SCORE_OFFSET + 1


def _c_round(v: float) -> int:
    """C's round(): halves away from zero (numpy rounds halves to even)."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def position_to_pixel(x: float, y: float) -> tuple[int, int]:
    """(row, col) of a stone; may be out of the grid."""
    return _c_round((y - HOG_Y) / CELL), WIDTH // 2 - _c_round(x / CELL)


def stone_pixels(stones: dict, hammer: str) -> np.ndarray:
    """(2, 8, 2) int8 pixel coordinates, [0] = non-hammer team, [1] = hammer team; -1 = no stone.

    `stones` is the server's `stone_coordinate.data`: {"team0": [{"x", "y"}, ...], "team1": [...]},
    where (0, 0) means not in play. Stones outside the grid are dropped, as in the client.
    """
    out = np.full((2, 8, 2), -1, dtype=np.int8)
    other = "team1" if hammer == "team0" else "team0"
    for ch, team in enumerate((other, hammer)):
        for i, s in enumerate((stones.get(team) or [])[:8]):
            if s["x"] == 0 and s["y"] == 0:
                continue
            r, c = position_to_pixel(s["x"], s["y"])
            if 0 <= r < HEIGHT and 0 <= c < WIDTH:
                out[ch, i] = (r, c)
    return out


def encode_batch(pixels, shot):
    """Model input from `stone_pixels` (B, 2, 8, 2) and shot numbers (B,), as torch tensors."""
    import torch

    b = pixels.shape[0]
    x = torch.zeros((b, N_CHANNEL, HEIGHT, WIDTH), device=pixels.device)
    pix = pixels.long()
    valid = pix[..., 0] >= 0  # (B, 2, 8)
    bi, ch, si = valid.nonzero(as_tuple=True)
    x[bi, ch, pix[bi, ch, si, 0], pix[bi, ch, si, 1]] = 1.0
    planes = torch.arange(N_SHOTS, device=pixels.device)[None, :] <= shot.long()[:, None]  # (B, 16)
    x[:, 2:] = planes[:, :, None, None].float()
    return x


def encode(stones: dict, hammer: str, shot: int):
    """Single (1, 18, 32, 16) input, for checks and notebooks."""
    import torch

    return encode_batch(torch.from_numpy(stone_pixels(stones, hammer))[None], torch.tensor([shot]))


def shot_to_action(speed: float, spin: float, angle: float, tol: float = 1e-3) -> int:
    """Policy index of a shot as the server logs it (`actual_*`); -1 if it is not on the policy grid.

    `spin` is the server's angular velocity, the negative of the client's (see the module doc).
    """
    v = int(np.argmin(np.abs(SPEEDS - speed)))
    a = int(np.argmin(np.abs(ANGLES - angle)))
    if abs(SPEEDS[v] - speed) > tol or abs(ANGLES[a] - angle) > tol or spin == 0:
        return -1
    rotation = 0 if spin < 0 else 1
    return rotation * POLICY_SPEED * POLICY_ANGLE + v * POLICY_ANGLE + a


def action_to_shot(action: int) -> tuple[float, float, float]:
    """(speed, spin as the server logs it, angle) of a policy index; inverse of `shot_to_action`."""
    rotation, rest = divmod(action, POLICY_SPEED * POLICY_ANGLE)
    v, a = divmod(rest, POLICY_ANGLE)
    return float(SPEEDS[v]), -1.0 if rotation == 0 else 1.0, float(ANGLES[a])


def score_class(hammer_points: int) -> int:
    """Score-head class of the hammer team's net points in an end."""
    return int(np.clip(hammer_points, -SCORE_OFFSET, SCORE_OFFSET)) + SCORE_OFFSET


# --- left-right mirror (data augmentation) ---
# x -> -x is a symmetry of the sheet: the mirrored position, with each shot's angle mirrored and its
# curl reversed, plays out the same, and the end score is unchanged. On the grids: stone column
# c -> 16 - c (column 8 is the centre line; stones in play use columns 3..13), shot angle index
# k -> 32 - k (16 is straight) with the other rotation. Index 0 has no mirror on the grid, and the
# client's legal mask is symmetric about 15.5 rather than 16, so a few mirrored shots aren't
# playable: those samples keep their score target and lose their policy target.


def mirror_pixels(pixels):
    """`stone_pixels` arrays (..., 2, 8, 2) mirrored left-right; -1 (no stone) stays -1."""
    out = pixels.copy()
    present = out[..., 0] >= 0
    out[..., 1] = np.where(present, WIDTH - out[..., 1], -1)
    return out


def mirror_actions(actions):
    """Policy indices mirrored left-right; -1 where the mirror is off the grid or not legal."""
    a = np.asarray(actions).astype(np.int64)
    rotation, rest = np.divmod(np.maximum(a, 0), POLICY_SPEED * POLICY_ANGLE)
    speed, angle = np.divmod(rest, POLICY_ANGLE)
    angle2 = POLICY_ANGLE - angle
    out = (1 - rotation) * POLICY_SPEED * POLICY_ANGLE + speed * POLICY_ANGLE + np.minimum(angle2, POLICY_ANGLE - 1)
    ok = (a >= 0) & (angle2 < POLICY_ANGLE)
    ok &= LEGAL_MASK[np.where(ok, out, 0)]
    return np.where(ok, out, -1)
