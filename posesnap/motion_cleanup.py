# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic F-Curve cleanup for retargeted VMD animations on MMR rigs."""

from __future__ import annotations

import math
import re
from typing import Any

from . import compat

_QUAT_PATH_RE = re.compile(r'^pose\.bones\[(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\')\]\.rotation_quaternion$')
_LOC_PATH_RE = re.compile(r'^pose\.bones\[(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\')\]\.location$')
_EULER_PATH_RE = re.compile(r'^pose\.bones\[(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\')\]\.rotation_euler$')

_POLE_BONE_TOKENS = ("ik_target", "pole")
_IK_EFFECTOR_TOKENS = ("foot_ik", "hand_ik")


def _extract_bone_name(pattern: re.Pattern[str], data_path: str) -> str | None:
    match = pattern.match(data_path or "")
    if match is None:
        return None
    raw = match.group(1) if match.group(1) is not None else match.group(2)
    if raw is None:
        return None
    return raw.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


def _fix_quaternion_flips(fcurves: Any) -> int:
    """Ensure consecutive quaternion keyframes stay in the same hemisphere (dot >= 0)."""
    groups: dict[str, dict[int, Any]] = {}
    for fc in fcurves:
        bname = _extract_bone_name(_QUAT_PATH_RE, getattr(fc, "data_path", ""))
        if bname is None:
            continue
        idx = int(getattr(fc, "array_index", -1))
        if 0 <= idx <= 3:
            groups.setdefault(bname, {})[idx] = fc

    flips_fixed = 0
    for channels in groups.values():
        if len(channels) < 4:
            continue
        frame_maps = [
            {round(float(kp.co.x), 3): kp for kp in channels[i].keyframe_points}
            for i in range(4)
        ]
        all_frames = sorted(
            set(frame_maps[0]) | set(frame_maps[1]) | set(frame_maps[2]) | set(frame_maps[3])
        )
        if len(all_frames) < 2:
            continue

        def _read_quat(frame_x: float) -> tuple[float, float, float, float]:
            return tuple(
                float(frame_maps[i][frame_x].co.y)
                if frame_x in frame_maps[i]
                else float(channels[i].evaluate(frame_x))
                for i in range(4)
            )

        prev_q = _read_quat(all_frames[0])
        changed = False
        for frame in all_frames[1:]:
            curr_q = _read_quat(frame)
            dot = sum(a * b for a, b in zip(prev_q, curr_q))
            if dot < 0.0:
                curr_q = tuple(-v for v in curr_q)
                for i in range(4):
                    kp = frame_maps[i].get(frame)
                    if kp is not None:
                        kp.co.y = -kp.co.y
                        kp.handle_left.y = -kp.handle_left.y
                        kp.handle_right.y = -kp.handle_right.y
                flips_fixed += 1
                changed = True
            prev_q = curr_q

        if changed:
            for i in range(4):
                channels[i].update()

    return flips_fixed


def _fix_euler_flips(fcurves: Any) -> int:
    """Unwrap sudden +/- 2*pi discontinuities on Euler rotation F-Curves."""
    flips_fixed = 0
    two_pi = 2.0 * math.pi
    for fc in fcurves:
        if _extract_bone_name(_EULER_PATH_RE, getattr(fc, "data_path", "")) is None:
            continue
        kps = list(fc.keyframe_points)
        if len(kps) < 2:
            continue
        changed = False
        offset = 0.0
        prev_y = float(kps[0].co.y)
        for kp in kps[1:]:
            val = float(kp.co.y) + offset
            diff = val - prev_y
            if abs(diff) > math.pi:
                turns = round(diff / two_pi)
                if turns != 0:
                    shift = -turns * two_pi
                    offset += shift
                    val += shift
                    flips_fixed += 1
            if abs(offset) > 1e-6:
                kp.co.y = val
                changed = True
            prev_y = val
        if changed:
            fc.update()
    return flips_fixed


def _find_spike_indices(
    xs: list[float],
    ys: list[float],
    cliff_tol: float,
    max_burst_keys: int = 8,
    max_burst_frames: float = 16.0,
) -> set[int]:
    """Detect 1-frame and short rectangular plateau glitch spikes between smooth anchors."""
    n = len(xs)
    if n < 3:
        return set()

    spike_indices: set[int] = set()
    i = 1
    while i < n - 1:
        left = i - 1
        matched_end = -1
        max_j = min(n - 2, i + max_burst_keys - 1)
        for j in range(max_j, i - 1, -1):
            right = j + 1
            span = xs[right] - xs[left]
            if span <= 1e-6 or span > max_burst_frames:
                continue
            if abs(ys[i] - ys[left]) <= cliff_tol or abs(ys[j] - ys[right]) <= cliff_tol:
                continue

            neighbor_gap = abs(ys[right] - ys[left])
            sign = 0
            valid = True
            for k in range(i, j + 1):
                base_k = ys[left] + (ys[right] - ys[left]) * ((xs[k] - xs[left]) / span)
                diff = ys[k] - base_k
                if abs(diff) <= cliff_tol or abs(diff) <= neighbor_gap * 2.0:
                    valid = False
                    break
                cur_sign = 1 if diff > 0.0 else -1
                if sign == 0:
                    sign = cur_sign
                elif cur_sign != sign:
                    valid = False
                    break
            if valid:
                matched_end = j
                break

        if matched_end >= i:
            spike_indices.update(range(i, matched_end + 1))
            i = matched_end + 2
        else:
            i += 1

    return spike_indices


def _remove_random_spikes(fcurves: Any) -> int:
    """Delete 1-frame and short plateau glitch keyframes across quaternion, Euler, and location curves."""
    spikes_removed = 0

    quat_groups: dict[str, dict[int, Any]] = {}
    for fc in fcurves:
        bname = _extract_bone_name(_QUAT_PATH_RE, getattr(fc, "data_path", ""))
        if bname is not None:
            idx = int(getattr(fc, "array_index", -1))
            if 0 <= idx <= 3:
                quat_groups.setdefault(bname, {})[idx] = fc

    for channels in quat_groups.values():
        spike_frames: set[float] = set()
        for fc in channels.values():
            kps = list(fc.keyframe_points)
            xs = [float(kp.co.x) for kp in kps]
            ys = [float(kp.co.y) for kp in kps]
            for idx in _find_spike_indices(xs, ys, cliff_tol=0.28):
                spike_frames.add(round(xs[idx], 3))

        if not spike_frames:
            continue

        # Remove synchronized across all 4 quaternion channels so W/X/Y/Z stay aligned.
        for fc in channels.values():
            kps = fc.keyframe_points
            to_del = [idx for idx, kp in enumerate(kps) if round(float(kp.co.x), 3) in spike_frames]
            if to_del:
                for idx in reversed(to_del):
                    kps.remove(kps[idx], fast=True)
                fc.update()
        spikes_removed += len(spike_frames)

    for fc in fcurves:
        data_path = getattr(fc, "data_path", "")
        bname = _extract_bone_name(_LOC_PATH_RE, data_path)
        if bname is not None:
            lower = bname.lower()
            is_ik = any(tok in lower for tok in _POLE_BONE_TOKENS) or any(
                tok in lower for tok in _IK_EFFECTOR_TOKENS
            )
            cliff_tol = 0.12 if is_ik else 0.15
        elif _extract_bone_name(_EULER_PATH_RE, data_path) is not None:
            cliff_tol = 0.55
        else:
            continue

        kps = fc.keyframe_points
        if len(kps) < 3:
            continue
        xs = [float(kp.co.x) for kp in kps]
        ys = [float(kp.co.y) for kp in kps]
        to_del = sorted(_find_spike_indices(xs, ys, cliff_tol=cliff_tol))
        if to_del:
            for idx in reversed(to_del):
                kps.remove(kps[idx], fast=True)
            fc.update()
            spikes_removed += len(to_del)

    return spikes_removed


def _clamp_bezier_handles(fcurves: Any) -> None:
    """Use AUTO_CLAMPED handles on Bezier bone curves to stop floor-sink and hold overshoot."""
    for fc in fcurves:
        data_path = getattr(fc, "data_path", "")
        if not data_path.startswith("pose.bones["):
            continue
        changed = False
        for kp in fc.keyframe_points:
            if kp.interpolation == "BEZIER":
                if kp.handle_left_type != "AUTO_CLAMPED" or kp.handle_right_type != "AUTO_CLAMPED":
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"
                    changed = True
        if changed:
            fc.update()


def _simplify_redundant_keyframes(fcurves: Any) -> int:
    """Remove flat or collinear intermediate keyframes that add no motion within tolerance."""
    removed = 0
    for fc in fcurves:
        data_path = getattr(fc, "data_path", "")
        if not data_path.startswith("pose.bones["):
            continue
        kps = fc.keyframe_points
        if len(kps) < 3:
            continue

        is_rot = "rotation_" in data_path
        tol = 5e-4 if is_rot else 1e-4

        to_remove: list[int] = []
        last_kept = 0
        for i in range(1, len(kps) - 1):
            x0, y0 = float(kps[last_kept].co.x), float(kps[last_kept].co.y)
            x1, y1 = float(kps[i].co.x), float(kps[i].co.y)
            x2, y2 = float(kps[i + 1].co.x), float(kps[i + 1].co.y)
            span = x2 - x0
            if span <= 1e-6:
                to_remove.append(i)
                continue
            expected = y0 + (y2 - y0) * ((x1 - x0) / span)
            if abs(y1 - expected) <= tol and abs(y1 - y0) <= tol and abs(y2 - y1) <= tol:
                to_remove.append(i)
            else:
                last_kept = i

        if to_remove:
            for idx in reversed(to_remove):
                kps.remove(kps[idx], fast=True)
            fc.update()
            removed += len(to_remove)

    return removed


def cleanup_rig_action(target_rig: Any) -> dict[str, int]:
    """Run all automatic cleanup passes on a target rig's active Action."""
    stats = {
        "quat_flips": 0,
        "euler_flips": 0,
        "ik_spikes": 0,
        "keys_removed": 0,
    }
    if target_rig is None:
        return stats

    anim_data = getattr(target_rig, "animation_data", None)
    action = getattr(anim_data, "action", None) if anim_data is not None else None
    if action is None:
        return stats

    slot = getattr(anim_data, "action_slot", None)
    fcurves = compat.get_action_fcurves(action, slot)
    if not fcurves:
        return stats

    stats["quat_flips"] = _fix_quaternion_flips(fcurves)
    stats["euler_flips"] = _fix_euler_flips(fcurves)
    stats["ik_spikes"] = _remove_random_spikes(fcurves)
    stats["quat_flips"] += _fix_quaternion_flips(fcurves)
    stats["keys_removed"] = _simplify_redundant_keyframes(fcurves)
    _clamp_bezier_handles(fcurves)
    return stats
