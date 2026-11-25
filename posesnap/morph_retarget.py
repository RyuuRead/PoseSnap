# SPDX-License-Identifier: GPL-3.0-or-later
"""Shape key retargeting with automatic MorphRig driver and 2D joystick resolution."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import re
import unicodedata
from typing import Any

import bpy

from . import compat

_AXIS_MAP = {
    "LOC_X": 0,
    "LOC_Y": 1,
    "LOC_Z": 2,
}

_DATA_PATH_RE = re.compile(r'^key_blocks\[(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\')\]\.value$')
_NUM_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class DriverBinding:
    armature: Any
    bone_name: str
    axis_index: int
    c0: float
    c1: float


def _normalize_morph_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().lower()


def _extract_key_block_name(data_path: str) -> str | None:
    match = _DATA_PATH_RE.match(data_path or "")
    if match is None:
        return None
    raw = match.group(1) if match.group(1) is not None else match.group(2)
    if raw is None:
        return None
    return raw.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


def _parse_linear_expression(expression: str, var_name: str) -> tuple[float, float] | None:
    """Parse simple single-variable linear driver expressions into (c0, c1) without eval."""
    expr = (expression or "").strip()
    if not expr or not var_name:
        return None

    # Unwrap outer clamp/max/min wrappers commonly used on slider drivers.
    for prefix in ("clamp(", "max(0.0,", "max(0,"):
        if expr.startswith(prefix) and expr.endswith(")"):
            inner = expr[len(prefix):-1]
            parts = [p.strip() for p in inner.split(",")]
            if prefix == "clamp(" and parts:
                expr = parts[0]
            elif len(parts) >= 2:
                expr = parts[1]
            break

    while expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1].strip()

    v = re.escape(var_name)
    m = re.fullmatch(rf"([+-]?)\s*{v}", expr)
    if m:
        return 0.0, (-1.0 if m.group(1) == "-" else 1.0)

    m = re.fullmatch(rf"([+-]?)\s*{v}\s*([*/])\s*([+-]?{_NUM_PATTERN})", expr)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        op = m.group(2)
        val = float(m.group(3))
        if op == "*":
            return 0.0, sign * val
        if abs(val) > 1e-9:
            return 0.0, sign / val
        return None

    m = re.fullmatch(rf"([+-]?{_NUM_PATTERN})\s*\*\s*([+-]?)\s*{v}", expr)
    if m:
        sign = -1.0 if m.group(2) == "-" else 1.0
        return 0.0, sign * float(m.group(1))

    return None


def _is_descendant_of(obj: Any, ancestor: Any) -> bool:
    curr = getattr(obj, "parent", None)
    visited: set[int] = set()
    while curr is not None:
        cid = id(curr)
        if cid in visited:
            return False
        visited.add(cid)
        if curr == ancestor:
            return True
        curr = getattr(curr, "parent", None)
    return False


def _collect_related_armatures(arm_obj: Any, candidates: Any) -> set[Any]:
    """Collect arm_obj, descendant armatures (e.g. MorphRig boards), and armatures constrained to it."""
    related: set[Any] = {arm_obj}
    for obj in candidates:
        if getattr(obj, "type", None) != "ARMATURE" or obj in related:
            continue
        if _is_descendant_of(obj, arm_obj):
            related.add(obj)
            continue
        pose = getattr(obj, "pose", None)
        if pose is None:
            continue
        for pbone in getattr(pose, "bones", ()):
            for con in getattr(pbone, "constraints", ()):
                if getattr(con, "mute", False):
                    continue
                if getattr(con, "target", None) == arm_obj:
                    related.add(obj)
                    break
            if obj in related:
                break
    return related


def _mesh_has_driver_for_armatures(shape_keys: Any, related_ids: set[int]) -> bool:
    anim_data = getattr(shape_keys, "animation_data", None)
    drivers = getattr(anim_data, "drivers", None) if anim_data is not None else None
    if not drivers:
        return False
    for fcurve in drivers:
        if getattr(fcurve, "mute", False):
            continue
        driver = getattr(fcurve, "driver", None)
        if driver is None:
            continue
        for var in getattr(driver, "variables", ()):
            if getattr(var, "type", None) != "TRANSFORMS":
                continue
            for tgt in getattr(var, "targets", ()):
                if id(getattr(tgt, "id", None)) in related_ids:
                    return True
    return False


def find_armature_meshes(arm_obj: Any, context: Any = None) -> list[Any]:
    """Return meshes with shape keys linked to arm_obj via modifier, parenting, constraints, or drivers."""
    if arm_obj is None:
        return []

    scene = getattr(context, "scene", None) if context is not None else None
    candidates = list(getattr(scene, "objects", ())) if scene is not None else []
    if not candidates:
        candidates = list(getattr(bpy.data, "objects", ()))

    related_arms = _collect_related_armatures(arm_obj, candidates)
    related_ids = {id(a) for a in related_arms}
    named_targets: set[str] = set()
    for arm in related_arms:
        if hasattr(arm, "get"):
            tname = arm.get("morphrig_target_mesh")
            if isinstance(tname, str) and tname:
                named_targets.add(tname)

    meshes: list[Any] = []
    seen: set[int] = set()

    for obj in candidates:
        if getattr(obj, "type", None) != "MESH":
            continue
        oid = id(obj)
        if oid in seen:
            continue
        data = getattr(obj, "data", None)
        shape_keys = getattr(data, "shape_keys", None) if data is not None else None
        key_blocks = getattr(shape_keys, "key_blocks", None) if shape_keys is not None else None
        if not key_blocks or len(key_blocks) <= 1:
            continue

        bound = obj.name in named_targets
        if not bound:
            for arm in related_arms:
                if _is_descendant_of(obj, arm):
                    bound = True
                    break
        if not bound:
            for mod in getattr(obj, "modifiers", ()):
                if getattr(mod, "type", None) == "ARMATURE" and id(getattr(mod, "object", None)) in related_ids:
                    bound = True
                    break
        if not bound:
            bound = _mesh_has_driver_for_armatures(shape_keys, related_ids)

        if bound:
            seen.add(oid)
            meshes.append(obj)

    return meshes


def capture_armature_morphs(src_arm: Any, context: Any = None) -> dict[str, float]:
    """Capture current shape key weights from meshes bound to a source armature."""
    morphs: dict[str, float] = {}
    for mesh_obj in find_armature_meshes(src_arm, context):
        shape_keys = mesh_obj.data.shape_keys
        for kb in shape_keys.key_blocks[1:]:
            val = float(kb.value)
            existing = morphs.get(kb.name)
            if existing is None or abs(val) > abs(existing):
                morphs[kb.name] = val
    return morphs


def _inspect_fcurve_binding(fcurve: Any) -> DriverBinding | None:
    if getattr(fcurve, "mute", False):
        return None
    driver = getattr(fcurve, "driver", None)
    if driver is None:
        return None

    target_var = None
    target_arm = None
    bone_name = ""
    axis_index = None

    for var in getattr(driver, "variables", ()):
        if getattr(var, "type", None) != "TRANSFORMS":
            continue
        targets = getattr(var, "targets", ())
        if not targets:
            continue
        tgt = targets[0]
        arm_obj = getattr(tgt, "id", None)
        bname = getattr(tgt, "bone_target", "") or ""
        t_type = getattr(tgt, "transform_type", "")
        if getattr(arm_obj, "type", None) != "ARMATURE" or not bname or t_type not in _AXIS_MAP:
            continue
        pose = getattr(arm_obj, "pose", None)
        if pose is None or pose.bones.get(bname) is None:
            continue
        target_var = var
        target_arm = arm_obj
        bone_name = bname
        axis_index = _AXIS_MAP[t_type]
        break

    if target_var is None or target_arm is None or axis_index is None:
        return None

    c0: float | None = None
    c1: float | None = None
    for mod in getattr(fcurve, "modifiers", ()):
        if getattr(mod, "mute", False):
            continue
        if getattr(mod, "type", None) == "GENERATOR" and getattr(mod, "mode", None) == "POLYNOMIAL":
            if getattr(mod, "poly_order", 0) >= 1 and len(mod.coefficients) >= 2:
                c0 = float(mod.coefficients[0])
                c1 = float(mod.coefficients[1])
                break

    if c1 is None:
        drv_type = getattr(driver, "type", "AVERAGE")
        if drv_type == "SCRIPTED":
            parsed = _parse_linear_expression(getattr(driver, "expression", ""), target_var.name)
            if parsed is None:
                return None
            c0, c1 = parsed
        elif drv_type in ("AVERAGE", "SUM", "MIN", "MAX"):
            c0, c1 = 0.0, 1.0
        else:
            return None

    if c0 is None or abs(c1) < 1e-9:
        return None

    return DriverBinding(
        armature=target_arm,
        bone_name=bone_name,
        axis_index=axis_index,
        c0=c0,
        c1=c1,
    )


def build_shape_key_driver_map(shape_keys: Any) -> dict[str, DriverBinding]:
    """Map shape key names to their active armature bone driver bindings."""
    bindings: dict[str, DriverBinding] = {}
    anim_data = getattr(shape_keys, "animation_data", None)
    drivers = getattr(anim_data, "drivers", None) if anim_data is not None else None
    if not drivers:
        return bindings

    for fcurve in drivers:
        kb_name = _extract_key_block_name(getattr(fcurve, "data_path", ""))
        if not kb_name:
            continue
        binding = _inspect_fcurve_binding(fcurve)
        if binding is not None:
            bindings[kb_name] = binding
    return bindings


def _clamp_to_bone_limits(pose_bone: Any, axis_index: int, value: float) -> float:
    axis_char = ("x", "y", "z")[axis_index]
    use_min_attr = f"use_min_{axis_char}"
    use_max_attr = f"use_max_{axis_char}"
    min_attr = f"min_{axis_char}"
    max_attr = f"max_{axis_char}"

    result = float(value)
    for con in getattr(pose_bone, "constraints", ()):
        if getattr(con, "mute", False) or getattr(con, "type", None) != "LIMIT_LOCATION":
            continue
        if getattr(con, "owner_space", "LOCAL") != "LOCAL":
            continue
        if getattr(con, use_min_attr, False):
            result = max(result, float(getattr(con, min_attr)))
        if getattr(con, use_max_attr, False):
            result = min(result, float(getattr(con, max_attr)))
    return result


def apply_morphs_to_target(
    context: Any,
    target_rig: Any,
    morphs: dict[str, float],
    insert_keyframes: bool = False,
    frame: int | None = None,
    meshes: list[Any] | None = None,
) -> dict[str, int]:
    """Apply morph weights to target meshes, routing driven shape keys to MorphRig bones."""
    stats = {
        "morphs_driven": 0,
        "morphs_direct": 0,
        "morph_bones": 0,
    }
    if not morphs or target_rig is None:
        return stats

    target_meshes = meshes if meshes is not None else find_armature_meshes(target_rig, context)
    if not target_meshes:
        return stats

    scene = getattr(context, "scene", None) if context is not None else None
    tool_settings = getattr(scene, "tool_settings", None) if scene is not None else None
    auto_key = bool(getattr(tool_settings, "use_keyframe_insert_auto", False))
    do_keyframe = bool(insert_keyframes or auto_key)
    key_frame = float(frame) if frame is not None else float(getattr(scene, "frame_current", 1))

    axis_contributions: dict[tuple[int, str, int], list[tuple[DriverBinding, float]]] = {}
    arm_by_id: dict[int, Any] = {}

    for mesh_obj in target_meshes:
        shape_keys = mesh_obj.data.shape_keys
        key_blocks = shape_keys.key_blocks
        norm_index = {
            _normalize_morph_name(kb.name): kb
            for kb in key_blocks[1:]
        }
        driver_map = build_shape_key_driver_map(shape_keys)

        for morph_name, raw_weight in morphs.items():
            kb = key_blocks.get(morph_name)
            if kb is None:
                kb = norm_index.get(_normalize_morph_name(morph_name))
            if kb is None or kb == key_blocks[0]:
                continue

            lo = float(getattr(kb, "slider_min", 0.0))
            hi = float(getattr(kb, "slider_max", 1.0))
            weight = max(lo, min(hi, float(raw_weight)))

            binding = driver_map.get(kb.name)
            if binding is not None:
                coord = (weight - binding.c0) / binding.c1
                arm_id = id(binding.armature)
                arm_by_id[arm_id] = binding.armature
                key = (arm_id, binding.bone_name, binding.axis_index)
                axis_contributions.setdefault(key, []).append((binding, coord))
                stats["morphs_driven"] += 1
            else:
                kb.value = weight
                if do_keyframe:
                    compat.insert_shape_key_keyframe(kb, frame=key_frame)
                stats["morphs_direct"] += 1

    touched_bones: set[tuple[int, str]] = set()
    for (arm_id, bone_name, axis_index), entries in axis_contributions.items():
        arm_obj = arm_by_id[arm_id]
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue

        pos_coords = [coord for binding, coord in entries if binding.c1 > 0.0]
        neg_coords = [coord for binding, coord in entries if binding.c1 < 0.0]
        if pos_coords and neg_coords:
            # Opposing poles on a shared joystick axis combine into a net signed coordinate.
            combined = max(pos_coords) + min(neg_coords)
        elif pos_coords:
            combined = max(pos_coords)
        else:
            combined = min(neg_coords)

        pbone.location[axis_index] = _clamp_to_bone_limits(pbone, axis_index, combined)
        touched_bones.add((arm_id, bone_name))

    if do_keyframe and touched_bones:
        for arm_id, bone_name in touched_bones:
            arm_obj = arm_by_id[arm_id]
            pbone = arm_obj.pose.bones.get(bone_name)
            if pbone is None:
                continue
            compat.ensure_action_and_slot(arm_obj)
            pbone.keyframe_insert(data_path="location", frame=key_frame, group=pbone.name)

    stats["morph_bones"] = len(touched_bones)
    return stats


def _sample_key_weight(keys: list[Any], frames: list[int], frame: int) -> float:
    if frame <= frames[0]:
        return float(keys[0].weight)
    if frame >= frames[-1]:
        return float(keys[-1].weight)
    idx = bisect_right(frames, frame)
    k0 = keys[idx - 1]
    k1 = keys[idx]
    if k0.frame == frame:
        return float(k0.weight)
    t = (frame - k0.frame) / float(k1.frame - k0.frame)
    return float(k0.weight) + (float(k1.weight) - float(k0.weight)) * t


def _set_linear_interpolation(id_block: Any, data_paths: set[str], target_frames: set[float]) -> None:
    anim_data = getattr(id_block, "animation_data", None)
    action = getattr(anim_data, "action", None) if anim_data is not None else None
    if action is None:
        return
    slot = getattr(anim_data, "action_slot", None)
    fcurves = compat.get_action_fcurves(action, slot)
    if not fcurves:
        return
    for fc in fcurves:
        if fc.data_path not in data_paths:
            continue
        changed = False
        for kp in fc.keyframe_points:
            if any(abs(kp.co.x - f) < 1e-4 for f in target_frames):
                kp.interpolation = "LINEAR"
                changed = True
        if changed:
            fc.update()


def apply_vmd_morph_tracks_sparse(
    context: Any,
    target_rig: Any,
    morph_tracks: dict[str, list[Any]],
    frame_start: int,
    frame_end: int,
    frame_offset: int = 1,
    meshes: list[Any] | None = None,
) -> dict[str, int]:
    """Insert keyframes only on original VMD morph frames per MorphRig bone or shape key."""
    stats = {
        "morphs_driven": 0,
        "morphs_direct": 0,
        "morph_bones": 0,
        "morph_keyframes": 0,
    }
    if not morph_tracks or target_rig is None:
        return stats

    target_meshes = meshes if meshes is not None else find_armature_meshes(target_rig, context)
    if not target_meshes:
        return stats

    # Group driven morph tracks by target bone so each slider or 2D joystick gets only its own VMD frames.
    bone_groups: dict[tuple[int, str], list[tuple[Any, DriverBinding, list[Any], list[int]]]] = {}
    arm_by_id: dict[int, Any] = {}
    direct_tracks: list[tuple[Any, Any, list[Any], list[int]]] = []
    seen_driven_keys: set[tuple[int, str, int, str]] = set()

    for mesh_obj in target_meshes:
        shape_keys = mesh_obj.data.shape_keys
        key_blocks = shape_keys.key_blocks
        norm_index = {_normalize_morph_name(kb.name): kb for kb in key_blocks[1:]}
        driver_map = build_shape_key_driver_map(shape_keys)

        for morph_name, keys in morph_tracks.items():
            if not keys:
                continue
            kb = key_blocks.get(morph_name)
            if kb is None:
                kb = norm_index.get(_normalize_morph_name(morph_name))
            if kb is None or kb == key_blocks[0]:
                continue

            frames = [int(k.frame) for k in keys]
            binding = driver_map.get(kb.name)
            if binding is not None:
                arm_id = id(binding.armature)
                dedup_key = (arm_id, binding.bone_name, binding.axis_index, kb.name)
                if dedup_key in seen_driven_keys:
                    continue
                seen_driven_keys.add(dedup_key)
                arm_by_id[arm_id] = binding.armature
                bone_groups.setdefault((arm_id, binding.bone_name), []).append((kb, binding, keys, frames))
                stats["morphs_driven"] += 1
            else:
                direct_tracks.append((shape_keys, kb, keys, frames))
                stats["morphs_direct"] += 1

    for (arm_id, bone_name), group_entries in bone_groups.items():
        arm_obj = arm_by_id[arm_id]
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue

        bone_vmd_frames = sorted(
            {f for _, _, _, frames in group_entries for f in frames if frame_start <= f <= frame_end}
        )
        if not bone_vmd_frames:
            bone_vmd_frames = [frame_start]

        compat.ensure_action_and_slot(arm_obj)
        inserted_bl_frames: set[float] = set()

        for vmd_frame in bone_vmd_frames:
            by_axis: dict[int, list[tuple[DriverBinding, float]]] = {}
            for kb, binding, keys, frames in group_entries:
                lo = float(getattr(kb, "slider_min", 0.0))
                hi = float(getattr(kb, "slider_max", 1.0))
                weight = max(lo, min(hi, _sample_key_weight(keys, frames, vmd_frame)))
                coord = (weight - binding.c0) / binding.c1
                by_axis.setdefault(binding.axis_index, []).append((binding, coord))

            for axis_index, items in by_axis.items():
                pos_coords = [c for b, c in items if b.c1 > 0.0]
                neg_coords = [c for b, c in items if b.c1 < 0.0]
                if pos_coords and neg_coords:
                    combined = max(pos_coords) + min(neg_coords)
                elif pos_coords:
                    combined = max(pos_coords)
                else:
                    combined = min(neg_coords)
                pbone.location[axis_index] = _clamp_to_bone_limits(pbone, axis_index, combined)

            bl_frame = float(vmd_frame + frame_offset)
            pbone.keyframe_insert(data_path="location", frame=bl_frame, group=pbone.name)
            inserted_bl_frames.add(bl_frame)
            stats["morph_keyframes"] += 1

        escaped = bone_name.replace("\\", "\\\\").replace('"', '\\"')
        _set_linear_interpolation(arm_obj, {f'pose.bones["{escaped}"].location'}, inserted_bl_frames)
        stats["morph_bones"] += 1

    for shape_keys, kb, keys, frames in direct_tracks:
        track_frames = [f for f in frames if frame_start <= f <= frame_end]
        if not track_frames:
            track_frames = [frame_start]

        lo = float(getattr(kb, "slider_min", 0.0))
        hi = float(getattr(kb, "slider_max", 1.0))
        inserted_bl_frames = set()

        for vmd_frame in track_frames:
            weight = max(lo, min(hi, _sample_key_weight(keys, frames, vmd_frame)))
            kb.value = weight
            bl_frame = float(vmd_frame + frame_offset)
            compat.insert_shape_key_keyframe(kb, frame=bl_frame)
            inserted_bl_frames.add(bl_frame)
            stats["morph_keyframes"] += 1

        escaped = kb.name.replace("\\", "\\\\").replace('"', '\\"')
        _set_linear_interpolation(shape_keys, {f'key_blocks["{escaped}"].value'}, inserted_bl_frames)

    return stats
