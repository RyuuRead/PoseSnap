# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct MikuMikuDance (.vmd) binary motion parser and MMR timeline baker."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path
import struct
from typing import Any

from mathutils import Quaternion

from . import bone_map, compat, vpd_parser
from .math_retarget import RetargetOptions, retarget_pose_from_snapshot

_MAGIC_V2 = b"Vocaloid Motion Data 0002"
_MAGIC_V1 = b"Vocaloid Motion Data file"
_BONE_STRUCT = struct.Struct("<15sI3f4f64B")
_MORPH_STRUCT = struct.Struct("<15sIf")
_UINT32 = struct.Struct("<I")


@dataclass(frozen=True)
class VmdBoneKey:
    frame: int
    location_mmd: tuple[float, float, float]
    rotation_mmd_xyzw: tuple[float, float, float, float]
    # Bezier control points (x1, y1, x2, y2) normalized to [0, 1] for X, Y, Z, and Rotation.
    interp_x: tuple[float, float, float, float]
    interp_y: tuple[float, float, float, float]
    interp_z: tuple[float, float, float, float]
    interp_r: tuple[float, float, float, float]


@dataclass(frozen=True)
class VmdMorphKey:
    frame: int
    weight: float


@dataclass
class VmdMotionData:
    filepath: str
    version: int
    model_name: str
    bones: dict[str, list[VmdBoneKey]] = field(default_factory=dict)
    morphs: dict[str, list[VmdMorphKey]] = field(default_factory=dict)
    bone_key_count: int = 0
    morph_key_count: int = 0
    frame_min: int = 0
    frame_max: int = 0


def _decode_cp932(raw: bytes) -> str:
    trimmed = raw.split(b"\x00", 1)[0]
    try:
        return trimmed.decode("cp932").strip()
    except UnicodeDecodeError:
        return trimmed.decode("cp932", errors="replace").strip()


def _extract_bezier(interp_bytes: tuple[int, ...], channel: int) -> tuple[float, float, float, float]:
    inv = 1.0 / 127.0
    return (
        interp_bytes[channel] * inv,
        interp_bytes[channel + 4] * inv,
        interp_bytes[channel + 8] * inv,
        interp_bytes[channel + 12] * inv,
    )


def eval_vmd_bezier(t: float, ctrl: tuple[float, float, float, float]) -> float:
    """Evaluate a VMD cubic Bezier easing curve at normalized time t in [0, 1]."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    x1, y1, x2, y2 = ctrl
    if abs(x1 - y1) < 1e-5 and abs(x2 - y2) < 1e-5:
        return t

    lo, hi = 0.0, 1.0
    u = t
    for _ in range(14):
        u = 0.5 * (lo + hi)
        inv = 1.0 - u
        bx = 3.0 * inv * inv * u * x1 + 3.0 * inv * u * u * x2 + u * u * u
        if bx < t:
            lo = u
        else:
            hi = u

    inv = 1.0 - u
    return 3.0 * inv * inv * u * y1 + 3.0 * inv * u * u * y2 + u * u * u


def parse_vmd_bytes(data: bytes, filepath: str = "") -> VmdMotionData:
    """Parse binary VMD bytes into sorted bone and morph keyframe tracks."""
    if len(data) < 40:
        raise ValueError("VMD data is truncated: header is shorter than 40 bytes.")

    magic_raw = data[:30]
    if magic_raw.startswith(_MAGIC_V2):
        version = 2
        name_len = 20
    elif magic_raw.startswith(_MAGIC_V1):
        version = 1
        name_len = 10
    else:
        raise ValueError("Invalid VMD header: expected Vocaloid Motion Data magic.")

    header_end = 30 + name_len
    if len(data) < header_end + 4:
        raise ValueError("VMD data is truncated before bone keyframe count.")

    model_name = _decode_cp932(data[30:header_end])
    offset = header_end

    (bone_count,) = _UINT32.unpack_from(data, offset)
    offset += 4
    bone_bytes_needed = bone_count * _BONE_STRUCT.size
    if offset + bone_bytes_needed > len(data):
        raise ValueError(
            f"VMD bone section truncated: declared {bone_count} keys, "
            f"only {len(data) - offset} bytes remain."
        )

    bone_tracks_raw: dict[str, dict[int, VmdBoneKey]] = {}
    frame_min: int | None = None
    frame_max: int | None = None

    for _ in range(bone_count):
        unpacked = _BONE_STRUCT.unpack_from(data, offset)
        offset += _BONE_STRUCT.size
        bname = _decode_cp932(unpacked[0])
        frame = int(unpacked[1])
        loc = (float(unpacked[2]), float(unpacked[3]), float(unpacked[4]))
        rot = (float(unpacked[5]), float(unpacked[6]), float(unpacked[7]), float(unpacked[8]))
        interp = unpacked[9:]
        if not bname:
            continue

        key = VmdBoneKey(
            frame=frame,
            location_mmd=loc,
            rotation_mmd_xyzw=rot,
            interp_x=_extract_bezier(interp, 0),
            interp_y=_extract_bezier(interp, 1),
            interp_z=_extract_bezier(interp, 2),
            interp_r=_extract_bezier(interp, 3),
        )
        bone_tracks_raw.setdefault(bname, {})[frame] = key
        if frame_min is None or frame < frame_min:
            frame_min = frame
        if frame_max is None or frame > frame_max:
            frame_max = frame

    morph_tracks_raw: dict[str, dict[int, VmdMorphKey]] = {}
    morph_count = 0
    if offset + 4 <= len(data):
        (morph_count,) = _UINT32.unpack_from(data, offset)
        offset += 4
        morph_bytes_needed = morph_count * _MORPH_STRUCT.size
        if offset + morph_bytes_needed > len(data):
            raise ValueError(
                f"VMD morph section truncated: declared {morph_count} keys, "
                f"only {len(data) - offset} bytes remain."
            )
        for _ in range(morph_count):
            raw_name, frame_u, weight = _MORPH_STRUCT.unpack_from(data, offset)
            offset += _MORPH_STRUCT.size
            mname = _decode_cp932(raw_name)
            frame = int(frame_u)
            if not mname:
                continue
            morph_tracks_raw.setdefault(mname, {})[frame] = VmdMorphKey(
                frame=frame,
                weight=float(weight),
            )
            if frame_min is None or frame < frame_min:
                frame_min = frame
            if frame_max is None or frame > frame_max:
                frame_max = frame

    bones = {
        name: [frame_map[f] for f in sorted(frame_map)]
        for name, frame_map in bone_tracks_raw.items()
    }
    morphs = {
        name: [frame_map[f] for f in sorted(frame_map)]
        for name, frame_map in morph_tracks_raw.items()
    }

    return VmdMotionData(
        filepath=filepath,
        version=version,
        model_name=model_name,
        bones=bones,
        morphs=morphs,
        bone_key_count=bone_count,
        morph_key_count=morph_count,
        frame_min=frame_min if frame_min is not None else 0,
        frame_max=frame_max if frame_max is not None else 0,
    )


def parse_vmd_file(filepath: str) -> VmdMotionData:
    """Read and parse a .vmd file from disk."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"VMD file not found: {filepath}")
    return parse_vmd_bytes(path.read_bytes(), filepath=str(path))


def _sample_bone_track(
    keys: list[VmdBoneKey],
    frames: list[int],
    frame: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if frame <= frames[0]:
        k = keys[0]
        return k.location_mmd, k.rotation_mmd_xyzw
    if frame >= frames[-1]:
        k = keys[-1]
        return k.location_mmd, k.rotation_mmd_xyzw

    idx = bisect_right(frames, frame)
    k0 = keys[idx - 1]
    k1 = keys[idx]
    if k0.frame == frame:
        return k0.location_mmd, k0.rotation_mmd_xyzw

    span = float(k1.frame - k0.frame)
    t = (frame - k0.frame) / span
    fx = eval_vmd_bezier(t, k1.interp_x)
    fy = eval_vmd_bezier(t, k1.interp_y)
    fz = eval_vmd_bezier(t, k1.interp_z)
    fr = eval_vmd_bezier(t, k1.interp_r)

    lx = k0.location_mmd[0] + (k1.location_mmd[0] - k0.location_mmd[0]) * fx
    ly = k0.location_mmd[1] + (k1.location_mmd[1] - k0.location_mmd[1]) * fy
    lz = k0.location_mmd[2] + (k1.location_mmd[2] - k0.location_mmd[2]) * fz

    x0, y0, z0, w0 = k0.rotation_mmd_xyzw
    x1, y1, z1, w1 = k1.rotation_mmd_xyzw
    q0 = Quaternion((w0, x0, y0, z0))
    q1 = Quaternion((w1, x1, y1, z1))
    q0 = q0.normalized() if q0.magnitude >= 1e-8 else Quaternion((1.0, 0.0, 0.0, 0.0))
    q1 = q1.normalized() if q1.magnitude >= 1e-8 else Quaternion((1.0, 0.0, 0.0, 0.0))
    if q0.dot(q1) < 0.0:
        q1 = -q1
    q = q0.slerp(q1, fr).normalized()
    return (lx, ly, lz), (q.x, q.y, q.z, q.w)


def _sample_morph_track(keys: list[VmdMorphKey], frames: list[int], frame: int) -> float:
    if frame <= frames[0]:
        return keys[0].weight
    if frame >= frames[-1]:
        return keys[-1].weight

    idx = bisect_right(frames, frame)
    k0 = keys[idx - 1]
    k1 = keys[idx]
    if k0.frame == frame:
        return k0.weight
    t = (frame - k0.frame) / float(k1.frame - k0.frame)
    return k0.weight + (k1.weight - k0.weight) * t


def sample_vmd_frame_as_vpd(
    vmd_data: VmdMotionData,
    frame: int,
    scale: float = 0.08,
    bone_frame_index: dict[str, list[int]] | None = None,
    morph_frame_index: dict[str, list[int]] | None = None,
) -> vpd_parser.VpdPoseData:
    """Evaluate a VMD motion at one integer frame and package it as VpdPoseData."""
    bones_dict: dict[str, vpd_parser.VpdBoneEntry] = {}
    raw_bones: list[vpd_parser.VpdBoneEntry] = []

    for bname, keys in vmd_data.bones.items():
        if not keys:
            continue
        frames = bone_frame_index[bname] if bone_frame_index is not None else [k.frame for k in keys]
        loc_mmd, rot_xyzw = _sample_bone_track(keys, frames, frame)
        canonical = bone_map.normalize_mmd_name(bname)
        entry = vpd_parser.VpdBoneEntry(
            name=bname,
            canonical_name=canonical,
            location_mmd=loc_mmd,
            rotation_mmd_xyzw=rot_xyzw,
            location_bl=vpd_parser.mmd_to_blender_location(loc_mmd, scale=scale),
            rotation_bl=vpd_parser.mmd_to_blender_quaternion(rot_xyzw),
        )
        raw_bones.append(entry)
        bones_dict[bname] = entry
        if canonical:
            bones_dict.setdefault(canonical, entry)

    morphs_dict: dict[str, vpd_parser.VpdMorphEntry] = {}
    for mname, keys in vmd_data.morphs.items():
        if not keys:
            continue
        frames = morph_frame_index[mname] if morph_frame_index is not None else [k.frame for k in keys]
        weight = _sample_morph_track(keys, frames, frame)
        morphs_dict[mname] = vpd_parser.VpdMorphEntry(name=mname, weight=weight)

    return vpd_parser.VpdPoseData(
        filepath=vmd_data.filepath,
        osm_name=vmd_data.model_name,
        declared_bone_count=len(raw_bones),
        bones=bones_dict,
        raw_bones=raw_bones,
        morphs=morphs_dict,
    )


def apply_vmd_to_mmr(
    context: Any,
    filepath: str,
    target_rig: Any,
    options: RetargetOptions | None = None,
    scale: float = 0.08,
    frame_offset: int = 1,
    frame_step: int = 1,
    frame_start: int | None = None,
    frame_end: int | None = None,
    set_timeline_range: bool = False,
    bake_all_frames: bool = False,
) -> dict[str, int]:
    """Parse a .vmd motion file and import bone and facial keyframes onto Target MMR."""
    vmd_data = parse_vmd_file(filepath)
    if not vmd_data.bones and not vmd_data.morphs:
        raise ValueError("VMD file contains no bone or morph keyframes.")

    opts = options or RetargetOptions()
    start = vmd_data.frame_min if frame_start is None else max(vmd_data.frame_min, int(frame_start))
    end = vmd_data.frame_max if frame_end is None else min(vmd_data.frame_max, int(frame_end))
    if end < start:
        start, end = end, start
    step = max(1, int(frame_step))
    offset = int(frame_offset)

    compat.ensure_action_and_slot(target_rig)
    scene = getattr(context, "scene", None) if context is not None else None
    original_frame = getattr(scene, "frame_current", start + offset) if scene is not None else start + offset

    bone_frame_index = {name: [k.frame for k in keys] for name, keys in vmd_data.bones.items()}
    morph_frame_index = {name: [k.frame for k in keys] for name, keys in vmd_data.morphs.items()}

    frames_baked = 0
    last_stats: dict[str, int] = {
        "torso": 0,
        "head": 0,
        "arms": 0,
        "legs": 0,
        "fingers": 0,
        "total": 0,
        "morphs_driven": 0,
        "morphs_direct": 0,
        "morph_bones": 0,
    }

    if not bake_all_frames:
        if vmd_data.bones:
            bone_frames = sorted(
                {f for frames in bone_frame_index.values() for f in frames if start <= f <= end}
            )
            bone_only_vmd = replace(vmd_data, morphs={})
            for vmd_frame in bone_frames:
                bl_frame = vmd_frame + offset
                pose_data = sample_vmd_frame_as_vpd(
                    bone_only_vmd,
                    vmd_frame,
                    scale=scale,
                    bone_frame_index=bone_frame_index,
                    morph_frame_index={},
                )
                snapshot = vpd_parser.build_virtual_snapshot_from_vpd(pose_data, target_rig, scale=scale)
                frame_opts = replace(opts, use_morphs=False, insert_keyframes=True, frame=bl_frame)
                last_stats = retarget_pose_from_snapshot(context, snapshot, target_rig, frame_opts)
                frames_baked += 1

        if opts.use_morphs and vmd_data.morphs:
            from . import morph_retarget

            morph_stats = morph_retarget.apply_vmd_morph_tracks_sparse(
                context,
                target_rig,
                vmd_data.morphs,
                frame_start=start,
                frame_end=end,
                frame_offset=offset,
            )
            last_stats.update(morph_stats)
            if frames_baked == 0:
                morph_frames = {
                    f for frames in morph_frame_index.values() for f in frames if start <= f <= end
                }
                frames_baked = len(morph_frames) or 1
    else:
        sample_frames = list(range(start, end + 1, step))
        if sample_frames and sample_frames[-1] != end:
            sample_frames.append(end)

        if not vmd_data.bones:
            from . import morph_retarget

            target_meshes = morph_retarget.find_armature_meshes(target_rig, context)
            for vmd_frame in sample_frames:
                bl_frame = vmd_frame + offset
                if opts.use_morphs:
                    morph_weights = {
                        mname: _sample_morph_track(keys, morph_frame_index[mname], vmd_frame)
                        for mname, keys in vmd_data.morphs.items()
                        if keys
                    }
                    morph_stats = morph_retarget.apply_morphs_to_target(
                        context,
                        target_rig,
                        morph_weights,
                        insert_keyframes=True,
                        frame=bl_frame,
                        meshes=target_meshes,
                    )
                    last_stats.update(morph_stats)
                frames_baked += 1
        else:
            for vmd_frame in sample_frames:
                bl_frame = vmd_frame + offset
                pose_data = sample_vmd_frame_as_vpd(
                    vmd_data,
                    vmd_frame,
                    scale=scale,
                    bone_frame_index=bone_frame_index,
                    morph_frame_index=morph_frame_index,
                )
                snapshot = vpd_parser.build_virtual_snapshot_from_vpd(pose_data, target_rig, scale=scale)
                frame_opts = replace(opts, insert_keyframes=True, frame=bl_frame)
                last_stats = retarget_pose_from_snapshot(context, snapshot, target_rig, frame_opts)
                frames_baked += 1

    if set_timeline_range and scene is not None:
        scene.frame_start = start + offset
        scene.frame_end = end + offset
    if scene is not None:
        scene.frame_set(int(original_frame))
    if context is not None and hasattr(context, "view_layer"):
        context.view_layer.update()

    return {
        **last_stats,
        "frames_baked": frames_baked,
        "bones_posed": last_stats.get("total", 0),
        "frame_start": start + offset,
        "frame_end": end + offset,
    }
