# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct MikuMikuDance (.vpd) parser and pure in-memory virtual skeleton solver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any

from mathutils import Matrix, Quaternion, Vector

from . import bone_map
from .math_retarget import (
    BoneTransform,
    PoseSnapshot,
    RetargetOptions,
    retarget_pose_from_snapshot,
)

_BONE_HEADER_RE = re.compile(r"^Bone\d*\s*\{(.*)$")
_MORPH_HEADER_RE = re.compile(r"^Morph\d*\s*\{(.*)$")


@dataclass
class VpdBoneEntry:
    name: str
    canonical_name: str
    location_mmd: tuple[float, float, float]
    rotation_mmd_xyzw: tuple[float, float, float, float]
    location_bl: Vector
    rotation_bl: Quaternion


@dataclass
class VpdMorphEntry:
    name: str
    weight: float


@dataclass
class VpdPoseData:
    filepath: str
    osm_name: str
    declared_bone_count: int
    bones: dict[str, VpdBoneEntry] = field(default_factory=dict)
    raw_bones: list[VpdBoneEntry] = field(default_factory=list)
    morphs: dict[str, VpdMorphEntry] = field(default_factory=dict)


def mmd_to_blender_location(loc_xyz: Sequence[float], scale: float = 0.08) -> Vector:
    """Convert left-handed MMD (X, Y, Z) translation to right-handed Blender (X, Z, Y)."""
    x = float(loc_xyz[0])
    y = float(loc_xyz[1])
    z = float(loc_xyz[2])
    s = float(scale)
    return Vector((x * s, z * s, y * s))


def mmd_to_blender_quaternion(rot_xyzw: Sequence[float]) -> Quaternion:
    """Convert left-handed MMD (X, Y, Z, W) quaternion to right-handed Blender (W, -X, -Z, -Y)."""
    x = float(rot_xyzw[0])
    y = float(rot_xyzw[1])
    z = float(rot_xyzw[2])
    w = float(rot_xyzw[3])
    q = Quaternion((w, -x, -z, -y))
    if q.magnitude < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    return q.normalized()


def apply_vpd_bone_local_delta(
    rest_matrix_local: Matrix,
    vpd_entry: VpdBoneEntry,
    scale: float = 0.08,
) -> Matrix:
    """Convert a VPD bone entry into a local basis 4x4 matrix relative to rest_matrix_local."""
    r_local = rest_matrix_local.to_3x3().normalized()
    mat = r_local.copy()
    mat[1], mat[2] = mat[2].copy(), mat[1].copy()
    conv_mat = mat.transposed()

    loc_basis = (conv_mat @ Vector(vpd_entry.location_mmd)) * float(scale)

    qx, qy, qz, qw = vpd_entry.rotation_mmd_xyzw
    rot_mmd = Quaternion((qw, qx, qy, qz))
    if rot_mmd.magnitude < 1e-8:
        rot_basis = Quaternion((1.0, 0.0, 0.0, 0.0))
    else:
        rot_mmd.normalize()
        # Swapping Y and Z rows creates a reflection (det = -1), so the right-handed axis is negated.
        axis_bl = -(conv_mat @ rot_mmd.axis)
        rot_basis = Quaternion(axis_bl, rot_mmd.angle).normalized()

    return Matrix.Translation(loc_basis) @ rot_basis.to_matrix().to_4x4()


def _strip_vpd_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def parse_vpd_text(text: str, filepath: str = "", scale: float = 0.08) -> VpdPoseData:
    """Parse VPD text contents into structured VpdPoseData."""
    cleaned_text = text.lstrip("\ufeff")
    raw_lines = cleaned_text.splitlines()

    idx = 0
    total_lines = len(raw_lines)
    while idx < total_lines and not raw_lines[idx].strip():
        idx += 1

    if idx >= total_lines or not raw_lines[idx].strip().startswith("Vocaloid Pose Data file"):
        raise ValueError("Invalid VPD file: missing 'Vocaloid Pose Data file' header")
    idx += 1

    osm_name = ""
    declared_bone_count = 0
    header_fields_found = 0

    while idx < total_lines and header_fields_found < 2:
        stripped = _strip_vpd_comment(raw_lines[idx])
        if not stripped:
            idx += 1
            continue
        if _BONE_HEADER_RE.match(stripped) or _MORPH_HEADER_RE.match(stripped):
            break
        val = stripped.split(";", 1)[0].strip()
        if header_fields_found == 0:
            osm_name = val
            header_fields_found = 1
        else:
            try:
                declared_bone_count = int(val)
            except ValueError:
                declared_bone_count = 0
            header_fields_found = 2
        idx += 1

    bones: dict[str, VpdBoneEntry] = {}
    raw_bones: list[VpdBoneEntry] = []
    morphs: dict[str, VpdMorphEntry] = {}

    while idx < total_lines:
        line = _strip_vpd_comment(raw_lines[idx])
        idx += 1
        if not line:
            continue

        bone_match = _BONE_HEADER_RE.match(line)
        if bone_match:
            bone_name = bone_match.group(1).strip()
            block_lines: list[str] = []
            while idx < total_lines:
                inner = _strip_vpd_comment(raw_lines[idx])
                idx += 1
                if not inner:
                    continue
                if inner.startswith("}"):
                    break
                if "}" in inner:
                    before_brace = inner.split("}", 1)[0].strip()
                    if before_brace:
                        block_lines.append(before_brace)
                    break
                block_lines.append(inner)

            if len(block_lines) < 2:
                raise ValueError(f"Malformed Bone block for {bone_name!r}")

            loc_parts = [p.strip() for p in block_lines[0].split(";", 1)[0].split(",") if p.strip()]
            rot_parts = [p.strip() for p in block_lines[1].split(";", 1)[0].split(",") if p.strip()]
            if len(loc_parts) != 3 or len(rot_parts) != 4:
                raise ValueError(f"Invalid transform values in Bone block for {bone_name!r}")

            loc_mmd = (float(loc_parts[0]), float(loc_parts[1]), float(loc_parts[2]))
            rx, ry, rz, rw = (
                float(rot_parts[0]),
                float(rot_parts[1]),
                float(rot_parts[2]),
                float(rot_parts[3]),
            )
            if abs(rx) + abs(ry) + abs(rz) + abs(rw) < 1e-8:
                rot_mmd = (0.0, 0.0, 0.0, 1.0)
            else:
                rot_mmd = (rx, ry, rz, rw)

            canonical = bone_map.normalize_mmd_name(bone_name) or bone_name
            loc_bl = mmd_to_blender_location(loc_mmd, scale=scale)
            rot_bl = mmd_to_blender_quaternion(rot_mmd)
            entry = VpdBoneEntry(
                name=bone_name,
                canonical_name=canonical,
                location_mmd=loc_mmd,
                rotation_mmd_xyzw=rot_mmd,
                location_bl=loc_bl,
                rotation_bl=rot_bl,
            )
            raw_bones.append(entry)
            bones[canonical] = entry
            continue

        morph_match = _MORPH_HEADER_RE.match(line)
        if morph_match:
            morph_name = morph_match.group(1).strip()
            weight_val = 0.0
            while idx < total_lines:
                inner = _strip_vpd_comment(raw_lines[idx])
                idx += 1
                if not inner:
                    continue
                if inner.startswith("}"):
                    break
                if "}" in inner:
                    before_brace = inner.split("}", 1)[0].strip()
                    if before_brace:
                        weight_val = float(before_brace.split(";", 1)[0].strip())
                    break
                weight_val = float(inner.split(";", 1)[0].strip())

            morphs[morph_name] = VpdMorphEntry(name=morph_name, weight=weight_val)

    return VpdPoseData(
        filepath=str(filepath),
        osm_name=osm_name,
        declared_bone_count=declared_bone_count,
        bones=bones,
        raw_bones=raw_bones,
        morphs=morphs,
    )


def parse_vpd_file(filepath: str, scale: float = 0.08) -> VpdPoseData:
    """Read and parse a .vpd file from disk supporting Shift-JIS/CP932 and UTF-8 encodings."""
    path = Path(filepath)
    raw_bytes = path.read_bytes()

    decoded_text: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            decoded_text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        decoded_text = raw_bytes.decode("cp932", errors="replace")

    return parse_vpd_text(decoded_text, filepath=str(path), scale=scale)


def _lookup_rig_bone_rest_local(
    target_rig: Any,
    candidates: tuple[str, ...],
    fallback_pos: Vector,
) -> Matrix:
    data_bones = getattr(getattr(target_rig, "data", None), "bones", None)
    pose_bones = getattr(getattr(target_rig, "pose", None), "bones", None)

    for cand in candidates:
        if data_bones is not None and cand in data_bones:
            return data_bones[cand].matrix_local.copy()
        if pose_bones is not None and cand in pose_bones:
            return pose_bones[cand].bone.matrix_local.copy()

    return Matrix.Translation(fallback_pos)


def _find_vpd_entry(vpd_data: VpdPoseData, bone_name: str) -> VpdBoneEntry | None:
    norm = bone_map.normalize_mmd_name(bone_name)
    if norm in vpd_data.bones:
        return vpd_data.bones[norm]
    for variant in bone_map.mmd_name_variants(bone_name):
        v_norm = bone_map.normalize_mmd_name(variant)
        if v_norm in vpd_data.bones:
            return vpd_data.bones[v_norm]
        if variant in vpd_data.bones:
            return vpd_data.bones[variant]
    return None


def _solve_two_bone_leg_ik_in_place(
    rest_local_map: dict[str, Matrix],
    pose_local_map: dict[str, Matrix],
    vpd_data: VpdPoseData,
    side: str,
) -> None:
    thigh_name = f"足.{side}"
    knee_name = f"ひざ.{side}"
    ankle_name = f"足首.{side}"
    toe_name = f"つま先.{side}"
    ik_name = f"足ＩＫ.{side}"

    if (
        thigh_name not in pose_local_map
        or knee_name not in pose_local_map
        or ankle_name not in pose_local_map
        or ik_name not in pose_local_map
    ):
        return

    knee_vpd = _find_vpd_entry(vpd_data, knee_name)
    ik_vpd = _find_vpd_entry(vpd_data, ik_name)
    groove_vpd = _find_vpd_entry(vpd_data, "グルーブ")
    center_vpd = _find_vpd_entry(vpd_data, "センター")
    ik_has_direct_target = ik_vpd is not None and (
        ik_vpd.location_bl.length > 1e-5 or ik_vpd.rotation_bl.angle > 1e-4
    )
    if not ik_has_direct_target and knee_vpd is not None and knee_vpd.rotation_bl.angle > 0.15:
        # Respect explicit FK knee rotations only when no IK target drives the foot.
        return

    has_ik_offset = (
        ik_has_direct_target
        or (groove_vpd is not None and groove_vpd.location_bl.length > 1e-5)
        or (center_vpd is not None and center_vpd.location_bl.length > 1e-5)
    )
    if not has_ik_offset:
        return

    w_thigh_0 = pose_local_map[thigh_name]
    w_knee_0 = pose_local_map[knee_name]
    w_ankle_0 = pose_local_map[ankle_name]
    w_ik = pose_local_map[ik_name]

    a_pos = w_thigh_0.translation.copy()
    b0_pos = w_knee_0.translation.copy()
    c0_pos = w_ankle_0.translation.copy()
    t_pos = w_ik.translation.copy()

    l1 = (b0_pos - a_pos).length
    l2 = (c0_pos - b0_pos).length
    if l1 < 1e-5 or l2 < 1e-5:
        return

    u_vec = t_pos - a_pos
    dist = u_vec.length
    if dist < 1e-5:
        return
    u_dir = u_vec / dist

    min_d = abs(l1 - l2) + 1e-4
    max_d = (l1 + l2) - 1e-4
    d_clamped = max(min_d, min(max_d, dist))

    cos_alpha = (l1 * l1 + d_clamped * d_clamped - l2 * l2) / (2.0 * l1 * d_clamped)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

    lower_pose = pose_local_map.get("下半身", Matrix.Identity(4))
    char_forward = lower_pose.to_3x3().normalized() @ Vector((0.0, -1.0, 0.0))

    pre_bend = (b0_pos - a_pos) - u_dir * (b0_pos - a_pos).dot(u_dir)
    if pre_bend.length > 1e-3 and pre_bend.dot(char_forward) >= -1e-3:
        bend_dir = pre_bend.normalized()
    else:
        ortho_fwd = char_forward - u_dir * char_forward.dot(u_dir)
        if ortho_fwd.length < 1e-5:
            fallback = Vector((1.0, 0.0, 0.0)) if abs(u_dir.x) < 0.9 else Vector((0.0, 0.0, 1.0))
            ortho_fwd = fallback - u_dir * fallback.dot(u_dir)
        bend_dir = ortho_fwd.normalized()

    b_ik = a_pos + u_dir * (l1 * cos_alpha) + bend_dir * (l1 * sin_alpha)
    c_ik = a_pos + u_dir * d_clamped

    v_thigh_0 = (b0_pos - a_pos).normalized()
    v_thigh_1 = (b_ik - a_pos).normalized()
    q_thigh = v_thigh_0.rotation_difference(v_thigh_1)
    r_thigh_0 = w_thigh_0.to_3x3().normalized()
    r_thigh_1 = (q_thigh.to_matrix() @ r_thigh_0).normalized()
    pose_local_map[thigh_name] = Matrix.Translation(a_pos) @ r_thigh_1.to_4x4()

    v_shin_0 = (c0_pos - b0_pos).normalized()
    v_shin_carried = (q_thigh.to_matrix() @ v_shin_0).normalized()
    v_shin_1 = (c_ik - b_ik).normalized()
    q_shin = v_shin_carried.rotation_difference(v_shin_1)
    r_shin_0 = w_knee_0.to_3x3().normalized()
    r_shin_1 = (q_shin.to_matrix() @ q_thigh.to_matrix() @ r_shin_0).normalized()
    pose_local_map[knee_name] = Matrix.Translation(b_ik) @ r_shin_1.to_4x4()

    r_ankle_1 = w_ik.to_3x3().normalized()
    w_ankle_1 = Matrix.Translation(c_ik) @ r_ankle_1.to_4x4()
    pose_local_map[ankle_name] = w_ankle_1

    if toe_name in pose_local_map and toe_name in rest_local_map:
        rel_toe = w_ankle_0.inverted_safe() @ pose_local_map[toe_name]
        pose_local_map[toe_name] = w_ankle_1 @ rel_toe


def build_virtual_snapshot_from_vpd(
    vpd_data: VpdPoseData,
    target_rig: Any,
    scale: float = 0.08,
) -> PoseSnapshot:
    """Construct a world-space PoseSnapshot from VPD data in memory without creating Blender objects."""
    arm_world = (getattr(target_rig, "matrix_world", None) or Matrix.Identity(4)).copy()

    has_upper_3 = _find_vpd_entry(vpd_data, "上半身3") is not None
    has_upper_1 = _find_vpd_entry(vpd_data, "上半身1") is not None
    if has_upper_3:
        upper2_parent = "上半身3"
    elif has_upper_1:
        upper2_parent = "上半身1"
    else:
        upper2_parent = "上半身"

    virtual_hierarchy: list[tuple[str, str | None, tuple[str, ...], Vector]] = [
        ("全ての親", None, ("root", "torso_root"), Vector((0.0, 0.0, 0.0))),
        ("センター", "全ての親", ("torso", "hips", "ORG-spine"), Vector((0.0, 0.0, 0.8))),
        ("グルーブ", "センター", ("torso", "hips", "ORG-spine"), Vector((0.0, 0.0, 0.8))),
        ("腰", "グルーブ", ("hips", "torso", "ORG-spine"), Vector((0.0, 0.0, 0.95))),
        ("下半身", "腰", ("hips", "spine_fk", "ORG-spine", "torso"), Vector((0.0, 0.0, 1.0))),
        ("上半身", "腰", ("spine_fk.001", "spine_fk", "ORG-spine.001"), Vector((0.0, 0.0, 1.0))),
        ("上半身3", "上半身", ("spine_fk.002", "ORG-spine.002"), Vector((0.0, 0.0, 1.15))),
        ("上半身1", "上半身", ("spine_fk.002", "ORG-spine.002"), Vector((0.0, 0.0, 1.15))),
        (
            "上半身2",
            upper2_parent,
            ("spine_fk.003", "chest", "ORG-spine.003"),
            Vector((0.0, 0.0, 1.25)),
        ),
        ("首", "上半身2", ("neck", "ORG-spine.004"), Vector((0.0, 0.0, 1.4))),
        ("頭", "首", ("head", "ORG-spine.006"), Vector((0.0, 0.0, 1.5))),
        ("両目", "頭", ("eyes", "head"), Vector((0.0, -0.05, 1.6))),
        ("目.L", "両目", ("eye.L", "ORG-eye.L"), Vector((0.03, -0.05, 1.6))),
        ("目.R", "両目", ("eye.R", "ORG-eye.R"), Vector((-0.03, -0.05, 1.6))),
    ]

    for side, s in (("L", 1.0), ("R", -1.0)):
        virtual_hierarchy.extend(
            [
                (
                    f"肩P.{side}",
                    "上半身2",
                    (f"shoulder.{side}", f"ORG-shoulder.{side}"),
                    Vector((s * 0.03, 0.0, 1.38)),
                ),
                (
                    f"肩.{side}",
                    f"肩P.{side}",
                    (f"shoulder.{side}", f"ORG-shoulder.{side}"),
                    Vector((s * 0.03, 0.0, 1.38)),
                ),
                (
                    f"肩C.{side}",
                    f"肩.{side}",
                    (f"upper_arm_fk.{side}", f"ORG-upper_arm.{side}"),
                    Vector((s * 0.15, 0.0, 1.40)),
                ),
                (
                    f"腕.{side}",
                    f"肩C.{side}",
                    (f"upper_arm_fk.{side}", f"ORG-upper_arm.{side}", f"DEF-upper_arm.{side}"),
                    Vector((s * 0.15, 0.0, 1.40)),
                ),
                (
                    f"腕捩.{side}",
                    f"腕.{side}",
                    (f"upper_arm_fk.{side}", f"ORG-upper_arm.{side}"),
                    Vector((s * 0.25, 0.0, 1.30)),
                ),
                (
                    f"ひじ.{side}",
                    f"腕捩.{side}",
                    (f"forearm_fk.{side}", f"ORG-forearm.{side}", f"DEF-forearm.{side}"),
                    Vector((s * 0.35, 0.0, 1.20)),
                ),
                (
                    f"手捩.{side}",
                    f"ひじ.{side}",
                    (f"forearm_fk.{side}", f"ORG-forearm.{side}"),
                    Vector((s * 0.45, 0.0, 1.10)),
                ),
                (
                    f"手首.{side}",
                    f"手捩.{side}",
                    (f"hand_fk.{side}", f"hand_ik.{side}", f"ORG-hand.{side}"),
                    Vector((s * 0.55, 0.0, 1.00)),
                ),
            ]
        )

        has_thumb0 = _find_vpd_entry(vpd_data, f"親指０.{side}") is not None or (
            _find_vpd_entry(vpd_data, f"親指先.{side}") is None
        )
        thumb_names = (
            (f"親指０.{side}", f"親指１.{side}", f"親指２.{side}")
            if has_thumb0
            else (f"親指１.{side}", f"親指２.{side}", f"親指先.{side}")
        )
        parent_name = f"手首.{side}"
        for seg_idx, mmd_thumb in enumerate(thumb_names, start=1):
            virtual_hierarchy.append(
                (
                    mmd_thumb,
                    parent_name,
                    (f"thumb.{seg_idx:02d}.{side}", f"ORG-thumb.{seg_idx:02d}.{side}"),
                    Vector((s * (0.58 + 0.02 * seg_idx), -0.03, 0.95)),
                )
            )
            parent_name = mmd_thumb

        for finger_idx, (en_finger, jp_finger) in enumerate(
            (
                ("f_index", "人指"),
                ("f_middle", "中指"),
                ("f_ring", "薬指"),
                ("f_pinky", "小指"),
            ),
            start=1,
        ):
            parent_name = f"手首.{side}"
            for seg_idx, fw_num in enumerate(("１", "２", "３"), start=1):
                mmd_f_name = f"{jp_finger}{fw_num}.{side}"
                virtual_hierarchy.append(
                    (
                        mmd_f_name,
                        parent_name,
                        (
                            f"{en_finger}.{seg_idx:02d}.{side}",
                            f"ORG-{en_finger}.{seg_idx:02d}.{side}",
                        ),
                        Vector((s * (0.58 + 0.02 * seg_idx), -0.03 + 0.015 * finger_idx, 0.95)),
                    )
                )
                parent_name = mmd_f_name

        virtual_hierarchy.extend(
            [
                (
                    f"腰キャンセル.{side}",
                    "下半身",
                    (f"thigh_fk.{side}", f"ORG-thigh.{side}"),
                    Vector((s * 0.10, 0.0, 0.95)),
                ),
                (
                    f"足.{side}",
                    f"腰キャンセル.{side}",
                    (f"thigh_fk.{side}", f"ORG-thigh.{side}", f"DEF-thigh.{side}"),
                    Vector((s * 0.10, 0.0, 0.95)),
                ),
                (
                    f"ひざ.{side}",
                    f"足.{side}",
                    (f"shin_fk.{side}", f"ORG-shin.{side}", f"DEF-shin.{side}"),
                    Vector((s * 0.10, 0.0, 0.50)),
                ),
                (
                    f"足首.{side}",
                    f"ひざ.{side}",
                    (f"foot_fk.{side}", f"foot_ik.{side}", f"ORG-foot.{side}"),
                    Vector((s * 0.10, 0.0, 0.10)),
                ),
                (
                    f"つま先.{side}",
                    f"足首.{side}",
                    (f"toe.{side}", f"ORG-toe.{side}"),
                    Vector((s * 0.10, -0.10, 0.02)),
                ),
                (
                    f"足IK親.{side}",
                    "全ての親",
                    (f"foot_ik.{side}", f"foot_fk.{side}", f"ORG-foot.{side}"),
                    Vector((s * 0.10, 0.0, 0.10)),
                ),
                (
                    f"足ＩＫ.{side}",
                    f"足IK親.{side}",
                    (f"foot_ik.{side}", f"foot_fk.{side}", f"ORG-foot.{side}"),
                    Vector((s * 0.10, 0.0, 0.10)),
                ),
                (
                    f"つま先ＩＫ.{side}",
                    f"足ＩＫ.{side}",
                    (f"toe.{side}", f"ORG-toe.{side}"),
                    Vector((s * 0.10, -0.10, 0.02)),
                ),
            ]
        )

    rest_local_map: dict[str, Matrix] = {}
    pose_local_map: dict[str, Matrix] = {}

    for mmd_name, parent_name, rig_cands, fallback_pos in virtual_hierarchy:
        rest_mat = _lookup_rig_bone_rest_local(target_rig, rig_cands, fallback_pos)
        if mmd_name.startswith("腕捩."):
            side = mmd_name[-1]
            ua_mat = rest_local_map.get(f"腕.{side}", rest_mat)
            fa_mat = _lookup_rig_bone_rest_local(
                target_rig,
                (f"forearm_fk.{side}", f"ORG-forearm.{side}"),
                fallback_pos,
            )
            mid_pos = (ua_mat.translation + fa_mat.translation) * 0.5
            rest_mat = Matrix.Translation(mid_pos) @ ua_mat.to_3x3().normalized().to_4x4()
        elif mmd_name.startswith("手捩."):
            side = mmd_name[-1]
            fa_mat = rest_local_map.get(f"ひじ.{side}", rest_mat)
            ha_mat = _lookup_rig_bone_rest_local(
                target_rig,
                (f"hand_fk.{side}", f"hand_ik.{side}", f"ORG-hand.{side}"),
                fallback_pos,
            )
            mid_pos = (fa_mat.translation + ha_mat.translation) * 0.5
            rest_mat = Matrix.Translation(mid_pos) @ fa_mat.to_3x3().normalized().to_4x4()

        rest_local_map[mmd_name] = rest_mat

        vpd_entry = _find_vpd_entry(vpd_data, mmd_name)
        if vpd_entry is not None:
            basis_mat = apply_vpd_bone_local_delta(rest_mat, vpd_entry, scale=scale)
        else:
            basis_mat = Matrix.Identity(4)

        if parent_name is None or parent_name not in rest_local_map:
            pose_mat = rest_mat @ basis_mat
        else:
            parent_rest = rest_local_map[parent_name]
            parent_pose = pose_local_map[parent_name]
            rest_rel = parent_rest.inverted_safe() @ rest_mat
            pose_mat = parent_pose @ rest_rel @ basis_mat

        pose_local_map[mmd_name] = pose_mat

    for side in ("L", "R"):
        _solve_two_bone_leg_ik_in_place(rest_local_map, pose_local_map, vpd_data, side)

    snapshot_bones: dict[str, BoneTransform] = {}
    for mmd_name, rest_local in rest_local_map.items():
        pose_local = pose_local_map[mmd_name]
        tf = BoneTransform(
            rest_world=arm_world @ rest_local,
            pose_world=arm_world @ pose_local,
        )
        snapshot_bones[mmd_name] = tf
        norm = bone_map.normalize_mmd_name(mmd_name)
        if norm:
            snapshot_bones.setdefault(norm, tf)

    morphs = {name: float(entry.weight) for name, entry in vpd_data.morphs.items()}
    return PoseSnapshot(armature_world=arm_world, bones=snapshot_bones, morphs=morphs)


def apply_vpd_to_mmr(
    context: Any,
    filepath: str,
    target_rig: Any,
    options: RetargetOptions | None = None,
    scale: float = 0.08,
) -> dict[str, int]:
    """Parse a .vpd file, solve its virtual MMD skeleton in memory, and retarget to an MMR rig."""
    opts = options or RetargetOptions()
    vpd_data = parse_vpd_file(filepath, scale=scale)
    snapshot = build_virtual_snapshot_from_vpd(vpd_data, target_rig, scale=scale)
    return retarget_pose_from_snapshot(context, snapshot, target_rig, opts)
