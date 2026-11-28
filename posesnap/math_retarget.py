# SPDX-License-Identifier: GPL-3.0-or-later
"""World-space pose retargeting math, IK pole solver, and animation range baker."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

from mathutils import Matrix, Quaternion, Vector

from . import bone_map, compat


@dataclass
class RetargetOptions:
    arms_mode: str = "IK"
    legs_mode: str = "IK"
    use_torso: bool = True
    use_head: bool = True
    use_arms: bool = True
    use_fingers: bool = True
    use_legs: bool = True
    use_morphs: bool = True
    hip_scale_ratio: float = 1.0
    insert_keyframes: bool = False
    frame: int | None = None


@dataclass
class BoneTransform:
    rest_world: Matrix
    pose_world: Matrix


@dataclass
class PoseSnapshot:
    armature_world: Matrix
    bones: dict[str, BoneTransform] = field(default_factory=dict)
    morphs: dict[str, float] = field(default_factory=dict)


def compute_rest_offset(src_rest_world: Matrix, tgt_rest_world: Matrix) -> Matrix:
    """Compute the 4x4 transformation offset from source rest frame to target rest frame."""
    return src_rest_world.inverted_safe() @ tgt_rest_world


def compute_rest_rotation_offset(src_rest_world: Matrix, tgt_rest_world: Matrix) -> Matrix:
    """Compute the normalized 3x3 rotation offset between source and target rest frames."""
    src_rot = src_rest_world.to_3x3().normalized()
    tgt_rot = tgt_rest_world.to_3x3().normalized()
    return (src_rot.inverted_safe() @ tgt_rot).normalized()


def merge_twist_world_rotation(
    base_rest: Matrix,
    base_pose: Matrix,
    twist_rest: Matrix | None,
    twist_pose: Matrix | None,
    twist_is_child: bool = True,
) -> Matrix:
    """Fold an MMD twist bone rotation into its parent limb bone without 180-degree flips."""
    if twist_rest is None or twist_pose is None:
        return base_pose.copy()

    r_base_rest = base_rest.to_3x3().normalized()
    r_base_pose = base_pose.to_3x3().normalized()
    r_twist_rest = twist_rest.to_3x3().normalized()
    r_twist_pose = twist_pose.to_3x3().normalized()

    delta_twist_world = (r_twist_pose @ r_twist_rest.inverted_safe()).normalized()
    if twist_is_child:
        # When twist is a child of base, twist_pose already carries base_pose's rotation delta.
        if delta_twist_world.to_quaternion().angle < 1e-6:
            r_merged = r_base_pose
        else:
            r_merged = (delta_twist_world @ r_base_rest).normalized()
    else:
        delta_base_world = (r_base_pose @ r_base_rest.inverted_safe()).normalized()
        r_merged = (delta_base_world @ delta_twist_world @ r_base_rest).normalized()

    if len(base_pose) == 3:
        return r_merged
    return Matrix.Translation(base_pose.translation) @ r_merged.to_4x4()


def compute_ik_pole_world(
    a_pos: Vector,
    b_pos: Vector,
    c_pos: Vector,
    limb_kind: str = "ARM",
    fallback_dir: Vector | None = None,
    pole_factor: float = 0.45,
) -> Vector:
    """Project mid-joint B onto AC to place the IK pole target along the bend plane."""
    a = Vector(a_pos)
    b = Vector(b_pos)
    c = Vector(c_pos)

    ac = c - a
    ab = b - a
    ac_len_sq = ac.length_squared
    ac_len = math.sqrt(ac_len_sq)

    if ac_len_sq > 1e-12:
        t = ab.dot(ac) / ac_len_sq
        p_proj = a + ac * t
    else:
        p_proj = a.copy()

    if fallback_dir is not None and Vector(fallback_dir).length > 1e-6:
        raw_dir = Vector(fallback_dir).normalized()
    else:
        # MMD and MMR models face -Y in Blender, so knees bend toward -Y and elbows toward +Y.
        kind = (limb_kind or "ARM").upper()
        raw_dir = Vector((0.0, 1.0, 0.0)) if kind == "ARM" else Vector((0.0, -1.0, 0.0))

    if ac_len > 1e-6:
        ac_unit = ac / ac_len
        ortho = raw_dir - ac_unit * raw_dir.dot(ac_unit)
        if ortho.length < 1e-6:
            alt = Vector((1.0, 0.0, 0.0)) if abs(ac_unit.x) < 0.9 else Vector((0.0, 0.0, 1.0))
            ortho = alt - ac_unit * alt.dot(ac_unit)
        safe_fallback = ortho.normalized()
    else:
        safe_fallback = raw_dir.normalized()

    d_bend = b - p_proj
    bend_len = d_bend.length
    # Smoothly blend toward the anatomical fallback when the limb is nearly straight to prevent pole snaps.
    blend_threshold = 0.015
    if bend_len >= blend_threshold:
        bend_dir = d_bend / bend_len
    elif bend_len >= 1e-4:
        alpha = bend_len / blend_threshold
        blended = safe_fallback * (1.0 - alpha) + (d_bend / bend_len) * alpha
        bend_dir = blended.normalized() if blended.length > 1e-6 else safe_fallback
    else:
        bend_dir = safe_fallback

    dist = pole_factor * ac_len if ac_len >= 1e-4 else max(pole_factor * ab.length * 2.0, 0.15)
    return b + bend_dir * dist


def _bone_rest_world_z(arm_obj: Any, bone_name: str) -> float | None:
    if arm_obj is None:
        return None
    arm_world = getattr(arm_obj, "matrix_world", None) or Matrix.Identity(4)
    data_bones = getattr(getattr(arm_obj, "data", None), "bones", None)
    if data_bones is not None and bone_name in data_bones:
        b_mat = arm_world @ data_bones[bone_name].matrix_local
        return float(b_mat.translation.z - arm_world.translation.z)
    pose_bones = getattr(getattr(arm_obj, "pose", None), "bones", None)
    if pose_bones is not None and bone_name in pose_bones:
        pbone = pose_bones[bone_name]
        b_mat = arm_world @ pbone.bone.matrix_local
        return float(b_mat.translation.z - arm_world.translation.z)
    return None


def compute_hip_height_ratio(src_arm: Any, tgt_rig: Any) -> float:
    """Compute target-to-source rest hip height ratio for root translation scaling."""
    if src_arm is None or tgt_rig is None:
        return 1.0

    src_hip_name = bone_map.resolve_source_bone(
        src_arm,
        ("下半身", "腰", "グルーブ", "センター"),
    )
    if src_hip_name is None:
        return 1.0
    src_z = _bone_rest_world_z(src_arm, src_hip_name)
    if src_z is None or abs(src_z) < 1e-4:
        return 1.0

    tgt_z: float | None = None
    for candidate in ("torso", "hips", "ORG-spine", "DEF-spine", "spine_fk"):
        val = _bone_rest_world_z(tgt_rig, candidate)
        if val is not None and abs(val) > 1e-4:
            tgt_z = val
            break

    if tgt_z is None:
        return 1.0

    ratio = abs(tgt_z / src_z)
    return max(0.05, min(10.0, float(ratio)))


def capture_armature_snapshot(src_arm: Any) -> PoseSnapshot:
    """Capture rest and posed world-space matrices for all non-helper bones on a source armature."""
    arm_world = (getattr(src_arm, "matrix_world", None) or Matrix.Identity(4)).copy()
    bones: dict[str, BoneTransform] = {}
    pose_bones = getattr(getattr(src_arm, "pose", None), "bones", None) or ()

    for pbone in pose_bones:
        bname = getattr(pbone, "name", "")
        if not bname or bname.startswith(("_dummy_", "_shadow_")):
            continue
        if getattr(pbone, "is_mmd_shadow_bone", False):
            continue
        rest_world = arm_world @ pbone.bone.matrix_local
        pose_world = arm_world @ pbone.matrix
        tf = BoneTransform(rest_world=rest_world, pose_world=pose_world)
        bones[bname] = tf

        mmd_bone = getattr(pbone, "mmd_bone", None)
        name_j = getattr(mmd_bone, "name_j", None) if mmd_bone is not None else None
        if isinstance(name_j, str) and name_j.strip() and not name_j.strip().startswith(("_dummy_", "_shadow_")):
            bones.setdefault(name_j.strip(), tf)

    from . import morph_retarget

    morphs = morph_retarget.capture_armature_morphs(src_arm)
    return PoseSnapshot(armature_world=arm_world, bones=bones, morphs=morphs)


def _has_modified_ancestor(pbone: Any, known_arm_matrices: dict[str, Matrix]) -> bool:
    curr = getattr(pbone, "parent", None)
    while curr is not None:
        if curr.name in known_arm_matrices:
            return True
        curr = getattr(curr, "parent", None)
    return False


def _get_effective_arm_matrix(pbone: Any, known_arm_matrices: dict[str, Matrix]) -> Matrix:
    if pbone.name in known_arm_matrices:
        return known_arm_matrices[pbone.name]
    if not _has_modified_ancestor(pbone, known_arm_matrices):
        return pbone.matrix.copy()

    parent = pbone.parent
    if parent is None:
        eff = pbone.bone.matrix_local @ pbone.matrix_basis
    else:
        parent_arm = _get_effective_arm_matrix(parent, known_arm_matrices)
        rest_rel = parent.bone.matrix_local.inverted_safe() @ pbone.bone.matrix_local
        eff = parent_arm @ rest_rel @ pbone.matrix_basis

    known_arm_matrices[pbone.name] = eff
    return eff


def _get_rest_in_parent_pose_arm(pbone: Any, known_arm_matrices: dict[str, Matrix]) -> Matrix:
    parent = getattr(pbone, "parent", None)
    if parent is None:
        return pbone.bone.matrix_local.copy()

    parent_arm = _get_effective_arm_matrix(parent, known_arm_matrices)
    rest_rel = parent.bone.matrix_local.inverted_safe() @ pbone.bone.matrix_local
    rest_in_pose = parent_arm @ rest_rel

    if not getattr(pbone.bone, "use_inherit_rotation", True):
        loc = rest_in_pose.translation
        rot = pbone.bone.matrix_local.to_3x3().normalized()
        return Matrix.Translation(loc) @ rot.to_4x4()
    return rest_in_pose


def _assign_pose_bone_basis_rotation(pbone: Any, r_basis: Matrix) -> None:
    q = r_basis.to_quaternion().normalized()
    mode = getattr(pbone, "rotation_mode", "QUATERNION")
    if mode == "QUATERNION":
        prev_q = Quaternion(pbone.rotation_quaternion)
        if prev_q.magnitude > 1e-6 and q.dot(prev_q) < 0.0:
            q = -q
        pbone.rotation_quaternion = q
    elif mode == "AXIS_ANGLE":
        axis, angle = q.to_axis_angle()
        pbone.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
    else:
        pbone.rotation_euler = r_basis.to_euler(mode, pbone.rotation_euler)


def _apply_world_transform_to_bone(
    target_rig: Any,
    pbone: Any,
    target_pos_world: Vector | None,
    target_rot_world: Matrix | None,
    known_arm_matrices: dict[str, Matrix],
    options: RetargetOptions,
    frame: float,
) -> None:
    arm_world = getattr(target_rig, "matrix_world", None) or Matrix.Identity(4)
    arm_world_inv = arm_world.inverted_safe()
    r_arm_inv = arm_world_inv.to_3x3().normalized()

    rest_in_pose = _get_rest_in_parent_pose_arm(pbone, known_arm_matrices)
    r_rest_in_pose = rest_in_pose.to_3x3().normalized()

    do_loc = target_pos_world is not None
    do_rot = target_rot_world is not None

    if do_loc:
        target_pos_arm = arm_world_inv @ target_pos_world
        v_loc = rest_in_pose.inverted_safe() @ target_pos_arm
        pbone.location = v_loc
    else:
        v_loc = Vector(pbone.location)

    if do_rot:
        r_target_arm = (r_arm_inv @ target_rot_world).normalized()
        r_basis = (r_rest_in_pose.inverted_safe() @ r_target_arm).normalized()
        _assign_pose_bone_basis_rotation(pbone, r_basis)
    else:
        r_basis = pbone.matrix_basis.to_3x3().normalized()

    new_basis = Matrix.Translation(v_loc) @ r_basis.to_4x4()
    known_arm_matrices[pbone.name] = rest_in_pose @ new_basis

    if options.insert_keyframes:
        compat.insert_pose_keyframes(
            pbone,
            frame=frame,
            do_loc=do_loc,
            do_rot=do_rot,
            do_scale=False,
            group_name=pbone.name,
        )


def _retarget_bone_rotation(
    target_rig: Any,
    pbone: Any,
    src_rest_world: Matrix,
    src_pose_world: Matrix,
    known_arm_matrices: dict[str, Matrix],
    options: RetargetOptions,
    frame: float,
) -> None:
    arm_world = getattr(target_rig, "matrix_world", None) or Matrix.Identity(4)
    tgt_rest_world = arm_world @ pbone.bone.matrix_local
    r_offset = compute_rest_rotation_offset(src_rest_world, tgt_rest_world)
    r_target_world = (src_pose_world.to_3x3().normalized() @ r_offset).normalized()
    _apply_world_transform_to_bone(
        target_rig=target_rig,
        pbone=pbone,
        target_pos_world=None,
        target_rot_world=r_target_world,
        known_arm_matrices=known_arm_matrices,
        options=options,
        frame=frame,
    )


def _world_rot_delta_angle(tf: BoneTransform) -> float:
    r_rest = tf.rest_world.to_3x3().normalized().to_quaternion()
    r_pose = tf.pose_world.to_3x3().normalized().to_quaternion()
    return float(r_rest.rotation_difference(r_pose).angle)


def _resolve_snapshot_tf(
    snapshot: PoseSnapshot,
    candidates: tuple[str, ...],
) -> tuple[str | None, BoneTransform | None]:
    src_name = bone_map.resolve_source_bone(snapshot.bones, candidates)
    if src_name is None:
        return None, None
    return src_name, snapshot.bones.get(src_name)


def retarget_pose_from_snapshot(
    context: Any,
    snapshot: PoseSnapshot,
    target_rig: Any,
    options: RetargetOptions | None = None,
) -> dict[str, int]:
    """Apply a world-space PoseSnapshot onto a MikuMikuRig control armature in parent-to-child order."""
    opts = options or RetargetOptions()
    scene = getattr(context, "scene", None) if context is not None else None
    frame = float(
        opts.frame
        if opts.frame is not None
        else (getattr(scene, "frame_current", 1) if scene is not None else 1)
    )

    if opts.insert_keyframes:
        compat.ensure_action_and_slot(target_rig)

    arm_world = getattr(target_rig, "matrix_world", None) or Matrix.Identity(4)
    pose_bones = target_rig.pose.bones
    known_arm_matrices: dict[str, Matrix] = {}
    stats: dict[str, int] = {
        "torso": 0,
        "head": 0,
        "arms": 0,
        "legs": 0,
        "fingers": 0,
        "total": 0,
    }

    if opts.use_torso:
        torso_pb = pose_bones.get("torso")
        if torso_pb is not None:
            hip_chain = ("腰", "下半身", "グルーブ", "センター", "全ての親")
            present_tfs: list[BoneTransform] = []
            for cand in hip_chain:
                _, tf = _resolve_snapshot_tf(snapshot, (cand,))
                if tf is not None and tf not in present_tfs:
                    present_tfs.append(tf)

            if present_tfs:
                best_pos_tf = max(
                    present_tfs,
                    key=lambda t: (t.pose_world.translation - t.rest_world.translation).length,
                )
                best_rot_tf = present_tfs[0]
                for tf in present_tfs:
                    if _world_rot_delta_angle(tf) > 1e-5:
                        best_rot_tf = tf
                        break

                delta_pos = (
                    best_pos_tf.pose_world.translation - best_pos_tf.rest_world.translation
                ) * float(opts.hip_scale_ratio)
                tgt_torso_rest = arm_world @ torso_pb.bone.matrix_local
                target_pos_world = tgt_torso_rest.translation + delta_pos
                r_offset = compute_rest_rotation_offset(best_rot_tf.rest_world, tgt_torso_rest)
                target_rot_world = (
                    best_rot_tf.pose_world.to_3x3().normalized() @ r_offset
                ).normalized()
                _apply_world_transform_to_bone(
                    target_rig=target_rig,
                    pbone=torso_pb,
                    target_pos_world=target_pos_world,
                    target_rot_world=target_rot_world,
                    known_arm_matrices=known_arm_matrices,
                    options=opts,
                    frame=frame,
                )
                stats["torso"] += 1

        hips_pb = pose_bones.get("hips")
        if hips_pb is not None:
            _, hips_tf = _resolve_snapshot_tf(snapshot, ("下半身", "腰"))
            if hips_tf is not None:
                _retarget_bone_rotation(
                    target_rig,
                    hips_pb,
                    hips_tf.rest_world,
                    hips_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["torso"] += 1

        spine_fk_pb = pose_bones.get("spine_fk")
        if spine_fk_pb is not None:
            _, sfk_tf = _resolve_snapshot_tf(
                snapshot,
                bone_map.TORSO_SPINE_HEAD_CANDIDATES["spine_fk"],
            )
            if sfk_tf is not None:
                _retarget_bone_rotation(
                    target_rig,
                    spine_fk_pb,
                    sfk_tf.rest_world,
                    sfk_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["torso"] += 1

        chest_pb = pose_bones.get("chest")
        if chest_pb is not None and (
            not getattr(chest_pb, "children", ()) or "spine_fk.001" not in pose_bones
        ):
            _, chest_tf = _resolve_snapshot_tf(
                snapshot,
                bone_map.TORSO_SPINE_HEAD_CANDIDATES["chest"],
            )
            if chest_tf is not None:
                _retarget_bone_rotation(
                    target_rig,
                    chest_pb,
                    chest_tf.rest_world,
                    chest_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["torso"] += 1

        for spine_name in ("spine_fk.001", "spine_fk.002", "spine_fk.003"):
            sp_pb = pose_bones.get(spine_name)
            if sp_pb is None:
                continue
            _, sp_tf = _resolve_snapshot_tf(
                snapshot,
                bone_map.TORSO_SPINE_HEAD_CANDIDATES[spine_name],
            )
            if sp_tf is not None:
                _retarget_bone_rotation(
                    target_rig,
                    sp_pb,
                    sp_tf.rest_world,
                    sp_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["torso"] += 1

        for side in ("L", "R"):
            sh_spec = bone_map.SHOULDER_CANDIDATES[side]
            sh_pb = pose_bones.get(sh_spec["target"][0])
            if sh_pb is None:
                continue
            _, sh_tf = _resolve_snapshot_tf(snapshot, sh_spec["shoulder"])
            _, sh_p_tf = _resolve_snapshot_tf(snapshot, sh_spec["shoulder_p"])
            chosen_tf = sh_tf
            if chosen_tf is None or (
                sh_p_tf is not None
                and _world_rot_delta_angle(chosen_tf) < 1e-5
                and _world_rot_delta_angle(sh_p_tf) > 1e-5
            ):
                chosen_tf = sh_p_tf
            if chosen_tf is not None:
                _retarget_bone_rotation(
                    target_rig,
                    sh_pb,
                    chosen_tf.rest_world,
                    chosen_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["torso"] += 1

    if context is not None and hasattr(context, "view_layer"):
        context.view_layer.update()
        known_arm_matrices.clear()

    if opts.use_head:
        for head_bone in ("neck", "head", "eyes", "eye.L", "eye.R"):
            h_pb = pose_bones.get(head_bone)
            if h_pb is None:
                continue
            _, h_tf = _resolve_snapshot_tf(
                snapshot,
                bone_map.TORSO_SPINE_HEAD_CANDIDATES[head_bone],
            )
            if h_tf is not None:
                _retarget_bone_rotation(
                    target_rig,
                    h_pb,
                    h_tf.rest_world,
                    h_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["head"] += 1

    if opts.use_arms:
        arms_ik = opts.arms_mode.upper() == "IK"
        for side in ("L", "R"):
            arm_spec = bone_map.ARM_SPEC[side]
            switch_pb = pose_bones.get(arm_spec["switch_bone"])
            _, upper_tf = _resolve_snapshot_tf(snapshot, (f"腕.{side}",))
            _, upper_twist_tf = _resolve_snapshot_tf(snapshot, arm_spec["twist"]["upper_twist"])
            _, forearm_tf = _resolve_snapshot_tf(snapshot, (f"ひじ.{side}",))
            _, lower_twist_tf = _resolve_snapshot_tf(snapshot, arm_spec["twist"]["lower_twist"])
            _, wrist_tf = _resolve_snapshot_tf(snapshot, (f"手首.{side}",))

            if arms_ik:
                if switch_pb is not None:
                    switch_pb["IK_FK"] = 0.0
                    switch_pb["pole_vector"] = True
                    if opts.insert_keyframes:
                        compat.insert_custom_prop_keyframe(switch_pb, "IK_FK", frame)
                        compat.insert_custom_prop_keyframe(switch_pb, "pole_vector", frame)

                hand_ik_pb = pose_bones.get(f"hand_ik.{side}")
                if hand_ik_pb is not None and wrist_tf is not None:
                    tgt_rest = arm_world @ hand_ik_pb.bone.matrix_local
                    r_offset = compute_rest_rotation_offset(wrist_tf.rest_world, tgt_rest)
                    r_target = (wrist_tf.pose_world.to_3x3().normalized() @ r_offset).normalized()
                    _apply_world_transform_to_bone(
                        target_rig=target_rig,
                        pbone=hand_ik_pb,
                        target_pos_world=wrist_tf.pose_world.translation,
                        target_rot_world=r_target,
                        known_arm_matrices=known_arm_matrices,
                        options=opts,
                        frame=frame,
                    )
                    stats["arms"] += 1

                pole_pb = pose_bones.get(arm_spec["ik"]["pole_target"])
                if (
                    pole_pb is not None
                    and upper_tf is not None
                    and forearm_tf is not None
                    and wrist_tf is not None
                ):
                    pole_pos = compute_ik_pole_world(
                        upper_tf.pose_world.translation,
                        forearm_tf.pose_world.translation,
                        wrist_tf.pose_world.translation,
                        limb_kind="ARM",
                    )
                    _apply_world_transform_to_bone(
                        target_rig=target_rig,
                        pbone=pole_pb,
                        target_pos_world=pole_pos,
                        target_rot_world=None,
                        known_arm_matrices=known_arm_matrices,
                        options=opts,
                        frame=frame,
                    )
                    stats["arms"] += 1
            else:
                if switch_pb is not None:
                    switch_pb["IK_FK"] = 1.0
                    if opts.insert_keyframes:
                        compat.insert_custom_prop_keyframe(switch_pb, "IK_FK", frame)

                ua_pb = pose_bones.get(f"upper_arm_fk.{side}")
                if ua_pb is not None and upper_tf is not None:
                    merged_ua = merge_twist_world_rotation(
                        upper_tf.rest_world,
                        upper_tf.pose_world,
                        upper_twist_tf.rest_world if upper_twist_tf is not None else None,
                        upper_twist_tf.pose_world if upper_twist_tf is not None else None,
                        twist_is_child=True,
                    )
                    _retarget_bone_rotation(
                        target_rig,
                        ua_pb,
                        upper_tf.rest_world,
                        merged_ua,
                        known_arm_matrices,
                        opts,
                        frame,
                    )
                    stats["arms"] += 1

                fa_pb = pose_bones.get(f"forearm_fk.{side}")
                if fa_pb is not None and forearm_tf is not None:
                    merged_fa = merge_twist_world_rotation(
                        forearm_tf.rest_world,
                        forearm_tf.pose_world,
                        lower_twist_tf.rest_world if lower_twist_tf is not None else None,
                        lower_twist_tf.pose_world if lower_twist_tf is not None else None,
                        twist_is_child=True,
                    )
                    _retarget_bone_rotation(
                        target_rig,
                        fa_pb,
                        forearm_tf.rest_world,
                        merged_fa,
                        known_arm_matrices,
                        opts,
                        frame,
                    )
                    stats["arms"] += 1

                ha_pb = pose_bones.get(f"hand_fk.{side}")
                if ha_pb is not None and wrist_tf is not None:
                    _retarget_bone_rotation(
                        target_rig,
                        ha_pb,
                        wrist_tf.rest_world,
                        wrist_tf.pose_world,
                        known_arm_matrices,
                        opts,
                        frame,
                    )
                    stats["arms"] += 1

    if opts.use_legs:
        legs_ik = opts.legs_mode.upper() == "IK"
        for side in ("L", "R"):
            leg_spec = bone_map.LEG_SPEC[side]
            switch_pb = pose_bones.get(leg_spec["switch_bone"])
            _, thigh_tf = _resolve_snapshot_tf(snapshot, leg_spec["fk"][f"thigh_fk.{side}"])
            _, shin_tf = _resolve_snapshot_tf(snapshot, leg_spec["fk"][f"shin_fk.{side}"])
            _, ankle_tf = _resolve_snapshot_tf(snapshot, leg_spec["ik"][f"foot_ik.{side}"])
            _, toe_tf = _resolve_snapshot_tf(snapshot, leg_spec["ik"][f"toe.{side}"])

            if legs_ik:
                if switch_pb is not None:
                    switch_pb["IK_FK"] = 0.0
                    switch_pb["pole_vector"] = True
                    if opts.insert_keyframes:
                        compat.insert_custom_prop_keyframe(switch_pb, "IK_FK", frame)
                        compat.insert_custom_prop_keyframe(switch_pb, "pole_vector", frame)

                foot_ik_pb = pose_bones.get(f"foot_ik.{side}")
                if foot_ik_pb is not None and ankle_tf is not None:
                    tgt_rest = arm_world @ foot_ik_pb.bone.matrix_local
                    r_offset = compute_rest_rotation_offset(ankle_tf.rest_world, tgt_rest)
                    r_target = (ankle_tf.pose_world.to_3x3().normalized() @ r_offset).normalized()
                    _apply_world_transform_to_bone(
                        target_rig=target_rig,
                        pbone=foot_ik_pb,
                        target_pos_world=ankle_tf.pose_world.translation,
                        target_rot_world=r_target,
                        known_arm_matrices=known_arm_matrices,
                        options=opts,
                        frame=frame,
                    )
                    stats["legs"] += 1

                pole_pb = pose_bones.get(leg_spec["ik"]["pole_target"])
                if (
                    pole_pb is not None
                    and thigh_tf is not None
                    and shin_tf is not None
                    and ankle_tf is not None
                ):
                    pole_pos = compute_ik_pole_world(
                        thigh_tf.pose_world.translation,
                        shin_tf.pose_world.translation,
                        ankle_tf.pose_world.translation,
                        limb_kind="LEG",
                    )
                    _apply_world_transform_to_bone(
                        target_rig=target_rig,
                        pbone=pole_pb,
                        target_pos_world=pole_pos,
                        target_rot_world=None,
                        known_arm_matrices=known_arm_matrices,
                        options=opts,
                        frame=frame,
                    )
                    stats["legs"] += 1

                toe_pb = pose_bones.get(f"toe.{side}")
                if toe_pb is not None and toe_tf is not None:
                    _retarget_bone_rotation(
                        target_rig,
                        toe_pb,
                        toe_tf.rest_world,
                        toe_tf.pose_world,
                        known_arm_matrices,
                        opts,
                        frame,
                    )
                    stats["legs"] += 1
            else:
                if switch_pb is not None:
                    switch_pb["IK_FK"] = 1.0
                    if opts.insert_keyframes:
                        compat.insert_custom_prop_keyframe(switch_pb, "IK_FK", frame)

                for tgt_bone, src_tf in (
                    (f"thigh_fk.{side}", thigh_tf),
                    (f"shin_fk.{side}", shin_tf),
                    (f"foot_fk.{side}", ankle_tf),
                    (f"toe.{side}", toe_tf),
                ):
                    pb = pose_bones.get(tgt_bone)
                    if pb is not None and src_tf is not None:
                        _retarget_bone_rotation(
                            target_rig,
                            pb,
                            src_tf.rest_world,
                            src_tf.pose_world,
                            known_arm_matrices,
                            opts,
                            frame,
                        )
                        stats["legs"] += 1

    if opts.use_fingers:
        if context is not None and hasattr(context, "view_layer"):
            context.view_layer.update()
            known_arm_matrices.clear()

        for side in ("L", "R"):
            for finger in bone_map.FINGER_NAMES:
                ik_switch_name = bone_map.FINGER_IK_SWITCH_BONES[(finger, side)]
                ik_pb = pose_bones.get(ik_switch_name)
                if ik_pb is not None:
                    # Rigify finger IK controllers use FK_IK = 0.0 to keep fingers in pure FK mode.
                    ik_pb["FK_IK"] = 0.0
                    if "IK_FK" in ik_pb:
                        ik_pb["IK_FK"] = 1.0
                    if opts.insert_keyframes:
                        compat.insert_custom_prop_keyframe(ik_pb, "FK_IK", frame)
                        if "IK_FK" in ik_pb:
                            compat.insert_custom_prop_keyframe(ik_pb, "IK_FK", frame)

                master_name = bone_map.FINGER_MASTER_BONES[(finger, side)]
                master_pb = pose_bones.get(master_name)
                if master_pb is not None:
                    master_pb.matrix_basis = Matrix.Identity(4)
                    master_pb.location = (0.0, 0.0, 0.0)
                    master_pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                    master_pb.rotation_euler = (0.0, 0.0, 0.0)
                    master_pb.scale = (1.0, 1.0, 1.0)
                    rest_in_pose = _get_rest_in_parent_pose_arm(master_pb, known_arm_matrices)
                    known_arm_matrices[master_pb.name] = rest_in_pose
                    if opts.insert_keyframes:
                        compat.insert_pose_keyframes(
                            master_pb,
                            frame=frame,
                            do_loc=True,
                            do_rot=True,
                            do_scale=True,
                            group_name=master_pb.name,
                        )

            fk_pairs = bone_map.resolve_finger_fk_pairs(snapshot.bones, target_rig, side)
            for src_name, tgt_name in fk_pairs:
                if "_master" in tgt_name or "_ik" in tgt_name:
                    continue
                src_tf = snapshot.bones.get(src_name)
                tgt_pb = pose_bones.get(tgt_name)
                if src_tf is None or tgt_pb is None:
                    continue
                _retarget_bone_rotation(
                    target_rig,
                    tgt_pb,
                    src_tf.rest_world,
                    src_tf.pose_world,
                    known_arm_matrices,
                    opts,
                    frame,
                )
                stats["fingers"] += 1

    if opts.use_morphs and snapshot.morphs:
        from . import morph_retarget

        morph_stats = morph_retarget.apply_morphs_to_target(
            context,
            target_rig,
            snapshot.morphs,
            insert_keyframes=opts.insert_keyframes,
            frame=frame,
        )
        stats.update(morph_stats)

    if context is not None and hasattr(context, "view_layer"):
        context.view_layer.update()

    stats["total"] = (
        stats["torso"] + stats["head"] + stats["arms"] + stats["legs"] + stats["fingers"]
    )
    return stats


def retarget_armature_frame(
    context: Any,
    src_arm: Any,
    target_rig: Any,
    options: RetargetOptions | None = None,
) -> dict[str, int]:
    """Capture the current frame from a live MMD armature and retarget it onto the MMR rig."""
    opts = options or RetargetOptions()
    scene = getattr(context, "scene", None) if context is not None else None
    if opts.frame is not None and scene is not None:
        scene.frame_set(int(opts.frame))
    if context is not None and hasattr(context, "view_layer"):
        context.view_layer.update()

    snapshot = capture_armature_snapshot(src_arm)
    return retarget_pose_from_snapshot(context, snapshot, target_rig, opts)


def bake_armature_range(
    context: Any,
    src_arm: Any,
    target_rig: Any,
    options: RetargetOptions | None = None,
    frame_start: int = 1,
    frame_end: int = 120,
    frame_step: int = 1,
) -> dict[str, int]:
    """Step through a frame range and bake retargeted keyframes onto the MMR control rig."""
    opts = options or RetargetOptions()
    start = int(frame_start)
    end = int(frame_end)
    if end < start:
        start, end = end, start
    step = max(1, int(frame_step))

    compat.ensure_action_and_slot(target_rig)
    scene = getattr(context, "scene", None) if context is not None else None
    original_frame = getattr(scene, "frame_current", start) if scene is not None else start

    frames_baked = 0
    last_stats: dict[str, int] = {
        "torso": 0,
        "head": 0,
        "arms": 0,
        "legs": 0,
        "fingers": 0,
        "total": 0,
    }

    for frame in range(start, end + 1, step):
        frame_opts = replace(opts, insert_keyframes=True, frame=frame)
        last_stats = retarget_armature_frame(context, src_arm, target_rig, frame_opts)
        frames_baked += 1

    if scene is not None:
        scene.frame_set(int(original_frame))
    if context is not None and hasattr(context, "view_layer"):
        context.view_layer.update()

    return {
        **last_stats,
        "frames_baked": frames_baked,
        "bones_posed": last_stats["total"],
    }
