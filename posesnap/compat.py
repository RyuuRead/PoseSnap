# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender API compatibility helpers across Goo Engine 4.4, 4.5 LTS, and 5.0+."""

from __future__ import annotations

from typing import Any, Iterable

import bpy

_ICON_CACHE: frozenset[str] | None = None
_DEFAULT_CATEGORY = "PoseSnap"


def _is_layered_action_version() -> bool:
    # Blender 5.0 removed native Action.fcurves; mmd_tools may monkey-patch it back.
    return tuple(bpy.app.version[:3]) >= (5, 0, 0)


def _find_matching_object_slot(action: Any, obj_name: str) -> Any:
    for slot in getattr(action, "slots", ()):
        if getattr(slot, "target_id_type", None) == "OBJECT":
            if getattr(slot, "name_display", None) == obj_name:
                return slot
    return None


def ensure_action_and_slot(
    arm_obj: bpy.types.Object,
    action_name: str | None = None,
) -> tuple[bpy.types.Action, Any]:
    """Ensure the armature object has AnimationData, an Action, and a 5.0+ ActionSlot."""
    anim_data = arm_obj.animation_data
    if anim_data is None:
        anim_data = arm_obj.animation_data_create()

    action = anim_data.action
    if action is None:
        name = action_name or f"{arm_obj.name}Action"
        action = bpy.data.actions.new(name=name)
        anim_data.action = action

    if not _is_layered_action_version():
        return action, None

    slot = anim_data.action_slot
    if slot is None:
        slot = _find_matching_object_slot(action, arm_obj.name)
        if slot is None:
            slot = action.slots.new(id_type="OBJECT", name=arm_obj.name)
        anim_data.action_slot = slot

    return action, slot


def _resolve_slot_for_read(action: Any, slot: Any, ensure: bool) -> Any:
    if slot is not None:
        return slot
    slots = getattr(action, "slots", None)
    if slots:
        for candidate in slots:
            if getattr(candidate, "target_id_type", None) == "OBJECT":
                return candidate
        return slots[0]
    if ensure and hasattr(action, "slots"):
        return action.slots.new(id_type="OBJECT", name="Slot")
    return None


def get_action_fcurves(action: Any, slot: Any = None, ensure: bool = False) -> Any:
    """Return the F-Curve collection for an Action across Blender 4.4, 4.5, and 5.0+."""
    if action is None:
        return None

    if not _is_layered_action_version():
        return action.fcurves

    resolved_slot = _resolve_slot_for_read(action, slot, ensure=ensure)
    if resolved_slot is None:
        return None

    layers = getattr(action, "layers", None)
    if layers is None:
        return None

    for layer in layers:
        for strip in getattr(layer, "strips", ()):
            if getattr(strip, "type", None) != "KEYFRAME":
                continue
            channelbag = strip.channelbag(resolved_slot, ensure=ensure)
            if channelbag is not None:
                return channelbag.fcurves

    if not ensure:
        return None

    layer = layers[0] if len(layers) > 0 else layers.new(name="Layer")
    strip = layer.strips.new(type="KEYFRAME")
    channelbag = strip.channelbag(resolved_slot, ensure=True)
    return channelbag.fcurves if channelbag is not None else None


def count_action_fcurves(action: Any, slot: Any = None) -> int:
    """Count F-Curves on an Action for a specific slot or across all slots."""
    if action is None:
        return 0

    if not _is_layered_action_version():
        return len(action.fcurves)

    total = 0
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            if getattr(strip, "type", None) != "KEYFRAME":
                continue
            if slot is not None:
                channelbag = strip.channelbag(slot, ensure=False)
                if channelbag is not None:
                    total += len(channelbag.fcurves)
            else:
                for channelbag in getattr(strip, "channelbags", ()):
                    total += len(channelbag.fcurves)
    return total


def _rotation_data_path(pose_bone: bpy.types.PoseBone) -> str:
    mode = getattr(pose_bone, "rotation_mode", "QUATERNION")
    if mode == "QUATERNION":
        return "rotation_quaternion"
    if mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_euler"


def insert_pose_keyframes(
    pose_bone: bpy.types.PoseBone,
    frame: float,
    do_loc: bool = True,
    do_rot: bool = True,
    do_scale: bool = False,
    group_name: str | None = None,
) -> None:
    """Insert location, rotation, and scale keyframes on a PoseBone safely across Blender versions."""
    arm_obj = pose_bone.id_data
    ensure_action_and_slot(arm_obj)

    group = group_name if group_name is not None else pose_bone.name
    paths: list[str] = []
    if do_loc:
        paths.append("location")
    if do_rot:
        paths.append(_rotation_data_path(pose_bone))
    if do_scale:
        paths.append("scale")

    for data_path in paths:
        pose_bone.keyframe_insert(data_path=data_path, frame=frame, group=group)


def insert_custom_prop_keyframe(
    pose_bone: bpy.types.PoseBone,
    prop_name: str,
    frame: float,
) -> None:
    """Insert a keyframe on a custom property of a PoseBone if the property exists."""
    if prop_name not in pose_bone:
        return

    ensure_action_and_slot(pose_bone.id_data)
    escaped = prop_name.replace('"', '\\"')
    pose_bone.keyframe_insert(
        data_path=f'["{escaped}"]',
        frame=frame,
        group=pose_bone.name,
    )


def _load_icon_names() -> frozenset[str]:
    global _ICON_CACHE
    if _ICON_CACHE is not None:
        return _ICON_CACHE

    rna = bpy.types.UILayout.bl_rna
    prop = rna.properties.get("icon")
    if prop is not None and hasattr(prop, "enum_items"):
        _ICON_CACHE = frozenset(prop.enum_items.keys())
        return _ICON_CACHE

    # Blender 4.4+ stores the icon enum on UILayout function parameters rather than properties.
    for fn_name in ("label", "prop", "operator"):
        fn = rna.functions.get(fn_name)
        if fn is None:
            continue
        param = fn.parameters.get("icon")
        if param is not None and hasattr(param, "enum_items"):
            _ICON_CACHE = frozenset(param.enum_items.keys())
            return _ICON_CACHE

    _ICON_CACHE = frozenset({"NONE"})
    return _ICON_CACHE


def safe_icon(icon_name: str, fallback: str = "NONE") -> str:
    """Validate a Blender UI icon name lazily and fall back safely if unavailable."""
    try:
        icons = _load_icon_names()
    except Exception:
        return fallback or "NONE"

    if icon_name in icons:
        return icon_name
    if fallback in icons:
        return fallback
    return "NONE"


def detect_mmr_category() -> str | None:
    """Detect the N-Panel category used by MikuMikuRig if its panels are registered."""
    for panel_id in ("SCENE_PT_MMR_Rig_0", "SCENE_PT_MMR_Rig_1"):
        panel_cls = getattr(bpy.types, panel_id, None)
        category = getattr(panel_cls, "bl_category", None) if panel_cls is not None else None
        if category:
            return str(category)

    fallback_category: str | None = None
    for attr_name in dir(bpy.types):
        if "_PT_MMR_" not in attr_name:
            continue
        panel_cls = getattr(bpy.types, attr_name, None)
        if panel_cls is None:
            continue
        if getattr(panel_cls, "bl_space_type", None) != "VIEW_3D":
            continue
        category = getattr(panel_cls, "bl_category", None)
        if not category:
            continue
        # MMR registers its Key Options panel under "Tool", so prefer non-Tool MMR tabs first.
        if category != "Tool":
            return str(category)
        if fallback_category is None:
            fallback_category = str(category)

    return fallback_category


def resolve_panel_category(dock_in_mmr: bool) -> str:
    """Return the MMR N-Panel tab category when docking is enabled and MMR is active."""
    if dock_in_mmr:
        mmr_category = detect_mmr_category()
        if mmr_category:
            return mmr_category
    return _DEFAULT_CATEGORY


def refresh_panel_category(panel_classes: Iterable[type], dock_in_mmr: bool) -> str:
    """Update bl_category on registered panels and re-register any whose category changed."""
    category = resolve_panel_category(dock_in_mmr)
    for panel_cls in panel_classes:
        if getattr(panel_cls, "bl_category", None) == category:
            continue
        was_registered = bool(getattr(panel_cls, "is_registered", False))
        if was_registered:
            bpy.utils.unregister_class(panel_cls)
        panel_cls.bl_category = category
        if was_registered:
            bpy.utils.register_class(panel_cls)
    return category
