# SPDX-License-Identifier: GPL-3.0-or-later
"""PoseSnap Blender add-on registration, preferences, and scene settings."""

from __future__ import annotations

import bpy

from . import compat, ui_panel

bl_info = {
    "name": "PoseSnap",
    "author": "RyuuRead",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > PoseSnap / MMR",
    "description": "MMD (.vpd/.vmd) pose and animation retargeter to MikuMikuRig IK/FK controllers",
    "category": "Animation",
}


def _get_preferences() -> bpy.types.AddonPreferences | None:
    prefs_root = getattr(bpy.context, "preferences", None)
    addons = getattr(prefs_root, "addons", None) if prefs_root is not None else None
    if addons is None:
        return None
    for key in (__package__, "posesnap"):
        if not key:
            continue
        entry = addons.get(key)
        if entry is not None:
            return getattr(entry, "preferences", None)
    return None


def _on_dock_in_mmr_update(self: PoseSnapPreferences, context: bpy.types.Context) -> None:
    compat.refresh_panel_category((ui_panel.POSESNAP_PT_main,), bool(self.dock_in_mmr))


def _poll_source_armature(self: PoseSnapSettings, obj: bpy.types.Object) -> bool:
    return obj.type == "ARMATURE" and obj != self.target_rig


def _poll_target_rig(self: PoseSnapSettings, obj: bpy.types.Object) -> bool:
    return obj.type == "ARMATURE" and obj != self.source_armature


class PoseSnapPreferences(bpy.types.AddonPreferences):
    """User preferences for PoseSnap sidebar docking and VPD scale"""

    bl_idname = __package__ or "posesnap"

    dock_in_mmr: bpy.props.BoolProperty(
        name="Dock in MMR Tab",
        description="Place PoseSnap inside the MikuMikuRig sidebar tab when MMR panels are active",
        default=True,
        update=_on_dock_in_mmr_update,
    )
    vpd_scale: bpy.props.FloatProperty(
        name="VPD Scale",
        description="World unit scale factor applied when converting .vpd translations to Blender",
        default=0.08,
        min=0.001,
        max=10.0,
        precision=4,
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.prop(self, "dock_in_mmr")
        layout.prop(self, "vpd_scale")


class PoseSnapSettings(bpy.types.PropertyGroup):
    """Scene-level rig selection, IK/FK modes, and baking settings"""

    source_armature: bpy.props.PointerProperty(
        name="Source MMD",
        description="Standard MMD armature with Japanese bone names",
        type=bpy.types.Object,
        poll=_poll_source_armature,
    )
    target_rig: bpy.props.PointerProperty(
        name="Target MMR",
        description="MikuMikuRig control rig (RIG-*) to receive retargeted poses",
        type=bpy.types.Object,
        poll=_poll_target_rig,
    )
    arms_mode: bpy.props.EnumProperty(
        name="Arms Mode",
        description="Choose IK targets or FK limb controllers for arm retargeting",
        items=(
            ("IK", "IK", "Retarget wrists to hand_ik and elbows to upper_arm_ik_target"),
            ("FK", "FK", "Retarget upper arm, forearm, and hand FK bones with twist merging"),
        ),
        default="IK",
    )
    legs_mode: bpy.props.EnumProperty(
        name="Legs Mode",
        description="Choose IK targets or FK limb controllers for leg retargeting",
        items=(
            ("IK", "IK", "Retarget ankles to foot_ik and knees to thigh_ik_target"),
            ("FK", "FK", "Retarget thigh, shin, foot, and toe FK bones"),
        ),
        default="IK",
    )
    use_torso: bpy.props.BoolProperty(
        name="Torso & Spine",
        description="Retarget root, hips, spine, chest, and shoulder bones",
        default=True,
    )
    use_head: bpy.props.BoolProperty(
        name="Head & Eyes",
        description="Retarget neck, head, and eye controllers",
        default=True,
    )
    use_arms: bpy.props.BoolProperty(
        name="Arms & Hands",
        description="Retarget arm IK or FK controllers",
        default=True,
    )
    use_fingers: bpy.props.BoolProperty(
        name="Fingers (FK)",
        description="Retarget finger bones using per-segment FK controls",
        default=True,
    )
    use_legs: bpy.props.BoolProperty(
        name="Legs & Feet",
        description="Retarget leg IK or FK controllers and toes",
        default=True,
    )
    use_morphs: bpy.props.BoolProperty(
        name="Face Morphs",
        description="Transfer facial shape keys and MorphRig sliders or mouth joystick",
        default=True,
    )
    hip_scale_ratio: bpy.props.FloatProperty(
        name="Hip Scale Ratio",
        description="Height ratio multiplier applied to root and torso translations",
        default=1.0,
        min=0.05,
        max=10.0,
        precision=3,
    )
    auto_insert_keyframe: bpy.props.BoolProperty(
        name="Auto Insert Keyframe",
        description="Insert keyframes on posed bones and IK/FK switches during pose snap",
        default=False,
    )
    frame_start: bpy.props.IntProperty(
        name="Start Frame",
        description="First frame to bake from Source MMD",
        default=1,
        min=0,
    )
    frame_end: bpy.props.IntProperty(
        name="End Frame",
        description="Last frame to bake from Source MMD",
        default=120,
        min=0,
    )
    frame_step: bpy.props.IntProperty(
        name="Frame Step",
        description="Step interval between baked frames (1 for every frame, 2 for on-twos)",
        default=1,
        min=1,
        max=10,
    )
    auto_cleanup_vmd: bpy.props.BoolProperty(
        name="Automatic Cleanup on VMD Import",
        description="Automatically fix quaternion or Euler flips, IK pole spikes, Bezier overshoot, and redundant keyframes after VMD import",
        default=True,
    )
    last_status: bpy.props.StringProperty(
        name="Last Status",
        description="Latest PoseSnap status or error notice",
        default="",
    )


_BASE_CLASSES = (
    PoseSnapPreferences,
    PoseSnapSettings,
)


def register() -> None:
    """Register PoseSnap preferences, scene properties, operators, and panel."""
    prefs = _get_preferences()
    dock_in_mmr = bool(getattr(prefs, "dock_in_mmr", True)) if prefs is not None else True
    ui_panel.POSESNAP_PT_main.bl_category = compat.resolve_panel_category(dock_in_mmr)

    for cls in (*_BASE_CLASSES, *ui_panel.CLASSES):
        bpy.utils.register_class(cls)

    bpy.types.Scene.posesnap = bpy.props.PointerProperty(type=PoseSnapSettings)


def unregister() -> None:
    """Unregister PoseSnap scene properties and Blender classes."""
    if hasattr(bpy.types.Scene, "posesnap"):
        del bpy.types.Scene.posesnap

    for cls in reversed((*_BASE_CLASSES, *ui_panel.CLASSES)):
        if getattr(cls, "is_registered", False):
            bpy.utils.unregister_class(cls)
