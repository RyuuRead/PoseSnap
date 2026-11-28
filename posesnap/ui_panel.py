# SPDX-License-Identifier: GPL-3.0-or-later
"""N-Panel UI and operators for PoseSnap MMD-to-MMR retargeting."""

from __future__ import annotations

import os
from typing import Any

import bpy

from . import compat


def _get_addon_vpd_scale(context: Any) -> float:
    prefs_root = getattr(context, "preferences", None) or getattr(bpy.context, "preferences", None)
    addons = getattr(prefs_root, "addons", None) if prefs_root is not None else None
    if addons is not None:
        for key in (__package__, "posesnap"):
            if not key:
                continue
            entry = addons.get(key)
            if entry is not None and getattr(entry, "preferences", None) is not None:
                return float(getattr(entry.preferences, "vpd_scale", 0.08))
    return 0.08


def _is_valid_armature(obj: Any) -> bool:
    return obj is not None and getattr(obj, "type", None) == "ARMATURE"


def _is_valid_rig_pair(settings: Any) -> bool:
    if settings is None:
        return False
    src = getattr(settings, "source_armature", None)
    tgt = getattr(settings, "target_rig", None)
    return _is_valid_armature(src) and _is_valid_armature(tgt) and src != tgt


def _has_any_filter(settings: Any) -> bool:
    return bool(
        settings.use_torso
        or settings.use_head
        or settings.use_arms
        or settings.use_fingers
        or settings.use_legs
        or getattr(settings, "use_morphs", True)
    )


def _build_retarget_options(
    settings: Any,
    math_retarget_mod: Any,
    insert_keyframes: bool = False,
    frame: int | None = None,
) -> Any:
    return math_retarget_mod.RetargetOptions(
        arms_mode=settings.arms_mode,
        legs_mode=settings.legs_mode,
        use_torso=bool(settings.use_torso),
        use_head=bool(settings.use_head),
        use_arms=bool(settings.use_arms),
        use_fingers=bool(settings.use_fingers),
        use_legs=bool(settings.use_legs),
        use_morphs=bool(getattr(settings, "use_morphs", True)),
        hip_scale_ratio=float(settings.hip_scale_ratio),
        insert_keyframes=insert_keyframes,
        frame=frame,
    )


class POSESNAP_OT_auto_detect(bpy.types.Operator):
    """Scan scene armatures and populate Source MMD and Target MMR"""

    bl_idname = "posesnap.auto_detect_rigs"
    bl_label = "Auto-Detect Rigs"
    bl_description = "Scan active scene armatures to select the Source MMD armature and Target MMR control rig"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        return scene is not None and hasattr(scene, "posesnap")

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import bone_map

        settings = context.scene.posesnap
        src, tgt = bone_map.auto_detect_rig_pair(context.scene)

        # Clear both pointers first so PointerProperty poll callbacks never block swapping rigs.
        settings.source_armature = None
        settings.target_rig = None
        if src is not None:
            settings.source_armature = src
        if tgt is not None and tgt != src:
            settings.target_rig = tgt

        if src is not None and tgt is not None:
            msg = f"Detected rigs: {src.name} -> {tgt.name}"
            settings.last_status = msg
            self.report({"INFO"}, msg)
            return {"FINISHED"}

        if src is not None or tgt is not None:
            found = src.name if src is not None else tgt.name
            msg = f"Found only {found}. Assign the remaining armature manually."
            settings.last_status = msg
            self.report({"WARNING"}, msg)
            return {"FINISHED"}

        msg = "No MMD or MMR armatures detected in the active scene."
        settings.last_status = msg
        self.report({"WARNING"}, msg)
        return {"CANCELLED"}


class POSESNAP_OT_calibrate_height(bpy.types.Operator):
    """Compute hip height ratio between Source MMD and Target MMR"""

    bl_idname = "posesnap.auto_calibrate_height"
    bl_label = "Auto-Calibrate Height"
    bl_description = "Calculate hip translation scale from rest-pose heights of Source MMD and Target MMR"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        return _is_valid_rig_pair(settings)

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import math_retarget

        settings = context.scene.posesnap
        if not _is_valid_rig_pair(settings):
            msg = "Select distinct Source MMD and Target MMR armatures before calibrating."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        ratio = math_retarget.compute_hip_height_ratio(
            settings.source_armature,
            settings.target_rig,
        )
        clamped = max(0.05, min(10.0, float(ratio)))
        settings.hip_scale_ratio = clamped
        msg = f"Calibrated hip scale ratio to {clamped:.3f}."
        settings.last_status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class POSESNAP_OT_snap_pose(bpy.types.Operator):
    """Retarget the current pose from Source MMD to Target MMR"""

    bl_idname = "posesnap.snap_pose"
    bl_label = "Snap Pose to MMR (Current Frame)"
    bl_description = "Transfer the current frame pose from Source MMD onto Target MMR IK/FK controls"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        return _is_valid_rig_pair(settings) and _has_any_filter(settings)

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import math_retarget

        settings = context.scene.posesnap
        if not _is_valid_rig_pair(settings):
            msg = "Select valid Source MMD and Target MMR armatures first."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if not _has_any_filter(settings):
            msg = "Enable at least one body filter before snapping."
            settings.last_status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        frame = int(context.scene.frame_current)
        options = _build_retarget_options(
            settings,
            math_retarget,
            insert_keyframes=bool(settings.auto_insert_keyframe),
            frame=frame,
        )
        stats = math_retarget.retarget_armature_frame(
            context,
            settings.source_armature,
            settings.target_rig,
            options,
        )
        total = int(stats.get("total", 0))
        if total == 0:
            msg = "No matching bones posed. Check rig bone names and body filters."
            settings.last_status = msg
            self.report({"WARNING"}, msg)
            return {"FINISHED"}

        key_note = " (keyframed)" if settings.auto_insert_keyframe else ""
        msg = f"Snapped {total} bones to {settings.target_rig.name} on frame {frame}{key_note}."
        settings.last_status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class POSESNAP_OT_import_vpd(bpy.types.Operator):
    """Import a .vpd pose file directly onto Target MMR"""

    bl_idname = "posesnap.import_vpd"
    bl_label = "Import .vpd Pose to MMR..."
    bl_description = "Parse a Vocaloid Pose Data (.vpd) file and apply it directly to Target MMR without a source rig"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="VPD File Path",
        description="Path to the .vpd pose file to apply",
        subtype="FILE_PATH",
    )
    filter_glob: bpy.props.StringProperty(
        default="*.vpd",
        options={"HIDDEN"},
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        if settings is None:
            return False
        return _is_valid_armature(settings.target_rig) and _has_any_filter(settings)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        if not self.filepath:
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}
        return self.execute(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import math_retarget, vpd_parser

        settings = context.scene.posesnap
        if not _is_valid_armature(settings.target_rig):
            msg = "Select a Target MMR rig before importing a .vpd pose."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if not _has_any_filter(settings):
            msg = "Enable at least one body filter before importing a .vpd pose."
            settings.last_status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        if not self.filepath:
            msg = "No .vpd file path provided."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        vpd_scale = _get_addon_vpd_scale(context)
        frame = int(context.scene.frame_current)
        options = _build_retarget_options(
            settings,
            math_retarget,
            insert_keyframes=bool(settings.auto_insert_keyframe),
            frame=frame,
        )

        try:
            stats = vpd_parser.apply_vpd_to_mmr(
                context,
                self.filepath,
                settings.target_rig,
                options,
                scale=vpd_scale,
            )
        except OSError as exc:
            msg = f"Failed to read .vpd file: {exc}"
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        except ValueError as exc:
            msg = f"Invalid .vpd data: {exc}"
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        total = int(stats.get("total", 0))
        filename = os.path.basename(self.filepath)
        msg = f"Applied {filename} ({total} bones) to {settings.target_rig.name}."
        settings.last_status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class POSESNAP_OT_import_vmd(bpy.types.Operator):
    """Import one or more .vmd motion files directly onto Target MMR"""

    bl_idname = "posesnap.import_vmd"
    bl_label = "Import .vmd Motion to MMR..."
    bl_description = "Select one or more .vmd files (e.g. body + face + lip) and merge their bone and facial keyframes onto Target MMR"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="VMD File Path",
        description="Path to a .vmd motion file to import",
        subtype="FILE_PATH",
    )
    directory: bpy.props.StringProperty(
        name="Directory",
        subtype="DIR_PATH",
        options={"HIDDEN"},
    )
    files: bpy.props.CollectionProperty(
        name="VMD Files",
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    filter_glob: bpy.props.StringProperty(
        default="*.vmd",
        options={"HIDDEN"},
    )
    frame_offset: bpy.props.IntProperty(
        name="Frame Offset",
        description="Blender frame added to VMD frame numbers (1 maps VMD frame 0 to frame 1)",
        default=1,
        min=-10000,
        max=100000,
    )
    frame_step: bpy.props.IntProperty(
        name="Frame Step",
        description="Sample every N frames from the VMD timeline",
        default=1,
        min=1,
        max=10,
    )
    set_timeline_range: bpy.props.BoolProperty(
        name="Set Timeline Range",
        description="Update scene Start and End frames to match the imported VMD range",
        default=True,
    )
    bake_all_frames: bpy.props.BoolProperty(
        name="Bake Every Frame",
        description="Keyframe every frame instead of preserving only original VMD keyframes",
        default=False,
    )
    auto_cleanup: bpy.props.BoolProperty(
        name="Automatic Cleanup",
        description="Fix quaternion or Euler flips, IK pole spikes, Bezier overshoot, and redundant keyframes after import",
        default=True,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        if settings is None:
            return False
        return _is_valid_armature(settings.target_rig) and _has_any_filter(settings)

    def _resolve_filepaths(self) -> list[str]:
        if self.files and self.directory:
            paths = [
                os.path.join(self.directory, item.name)
                for item in self.files
                if getattr(item, "name", "")
            ]
            if paths:
                return paths
        if self.filepath:
            return [self.filepath]
        return []

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        scene = getattr(context, "scene", None)
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        if settings is not None:
            self.frame_step = max(1, int(settings.frame_step))
            self.auto_cleanup = bool(getattr(settings, "auto_cleanup_vmd", True))
        if not self._resolve_filepaths():
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}
        return self.execute(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import math_retarget, vmd_parser

        settings = context.scene.posesnap
        if not _is_valid_armature(settings.target_rig):
            msg = "Select a Target MMR rig before importing a .vmd motion."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if not _has_any_filter(settings):
            msg = "Enable at least one body or morph filter before importing a .vmd motion."
            settings.last_status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        filepaths = self._resolve_filepaths()
        if not filepaths:
            msg = "No .vmd file path provided."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        vpd_scale = _get_addon_vpd_scale(context)
        options = _build_retarget_options(
            settings,
            math_retarget,
            insert_keyframes=True,
            frame=None,
        )

        try:
            stats = vmd_parser.apply_vmds_to_mmr(
                context,
                filepaths,
                settings.target_rig,
                options,
                scale=vpd_scale,
                frame_offset=int(self.frame_offset),
                frame_step=int(self.frame_step),
                set_timeline_range=bool(self.set_timeline_range),
                bake_all_frames=bool(self.bake_all_frames),
                auto_cleanup=bool(self.auto_cleanup),
            )
        except OSError as exc:
            msg = f"Failed to read .vmd file: {exc}"
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        except ValueError as exc:
            msg = f"Invalid .vmd data: {exc}"
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        frames_baked = int(stats.get("frames_baked", 0))
        bones_posed = int(stats.get("bones_posed", 0))
        morphs_total = int(stats.get("morphs_driven", 0)) + int(stats.get("morphs_direct", 0))
        morph_note = f", {morphs_total} morphs" if morphs_total > 0 else ""
        if len(filepaths) == 1:
            source_label = os.path.basename(filepaths[0])
        else:
            source_label = f"{len(filepaths)} .vmd files"
        msg = f"Imported {source_label} ({frames_baked} frames, {bones_posed} bones{morph_note}) to {settings.target_rig.name}."
        settings.last_status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class POSESNAP_OT_cleanup_motion(bpy.types.Operator):
    """Automatically clean up quaternion flips, IK spikes, and overshoot on Target MMR"""

    bl_idname = "posesnap.cleanup_motion"
    bl_label = "Automatic Cleanup Motion"
    bl_description = "Fix quaternion or Euler flips, IK pole spikes, Bezier overshoot, and redundant keyframes on Target MMR"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        if settings is None or not _is_valid_armature(settings.target_rig):
            return False
        anim_data = getattr(settings.target_rig, "animation_data", None)
        return getattr(anim_data, "action", None) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import motion_cleanup

        settings = context.scene.posesnap
        if not _is_valid_armature(settings.target_rig):
            msg = "Select a Target MMR rig before running cleanup."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        stats = motion_cleanup.cleanup_rig_action(settings.target_rig)
        if hasattr(context, "view_layer"):
            context.view_layer.update()

        msg = (
            f"Cleaned {settings.target_rig.name}: "
            f"{stats['quat_flips']} quat flips, "
            f"{stats['euler_flips']} Euler flips, "
            f"{stats['ik_spikes']} IK spikes, "
            f"{stats['keys_removed']} redundant keys removed."
        )
        settings.last_status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class POSESNAP_OT_bake_animation(bpy.types.Operator):
    """Bake Source MMD animation across a frame range onto Target MMR"""

    bl_idname = "posesnap.bake_animation"
    bl_label = "Bake Animation Range to MMR"
    bl_description = "Evaluate Source MMD across Start to End frames and keyframe Target MMR controls"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None) if context is not None else None
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        if not _is_valid_rig_pair(settings) or not _has_any_filter(settings):
            return False
        return int(settings.frame_end) >= int(settings.frame_start)

    def execute(self, context: bpy.types.Context) -> set[str]:
        from . import math_retarget

        settings = context.scene.posesnap
        if not _is_valid_rig_pair(settings):
            msg = "Select valid Source MMD and Target MMR armatures before baking."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if int(settings.frame_end) < int(settings.frame_start):
            msg = "End Frame must be greater than or equal to Start Frame."
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if not _has_any_filter(settings):
            msg = "Enable at least one body filter before baking."
            settings.last_status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        options = _build_retarget_options(
            settings,
            math_retarget,
            insert_keyframes=True,
            frame=None,
        )
        stats = math_retarget.bake_armature_range(
            context,
            settings.source_armature,
            settings.target_rig,
            options,
            frame_start=int(settings.frame_start),
            frame_end=int(settings.frame_end),
            frame_step=int(settings.frame_step),
        )
        frames_baked = int(stats.get("frames_baked", 0))
        bones_posed = int(stats.get("bones_posed", 0))
        msg = (
            f"Baked {frames_baked} frames ({bones_posed} bones/frame) "
            f"to {settings.target_rig.name}."
        )
        settings.last_status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class POSESNAP_PT_main(bpy.types.Panel):
    """Main sidebar panel for PoseSnap retargeting"""

    bl_label = "PoseSnap"
    bl_idname = "POSESNAP_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PoseSnap"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = getattr(context, "scene", None)
        settings = getattr(scene, "posesnap", None) if scene is not None else None
        if settings is None:
            layout.label(text="PoseSnap scene settings unavailable.", icon=compat.safe_icon("ERROR"))
            return

        src = settings.source_armature
        tgt = settings.target_rig
        has_src = _is_valid_armature(src)
        has_tgt = _is_valid_armature(tgt)
        pair_valid = _is_valid_rig_pair(settings)
        any_filter = _has_any_filter(settings)

        box1 = layout.box()
        box1.label(text="Armature Pair Selection", icon=compat.safe_icon("OUTLINER_OB_ARMATURE"))
        box1.prop(settings, "source_armature")
        box1.prop(settings, "target_rig")
        box1.operator(
            "posesnap.auto_detect_rigs",
            icon=compat.safe_icon("VIEWZOOM", "ZOOM_ALL"),
        )
        if not has_src and not has_tgt:
            box1.label(
                text="Select Source MMD and Target MMR, or Auto-Detect.",
                icon=compat.safe_icon("INFO"),
            )
        elif not has_tgt:
            box1.label(
                text="Target MMR missing: required for snap and .vpd import.",
                icon=compat.safe_icon("ERROR"),
            )
        elif src == tgt:
            box1.label(
                text="Source and Target cannot be the same armature.",
                icon=compat.safe_icon("ERROR"),
            )
        elif not has_src:
            box1.label(
                text="No Source MMD set: .vpd / .vmd import to Target MMR is ready.",
                icon=compat.safe_icon("INFO"),
            )
        else:
            box1.label(
                text=f"Pair ready: {src.name} -> {tgt.name}",
                icon=compat.safe_icon("CHECKMARK"),
            )

        box2 = layout.box()
        box2.label(text="IK / FK Retargeting Mode", icon=compat.safe_icon("CONSTRAINT_BONE"))

        split_arms = box2.split(factor=0.38)
        split_arms.label(text="Arms Mode")
        row_arms = split_arms.row(align=True)
        row_arms.prop(settings, "arms_mode", expand=True)

        split_legs = box2.split(factor=0.38)
        split_legs.label(text="Legs Mode")
        row_legs = split_legs.row(align=True)
        row_legs.prop(settings, "legs_mode", expand=True)

        # Render finger mode as an explicit read-only status row rather than an interactive control.
        split_fingers = box2.split(factor=0.38)
        split_fingers.label(text="Fingers Mode")
        split_fingers.label(text="FK (Locked)", icon=compat.safe_icon("LOCKED"))

        filters_col = box2.column(align=True)
        filters_grid = filters_col.grid_flow(
            row_major=True,
            columns=2,
            even_columns=True,
            align=True,
        )
        filters_grid.prop(settings, "use_torso")
        filters_grid.prop(settings, "use_head")
        filters_grid.prop(settings, "use_arms")
        filters_grid.prop(settings, "use_fingers")
        filters_grid.prop(settings, "use_legs")
        filters_grid.prop(settings, "use_morphs")

        if not any_filter:
            box2.label(
                text="Enable at least one body filter to transfer poses.",
                icon=compat.safe_icon("ERROR"),
            )

        box2.prop(settings, "hip_scale_ratio")
        cal_col = box2.column(align=True)
        cal_col.enabled = pair_valid
        cal_col.operator(
            "posesnap.auto_calibrate_height",
            icon=compat.safe_icon("DRIVER_DISTANCE", "FILE_REFRESH"),
        )
        if not pair_valid:
            box2.label(
                text="Height calibration requires both Source MMD and Target MMR.",
                icon=compat.safe_icon("INFO"),
            )

        box3 = layout.box()
        box3.label(text="Single Pose & Direct VPD Transfer", icon=compat.safe_icon("POSE_HLT"))
        snap_col = box3.column(align=True)
        snap_col.enabled = pair_valid and any_filter
        snap_col.scale_y = 1.5
        snap_col.operator(
            "posesnap.snap_pose",
            icon=compat.safe_icon("SNAP_ON"),
        )
        box3.prop(settings, "auto_insert_keyframe")
        if not pair_valid:
            box3.label(
                text="Assign both Source MMD and Target MMR to snap live poses.",
                icon=compat.safe_icon("INFO"),
            )

        vpd_col = box3.column(align=True)
        vpd_col.enabled = has_tgt and any_filter
        vpd_col.operator(
            "posesnap.import_vpd",
            icon=compat.safe_icon("IMPORT", "FILE_FOLDER"),
        )
        if not has_tgt:
            box3.label(
                text="Select a Target MMR rig to enable .vpd pose import.",
                icon=compat.safe_icon("INFO"),
            )

        box4 = layout.box()
        box4.label(text="Animation & Direct VMD Transfer", icon=compat.safe_icon("RENDER_ANIMATION"))
        range_row = box4.row(align=True)
        range_row.prop(settings, "frame_start", text="Start")
        range_row.prop(settings, "frame_end", text="End")
        range_row.prop(settings, "frame_step", text="Step")

        range_valid = int(settings.frame_end) >= int(settings.frame_start)
        if not range_valid:
            box4.label(
                text="End Frame must be greater than or equal to Start Frame.",
                icon=compat.safe_icon("ERROR"),
            )

        bake_col = box4.column(align=True)
        bake_col.enabled = pair_valid and range_valid and any_filter
        bake_col.operator(
            "posesnap.bake_animation",
            icon=compat.safe_icon("KEYFRAME_HLT"),
        )

        vmd_col = box4.column(align=True)
        vmd_col.enabled = has_tgt and any_filter
        vmd_col.prop(settings, "auto_cleanup_vmd")
        vmd_col.operator(
            "posesnap.import_vmd",
            icon=compat.safe_icon("IMPORT", "FILE_FOLDER"),
        )
        has_action = has_tgt and getattr(getattr(tgt, "animation_data", None), "action", None) is not None
        cleanup_col = box4.column(align=True)
        cleanup_col.enabled = has_action
        cleanup_col.operator(
            "posesnap.cleanup_motion",
            icon=compat.safe_icon("BRUSH_DATA", "MOD_SMOOTH"),
        )
        if not has_tgt:
            box4.label(
                text="Select a Target MMR rig to enable .vmd motion import.",
                icon=compat.safe_icon("INFO"),
            )

        if settings.last_status:
            status_box = layout.box()
            status_box.label(
                text=settings.last_status,
                icon=compat.safe_icon("INFO"),
            )


CLASSES = (
    POSESNAP_OT_auto_detect,
    POSESNAP_OT_calibrate_height,
    POSESNAP_OT_snap_pose,
    POSESNAP_OT_import_vpd,
    POSESNAP_OT_import_vmd,
    POSESNAP_OT_cleanup_motion,
    POSESNAP_OT_bake_animation,
    POSESNAP_PT_main,
)
