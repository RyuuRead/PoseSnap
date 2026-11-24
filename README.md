<div align="center">

# PoseSnap

**MMD (.vpd / .vmd) Pose and Animation Retargeter for MikuMikuRig**

</div>

## Overview

PoseSnap (`v1.0.0`) transfers MikuMikuDance poses and animations from standard MMD armatures or `.vpd` files onto MikuMikuRig (`RIG-*`) control rigs. It computes world-space rest-pose offsets so bone roll and rest orientation differences do not twist limbs, and docks into the `MMR` sidebar tab when MikuMikuRig is active.

---

## Features

| Feature | Description |
| --- | --- |
| **Arm & Leg IK / FK Modes** | Switches `upper_arm_parent.L/R` and `thigh_parent.L/R` (`IK_FK` and `pole_vector`) independently between IK targets and FK limb chains. |
| **Automatic IK Pole Targets** | Projects `upper_arm_ik_target.L/R` and `thigh_ik_target.L/R` from the 3-point limb plane with collinear fallback. |
| **Fingers Always FK** | Retargets all 30 finger segments (`thumb.01..03` through `f_pinky.01..03`) to per-bone FK controls and sets `FK_IK = 0.0`, never driving `*_master` or `*_ik`. |
| **Twist & Rig Variations** | Folds `腕捩` / `手捩` twist rotations into FK arms and handles `上半身3`, `上半身1`, `肩P`, `肩C`, `腰キャンセル`, and `親指０` fallback. |
| **Direct `.vpd` Importer** | Parses Shift-JIS, CP932, and UTF-8 `.vpd` files and solves a virtual MMD skeleton in memory without creating helper objects. |
| **Animation Range Baker** | Steps through `.vmd` motions on an MMD armature (`Step 1` or `2`) and records keyframes across legacy F-Curves and Blender 5.0+ slotted actions. |

---

## Compatibility & Installation

Supported targets: **Goo Engine 4.4.3**, **Blender 4.5 LTS**, and **Blender 5.2+**.

- **Blender 4.5 LTS and 5.2+:** Package `posesnap/` as a `.zip` with `blender_manifest.toml` at the root, then install via **Edit > Preferences > Get Extensions > Install from Disk**.
- **Goo Engine 4.4.3:** Install the `.zip` or copy `posesnap/` into `scripts/addons/`, then enable **Animation: PoseSnap** under **Edit > Preferences > Add-ons**.

---

## Usage

1. Open the 3D Viewport sidebar (`N`) and select the **MMR** tab (or **PoseSnap** if MikuMikuRig is not active).
2. Assign **Source MMD** and **Target MMR** (or click **Auto-Detect Rigs**), then click **Auto-Calibrate Height** if character proportions differ.
3. Choose **IK** or **FK** for **Arms Mode** and **Legs Mode** (fingers stay locked to **FK**), and toggle body filters as needed.
4. Click **Snap Pose to MMR (Current Frame)**, **Import .vpd Pose to MMR...** (no source rig required), or **Bake Animation Range to MMR** for `.vmd` animations.

---

## Credits & License

- Built as a companion add-on for [MikuMikuRig](https://github.com/XiaoFFGel/MikuMikuRig).
- Verify headless tests with: `blender --background --factory-startup --python tests/test_posesnap.py`
- Licensed under `GPL-3.0-or-later` ([LICENSE](LICENSE)).
