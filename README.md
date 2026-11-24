# PoseSnap

PoseSnap is a Blender add-on that retargets MikuMikuDance (MMD) poses and animations from standard MMD armatures or `.vpd` files onto MikuMikuRig (MMR) control rigs.

It targets Goo Engine 4.4.3, Blender 4.5 LTS, and Blender 5.2 LTS+, using world-space rest-pose compensation so bone roll and rest orientation differences between MMD and Rigify/MMR rigs do not twist limbs or flip joints.

## Features

- Independent IK/FK switching for arms (`upper_arm_parent.L/R`) and legs (`thigh_parent.L/R`), setting Rigify/MMR `IK_FK` and `pole_vector` properties automatically.
- Automatic elbow and knee IK pole target placement (`upper_arm_ik_target.L/R`, `thigh_ik_target.L/R`) projected from the 3-point limb plane.
- Fingers locked to per-segment FK (`thumb.01..03` through `f_pinky.01..03`), setting finger `FK_IK = 0.0` instead of driving `*_master` or `*_ik` sliders.
- Twist-bone accumulation (`腕捩` and `手捩`) for FK arm retargeting and optional upper-chest/shoulder variation handling (`上半身3`, `肩P`, `肩C`, `腰キャンセル`, `親指０`).
- Direct `.vpd` (Vocaloid Pose Data) parser and importer supporting Shift-JIS, CP932, and UTF-8 files without creating temporary armatures in the Outliner.
- Frame-range animation baker for `.vmd` motions imported onto an MMD armature, compatible with both legacy F-Curves (4.4) and slotted actions (4.5/5.0+).
- Optional N-Panel docking inside the MikuMikuRig (`MMR`) sidebar tab when MMR is installed.

## Installation

### Blender 4.5 LTS and 5.2+ (Extension format)

1. Compress the `posesnap/` directory into a `.zip` archive (containing `blender_manifest.toml` at its root inside the zip) or use **Edit > Preferences > Get Extensions > Install from Disk** in Blender 4.2+.
2. Enable **PoseSnap** in the Extensions / Add-ons list.

### Goo Engine 4.4.3 / Legacy Add-on Install

1. Open **Edit > Preferences > Add-ons > Install...** and select the `posesnap.zip` archive (or symlink/copy the `posesnap/` folder into your Blender `scripts/addons/` directory).
2. Enable **Animation: PoseSnap**.

## Add-on Preferences

In **Edit > Preferences > Add-ons > PoseSnap**:

- **Dock in MMR Tab**: When enabled (default), PoseSnap places its panel in the `MMR` sidebar tab if MikuMikuRig panels are registered, falling back to a dedicated `PoseSnap` tab otherwise.
- **VPD Scale**: World unit scale applied when parsing `.vpd` translations (default `0.08`, matching standard MMD to Blender scaling).

## Usage

Open the 3D Viewport sidebar (`N`) and select the **MMR** tab (or **PoseSnap** tab if MikuMikuRig is not active).

### 1. Armature Pair Selection

- Pick the **Source MMD** armature (imported via `mmd_tools`) and the **Target MMR** control rig (`RIG-*`).
- Click **Auto-Detect Rigs** to score armatures in the active scene and fill both fields automatically.

### 2. IK / FK Mode and Body Filters

- **Arms Mode** (`IK` / `FK`): Choose whether MMD wrist targets drive `hand_ik.L/R` plus `upper_arm_ik_target.L/R`, or MMD arm rotations drive `upper_arm_fk.L/R`, `forearm_fk.L/R`, and `hand_fk.L/R`.
- **Legs Mode** (`IK` / `FK`): Choose whether MMD ankles drive `foot_ik.L/R` plus `thigh_ik_target.L/R`, or MMD leg rotations drive `thigh_fk.L/R`, `shin_fk.L/R`, `foot_fk.L/R`, and `toe.L/R`.
- **Fingers Mode (FK Locked)**: Finger bones always retarget to individual FK controls (`thumb.01..03`, `f_index.01..03`, `f_middle.01..03`, `f_ring.01..03`, `f_pinky.01..03`).
- **Body Filters**: Toggle Torso & Spine, Head & Eyes, Arms & Hands, Fingers (FK), and Legs & Feet independently.
- **Hip Scale Ratio / Auto-Calibrate Height**: Scales root/hip translation by the rest-pose height ratio between the MMD source center bone and the MMR `torso` bone so feet stay grounded.

### 3. Single Pose & Direct `.vpd` Import

- **Snap Pose to MMR (Current Frame)**: Transfers the current pose from the Source MMD armature to the Target MMR rig. Enable **Auto Insert Keyframe** to key posed bones and IK/FK switches on the current frame.
- **Import .vpd Pose to MMR...**: Opens a `.vpd` file browser and applies the pose directly to the Target MMR rig using the selected IK/FK modes and body filters. A source MMD armature is not required for `.vpd` import.

### 4. Animation Range Baker (VMD Workflow)

1. Import your `.vmd` motion onto a standard MMD armature using `mmd_tools`.
2. Select that MMD armature as **Source MMD** and your MMR control rig as **Target MMR**.
3. Set **Start**, **End**, and **Step** (`1` for full frame rate, `2` for on-twos).
4. Click **Bake Animation Range to MMR** to evaluate each stepped frame and record keyframes into the MMR rig's active action.

## License

PoseSnap is licensed under the GNU General Public License v3.0 or later (`GPL-3.0-or-later`). See [LICENSE](LICENSE).
