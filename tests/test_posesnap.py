# SPDX-License-Identifier: GPL-3.0-or-later
"""PoseSnap test suite.

Run inside Blender, from any working directory:
    blender --background --python tests/test_posesnap.py
    blender --factory-startup --background --python tests/test_posesnap.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# A copy of posesnap installed as a user add-on would shadow the repo version.
for _mod_name in [m for m in sys.modules if m == "posesnap" or m.startswith("posesnap.")]:
    _mod_file = getattr(sys.modules[_mod_name], "__file__", "") or ""
    if not _mod_file.startswith(str(REPO_ROOT)):
        del sys.modules[_mod_name]

HEAVY_MODULES = ("posesnap.bone_map", "posesnap.math_retarget", "posesnap.vpd_parser")

# Startup is timed before anything else imports the heavy submodules.
try:
    _t0 = time.perf_counter()
    import posesnap  # noqa: E402
    _t1 = time.perf_counter()
    posesnap.register()
    _t2 = time.perf_counter()
except Exception:  # noqa: BLE001
    # Blender exits with code 0 on uncaught script errors, so fail explicitly.
    import traceback

    traceback.print_exc()
    sys.exit(1)
STARTUP = {
    "import_ms": (_t1 - _t0) * 1000.0,
    "register_ms": (_t2 - _t1) * 1000.0,
    "total_ms": (_t2 - _t0) * 1000.0,
    "heavy_loaded": [m for m in HEAVY_MODULES if m in sys.modules],
}

import math  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

import bpy  # noqa: E402
from mathutils import Matrix, Quaternion, Vector  # noqa: E402

from posesnap import bone_map, compat, math_retarget, vpd_parser  # noqa: E402

SIDES = ("L", "R")
FINGERS = ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")
MMD_FINGERS = ("親指", "人指", "中指", "薬指", "小指")
ANGLE_TOL = math.radians(1.0)
POS_TOL = 1e-3

HIERARCHY_FILES = {
    "hutao": "胡桃_arm.txt",
    "keqing": "刻晴_arm.txt",
    # The workspace reference file name contains a literal em dash.
    "firefly": "星穹铁道—流萤_春日手信_arm.txt",
    "rig": "RIG-胡桃_arm.txt",
}


def parse_hierarchy_text(text):
    """Return an ordered {bone: parent} dict from two-space indented dumps."""
    parents = {}
    stack = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        depth = (len(line) - len(line.lstrip(" "))) // 2
        name = line.strip()
        del stack[depth:]
        parents[name] = stack[-1] if stack else None
        stack.append(name)
    return parents


def load_hierarchy(key):
    path = REPO_ROOT / HIERARCHY_FILES[key]
    if path.is_file():
        return parse_hierarchy_text(path.read_text(encoding="utf-8")), True
    return parse_hierarchy_text(FALLBACK_HIERARCHIES[key]()), False


def _mock_mmd_text(upper3, shoulder_p, cancel, thumb0, index_name):
    lines = ["全ての親", "  センター", "    グルーブ", "      腰"]
    leg_parent_depth = 4
    lines.append("        下半身")
    for side in SIDES:
        depth = leg_parent_depth + 1
        if cancel:
            lines.append("  " * depth + f"腰キャンセル.{side}")
            depth += 1
        lines += [
            "  " * depth + f"足.{side}",
            "  " * (depth + 1) + f"ひざ.{side}",
            "  " * (depth + 2) + f"足首.{side}",
            "  " * (depth + 3) + f"つま先.{side}",
            "  " * (depth + 1) + f"_dummy_足D.{side}",
            "  " * (depth + 1) + f"_shadow_ひざD.{side}",
        ]
    lines.append("        上半身")
    depth = 5
    if upper3:
        lines.append("          上半身3")
        depth = 6
    elif upper3 is None:
        lines.append("          上半身1")
        depth = 6
    lines.append("  " * depth + "上半身2")
    lines.append("  " * (depth + 1) + "首")
    lines.append("  " * (depth + 2) + "頭")
    for side in SIDES:
        d = depth + 1
        if shoulder_p:
            lines.append("  " * d + f"肩P.{side}")
            d += 1
        lines.append("  " * d + f"肩.{side}")
        d += 1
        if shoulder_p:
            lines.append("  " * d + f"肩C.{side}")
            d += 1
        lines += [
            "  " * d + f"腕.{side}",
            "  " * (d + 1) + f"腕捩.{side}",
            "  " * (d + 2) + f"ひじ.{side}",
            "  " * (d + 3) + f"手捩.{side}",
            "  " * (d + 4) + f"手首.{side}",
            "  " * (d + 2) + f"_dummy_腕捩1.{side}",
            "  " * (d + 2) + f"_shadow_腕捩1.{side}",
        ]
        hand = d + 5
        thumb = ["親指０", "親指１", "親指２"] if thumb0 else ["親指１", "親指２"]
        thumb.append("親指先")
        for i, nm in enumerate(thumb):
            lines.append("  " * (hand + i) + f"{nm}.{side}")
        for base in (index_name, "中指", "薬指", "小指"):
            for i, suffix in enumerate(("１", "２", "３", "先")):
                lines.append("  " * (hand + i) + f"{base}{suffix}.{side}")
    return "\n".join(lines)


FALLBACK_HIERARCHIES = {
    "hutao": lambda: _mock_mmd_text(True, True, False, True, "人指"),
    "keqing": lambda: _mock_mmd_text(False, False, False, True, "人指"),
    "firefly": lambda: _mock_mmd_text(None, True, True, True, "人差指"),
}


def _mock_rig_text():
    out = ["root", "  torso", "    hips", "      ORG-spine", "        spine_fk"]
    out += ["    chest", "      spine_fk.001", "        spine_fk.002", "          spine_fk.003",
            "            ORG-spine.003", "            neck", "              head"]
    for s in SIDES:
        out += [f"            shoulder.{s}", f"              ORG-shoulder.{s}",
                f"              upper_arm_parent.{s}", f"                upper_arm_fk.{s}",
                f"                  forearm_fk.{s}", f"                    hand_fk.{s}",
                f"              ORG-upper_arm.{s}", f"                ORG-forearm.{s}",
                f"                  ORG-hand.{s}"]
        for f in FINGERS:
            out += [f"                    {f}.01_master.{s}", f"                    {f}.01.{s}",
                    f"                      {f}.02.{s}", f"                        {f}.03.{s}",
                    f"                    ORG-{f}.01.{s}"]
        out += [f"      thigh_parent.{s}", f"        thigh_fk.{s}", f"          shin_fk.{s}",
                f"            foot_fk.{s}", f"              toe.{s}", f"      ORG-thigh.{s}",
                f"        ORG-shin.{s}", f"          ORG-foot.{s}", f"            ORG-toe.{s}"]
    for s in SIDES:
        out += [f"  hand_ik.{s}", f"  upper_arm_ik_target.{s}", f"  foot_ik.{s}",
                f"  thigh_ik_target.{s}"]
        out += [f"  {f}.01_ik.{s}" for f in FINGERS]
    return "\n".join(out)


FALLBACK_HIERARCHIES["rig"] = _mock_rig_text


def _v(*xyz):
    return Vector(xyz)


def _limb_points(s):
    """Shared rest layout: 45-degree arms, straight legs, character facing -Y."""
    shoulder = _v(s * 0.03, 0.0, 1.38)
    arm_a = _v(s * 0.15, 0.0, 1.40)
    elbow = _v(s * 0.35, 0.0, 1.20)
    wrist = _v(s * 0.55, 0.0, 1.00)
    hand_tip = _v(s * 0.60, 0.0, 0.95)
    hip = _v(s * 0.10, 0.0, 0.95)
    knee = _v(s * 0.10, 0.0, 0.50)
    ankle = _v(s * 0.10, 0.0, 0.10)
    toe = _v(s * 0.10, -0.10, 0.02)
    toe_tip = _v(s * 0.10, -0.15, 0.00)
    return locals()


def _finger_segments(s, finger_index):
    p = _limb_points(s)
    direction = (p["hand_tip"] - p["wrist"]).normalized()
    base = p["hand_tip"] + _v(0.0, -0.03 + 0.015 * finger_index, 0.0)
    pts = [base + direction * (0.02 * i) for i in range(4)]
    return [(pts[i], pts[i + 1]) for i in range(3)]


def rig_rest_table(scale=1.0):
    t = {
        "root": (_v(0, 0, 0), _v(0, 0.3, 0)),
        "torso": (_v(0, 0, 1.0), _v(0, 0.2, 1.0)),
        "hips": (_v(0, 0, 1.0), _v(0, 0, 0.9)),
        "chest": (_v(0, 0, 1.15), _v(0, 0, 1.4)),
        "spine": (_v(0, 0, 0.9), _v(0, 0, 1.0)),
        "spine_fk": (_v(0, 0, 0.9), _v(0, 0, 1.0)),
        "spine.001": (_v(0, 0, 1.0), _v(0, 0, 1.15)),
        "spine_fk.001": (_v(0, 0, 1.0), _v(0, 0, 1.15)),
        "spine.002": (_v(0, 0, 1.15), _v(0, 0, 1.25)),
        "spine_fk.002": (_v(0, 0, 1.15), _v(0, 0, 1.25)),
        "spine.003": (_v(0, 0, 1.25), _v(0, 0, 1.4)),
        "spine_fk.003": (_v(0, 0, 1.25), _v(0, 0, 1.4)),
        "spine.004": (_v(0, 0, 1.4), _v(0, 0, 1.5)),
        "neck": (_v(0, 0, 1.4), _v(0, 0, 1.5)),
        "spine.006": (_v(0, 0, 1.5), _v(0, 0, 1.7)),
        "head": (_v(0, 0, 1.5), _v(0, 0, 1.7)),
    }
    for side, s in (("L", 1.0), ("R", -1.0)):
        p = _limb_points(s)
        limb = {
            "shoulder": (p["shoulder"], p["arm_a"]),
            "upper_arm": (p["arm_a"], p["elbow"]),
            "forearm": (p["elbow"], p["wrist"]),
            "hand": (p["wrist"], p["hand_tip"]),
            "upper_arm_ik_target": (p["elbow"] + _v(0, 0.3, 0), p["elbow"] + _v(0, 0.35, 0)),
            "thigh": (p["hip"], p["knee"]),
            "shin": (p["knee"], p["ankle"]),
            "foot": (p["ankle"], p["toe"]),
            "toe": (p["toe"], p["toe_tip"]),
            "thigh_ik_target": (p["knee"] + _v(0, -0.4, 0), p["knee"] + _v(0, -0.45, 0)),
        }
        for i, f in enumerate(FINGERS):
            for j, seg in enumerate(_finger_segments(s, i)):
                limb[f"{f}.0{j + 1}"] = seg
        for k, val in limb.items():
            t[f"{k}.{side}"] = val
    return {k: (h * scale, tl * scale) for k, (h, tl) in t.items()}


ALIASES = {
    "upper_arm_fk": "upper_arm", "upper_arm_ik": "upper_arm", "upper_arm_parent": "upper_arm",
    "forearm_fk": "forearm", "hand_fk": "hand", "hand_ik": "hand",
    "thigh_fk": "thigh", "thigh_ik": "thigh", "thigh_parent": "thigh",
    "shin_fk": "shin", "foot_fk": "foot", "foot_ik": "foot",
}


def rig_bone_rest(name, table):
    if name in table:
        return table[name]
    core = re.sub(r"^(ORG-|DEF-|MCH-|VIS_)", "", name)
    core = re.sub(r"_parent$", "", core)
    side = None
    m = re.search(r"\.([LR])(?=$|[._])", core)
    if m:
        side = m.group(1)
        core = core[: m.start()]
    core = ALIASES.get(core, core)
    for candidate in (core, re.sub(r"_(master|ik|fk|tweak)$", "", core)):
        key = f"{candidate}.{side}" if side else candidate
        if key in table:
            return table[key]
    return None


def build_armature(name, bones):
    """bones: list of (name, head, tail, parent_name or None)."""
    arm_data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(obj)
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_data.edit_bones
    for bname, head, tail, _parent in bones:
        eb = ebones.new(bname)
        eb.head = head
        eb.tail = tail
        eb.roll = 0.0
    for bname, _head, _tail, parent in bones:
        if parent:
            ebones[bname].parent = ebones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return obj


def build_mmd_armature(name="SyntheticMMD"):
    bones = [
        ("全ての親", _v(0, 0, 0), _v(0, 0, 0.2), None),
        ("センター", _v(0, 0, 0.8), _v(0, 0, 0.7), "全ての親"),
        ("グルーブ", _v(0, 0, 0.8), _v(0, 0, 0.9), "センター"),
        ("腰", _v(0, 0, 0.95), _v(0, 0, 1.0), "グルーブ"),
        ("上半身", _v(0, 0, 1.0), _v(0, 0, 1.15), "腰"),
        ("上半身3", _v(0, 0, 1.15), _v(0, 0, 1.25), "上半身"),
        ("上半身2", _v(0, 0, 1.25), _v(0, 0, 1.4), "上半身3"),
        ("首", _v(0, 0, 1.4), _v(0, 0, 1.5), "上半身2"),
        ("頭", _v(0, 0, 1.5), _v(0, 0, 1.7), "首"),
        ("下半身", _v(0, 0, 1.0), _v(0, 0, 0.9), "腰"),
    ]
    for side, s in (("L", 1.0), ("R", -1.0)):
        p = _limb_points(s)
        mid_ua = (p["arm_a"] + p["elbow"]) / 2
        mid_fa = (p["elbow"] + p["wrist"]) / 2
        bones += [
            (f"肩P.{side}", p["shoulder"], p["shoulder"] + _v(0, 0, 0.04), "上半身2"),
            (f"肩.{side}", p["shoulder"], p["arm_a"], f"肩P.{side}"),
            (f"腕.{side}", p["arm_a"], p["elbow"], f"肩.{side}"),
            (f"腕捩.{side}", mid_ua, p["elbow"], f"腕.{side}"),
            (f"_dummy_腕捩1.{side}", mid_ua, p["elbow"], f"腕.{side}"),
            (f"_shadow_腕捩1.{side}", mid_ua, p["elbow"], f"腕.{side}"),
            (f"ひじ.{side}", p["elbow"], p["wrist"], f"腕捩.{side}"),
            (f"手捩.{side}", mid_fa, p["wrist"], f"ひじ.{side}"),
            (f"手首.{side}", p["wrist"], p["hand_tip"], f"手捩.{side}"),
            (f"足.{side}", p["hip"], p["knee"], "下半身"),
            (f"_dummy_足D.{side}", p["hip"], p["knee"], "下半身"),
            (f"_shadow_足D.{side}", p["hip"], p["knee"], "下半身"),
            (f"ひざ.{side}", p["knee"], p["ankle"], f"足.{side}"),
            (f"足首.{side}", p["ankle"], p["toe"], f"ひざ.{side}"),
            (f"つま先.{side}", p["toe"], p["toe_tip"], f"足首.{side}"),
        ]
        for i, mmd_f in enumerate(MMD_FINGERS):
            names = ["親指０", "親指１", "親指２"] if i == 0 else [f"{mmd_f}{n}" for n in "１２３"]
            parent = f"手首.{side}"
            for nm, (h, t) in zip(names, _finger_segments(s, i)):
                bones.append((f"{nm}.{side}", h, t, parent))
                parent = f"{nm}.{side}"
    return build_armature(name, bones)


def build_mmr_rig(name="SyntheticMMR", scale=1.0):
    parents, _from_file = load_hierarchy("rig")
    table = rig_rest_table(scale)
    root_rest = table["root"]
    rests = {}
    bones = []
    for bname, parent in parents.items():
        rest = rig_bone_rest(bname, table)
        if rest is None:
            rest = rests.get(parent, root_rest)
        rests[bname] = rest
        bones.append((bname, rest[0], rest[1], parent))
    obj = build_armature(name, bones)
    for side in SIDES:
        for switch in (f"upper_arm_parent.{side}", f"thigh_parent.{side}"):
            pb = obj.pose.bones[switch]
            pb["IK_FK"] = 0.5
            pb["pole_vector"] = False
        for f in FINGERS:
            obj.pose.bones[f"{f}.01_ik.{side}"]["FK_IK"] = 1.0
    return obj


def world_matrix(obj, bone_name):
    return obj.matrix_world @ obj.pose.bones[bone_name].matrix


def rot_angle(m_a, m_b):
    qa = m_a.to_3x3().normalized().to_quaternion()
    qb = m_b.to_3x3().normalized().to_quaternion()
    return qa.rotation_difference(qb).angle


def reset_pose(obj):
    for pb in obj.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)


def pose_source(src):
    rot = Quaternion
    poses = {
        "センター": (rot(), _v(0.0, 0.05, -0.05)),
        "上半身": (rot((1, 0, 0), math.radians(10)), None),
        "頭": (rot((0, 0, 1), math.radians(15)), None),
    }
    for side, s in (("L", 1.0), ("R", -1.0)):
        poses.update({
            f"腕.{side}": (rot((1, 0, 0), math.radians(30)) @ rot((0, 0, 1), math.radians(10 * s)), None),
            f"腕捩.{side}": (rot((0, 1, 0), math.radians(20 * s)), None),
            f"ひじ.{side}": (rot((1, 0, 0), math.radians(40)), None),
            f"手捩.{side}": (rot((0, 1, 0), math.radians(15 * s)), None),
            f"手首.{side}": (rot((0, 0, 1), math.radians(12)), None),
            f"足.{side}": (rot((1, 0, 0), math.radians(-30)), None),
            f"ひざ.{side}": (rot((1, 0, 0), math.radians(50)), None),
            f"足首.{side}": (rot((1, 0, 0), math.radians(-15)), None),
            f"つま先.{side}": (rot((1, 0, 0), math.radians(10)), None),
        })
        for i, mmd_f in enumerate(MMD_FINGERS):
            names = ["親指０", "親指１", "親指２"] if i == 0 else [f"{mmd_f}{n}" for n in "１２３"]
            for j, nm in enumerate(names):
                poses[f"{nm}.{side}"] = (rot((1, 0, 0), math.radians(15 + 5 * j + 2 * i)), None)
    for bname, (q, loc) in poses.items():
        pb = src.pose.bones[bname]
        pb.rotation_quaternion = q
        if loc is not None:
            pb.location = loc
    bpy.context.view_layer.update()


def finger_pairs(side):
    pairs = []
    for i, f in enumerate(FINGERS):
        names = ["親指０", "親指１", "親指２"] if i == 0 else [f"{MMD_FINGERS[i]}{n}" for n in "１２３"]
        for j, nm in enumerate(names):
            pairs.append((f"{nm}.{side}", f"{f}.0{j + 1}.{side}"))
    return pairs


def collect_pairs(obj, out=None):
    """Flatten any nested mapping report into a set of (source, target) string pairs."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, str):
                out.add((k, v))
                out.add((v, k))
            else:
                collect_pairs(v, out)
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) == 2 and all(isinstance(x, str) for x in obj):
            out.add(tuple(obj))
            out.add((obj[1], obj[0]))
        else:
            for item in obj:
                collect_pairs(item, out)
    return out


def collect_strings(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_strings(k, out)
            collect_strings(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            collect_strings(item, out)
    return out


def clear_scene_objects():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for arm in list(bpy.data.armatures):
        if arm.users == 0:
            bpy.data.armatures.remove(arm)
    for act in list(bpy.data.actions):
        if act.users == 0:
            bpy.data.actions.remove(act)


def set_settings(src, rig, arms="IK", legs="IK"):
    st = bpy.context.scene.posesnap
    st.source_armature = src
    st.target_rig = rig
    st.arms_mode = arms
    st.legs_mode = legs
    st.use_torso = st.use_head = st.use_arms = st.use_fingers = st.use_legs = True
    st.hip_scale_ratio = 1.0
    st.auto_insert_keyframe = False
    return st


class TestStartupAndRegistration(unittest.TestCase):
    def test_lazy_startup(self):
        print(f"\n[startup] import={STARTUP['import_ms']:.3f} ms register={STARTUP['register_ms']:.3f} ms "
              f"total={STARTUP['total_ms']:.3f} ms (target < 5 ms, CI limit < 50 ms)")
        self.assertEqual(STARTUP["heavy_loaded"], [], "heavy modules loaded at startup")
        self.assertLess(STARTUP["total_ms"], 50.0)

    def test_registration(self):
        self.assertTrue(hasattr(bpy.types, "POSESNAP_PT_main"))
        for op in ("POSESNAP_OT_auto_detect_rigs", "POSESNAP_OT_auto_calibrate_height",
                   "POSESNAP_OT_snap_pose", "POSESNAP_OT_import_vpd",
                   "POSESNAP_OT_import_vmd", "POSESNAP_OT_bake_animation"):
            self.assertTrue(hasattr(bpy.types, op), op)
        for idname in ("auto_detect_rigs", "auto_calibrate_height", "snap_pose", "import_vpd",
                       "import_vmd", "bake_animation"):
            getattr(bpy.ops.posesnap, idname).get_rna_type()
        self.assertTrue(hasattr(bpy.types.Scene, "posesnap"))
        self.assertEqual(bpy.context.scene.posesnap.arms_mode, "IK")
        self.assertEqual(bpy.context.scene.posesnap.legs_mode, "IK")

    def test_safe_icon_and_category(self):
        self.assertEqual(compat.safe_icon("ARMATURE_DATA"), "ARMATURE_DATA")
        self.assertEqual(compat.safe_icon("NOT_A_REAL_ICON_XYZ"), "NONE")
        self.assertEqual(compat.safe_icon("NOT_A_REAL_ICON_XYZ", "ERROR"), "ERROR")
        self.assertEqual(compat.resolve_panel_category(False), "PoseSnap")
        cat = compat.resolve_panel_category(True)
        self.assertIsInstance(cat, str)
        self.assertTrue(cat)
        mmr = compat.detect_mmr_category()
        self.assertEqual(cat, mmr if mmr else "PoseSnap")


class TestHierarchyMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig_names = set(load_hierarchy("rig")[0])
        cls.models = {k: set(load_hierarchy(k)[0]) for k in ("hutao", "keqing", "firefly")}
        no_thumb0 = parse_hierarchy_text(_mock_mmd_text(True, True, False, False, "人指"))
        cls.models["no_thumb0"] = set(no_thumb0)

    def _check_common(self, key):
        src = self.models[key]
        report = bone_map.inspect_hierarchy_mapping(src, self.rig_names)
        self.assertIsInstance(report, dict)
        strings = collect_strings(report)
        for s in strings:
            self.assertFalse(s.startswith(("_dummy_", "_shadow_")), f"{key}: matched helper bone {s}")
        for side in SIDES:
            pairs = bone_map.resolve_finger_fk_pairs(src, self.rig_names, side)
            self.assertGreaterEqual(len(pairs), 15, f"{key} {side}: {pairs}")
            for s_name, t_name in pairs:
                self.assertIn(s_name, src)
                self.assertIn(t_name, self.rig_names)
                self.assertFalse(t_name.endswith(f"_master.{side}") or t_name.endswith(f"_ik.{side}"))
                self.assertNotIn("_master", t_name)
                self.assertNotIn("_ik", t_name)
                self.assertFalse(s_name.startswith(("_dummy_", "_shadow_")))
            targets = {t for _s, t in pairs}
            for f in FINGERS:
                for j in (1, 2, 3):
                    self.assertIn(f"{f}.0{j}.{side}", targets, f"{key}: missing {f}.0{j}.{side}")
        return report, src

    def _thumb_map(self, src, side):
        return {t: s for s, t in bone_map.resolve_finger_fk_pairs(src, self.rig_names, side)}

    def test_hutao(self):
        report, src = self._check_common("hutao")
        self.assertIn("上半身3", src)
        self.assertIn("肩P.L", src)
        self.assertIn("肩C.L", src)
        pairs = collect_pairs(report)
        self.assertIn(("上半身3", "spine_fk.002"), pairs)
        self.assertEqual(bone_map.resolve_source_bone(src, ("上半身3", "上半身1")), "上半身3")
        for side in SIDES:
            tm = self._thumb_map(src, side)
            self.assertEqual([tm[f"thumb.0{j}.{side}"] for j in (1, 2, 3)],
                             [f"親指０.{side}", f"親指１.{side}", f"親指２.{side}"])
            self.assertEqual(tm[f"f_index.01.{side}"], f"人指１.{side}")

    def test_keqing(self):
        report, src = self._check_common("keqing")
        self.assertNotIn("上半身3", src)
        self.assertNotIn("肩P.L", src)
        self.assertNotIn("肩C.L", src)
        self.assertNotIn("上半身3", collect_strings(report))
        for side in SIDES:
            tm = self._thumb_map(src, side)
            self.assertEqual(tm[f"thumb.01.{side}"], f"親指０.{side}")
            self.assertEqual(tm[f"thumb.03.{side}"], f"親指２.{side}")

    def test_firefly(self):
        report, src = self._check_common("firefly")
        self.assertIn("上半身1", src)
        self.assertIn("腰キャンセル.L", src)
        self.assertIn("人差指先.L", src)
        self.assertEqual(bone_map.resolve_source_bone(src, ("上半身3", "上半身1")), "上半身1")
        self.assertIn(("上半身1", "spine_fk.002"), collect_pairs(report))
        for side in SIDES:
            tm = self._thumb_map(src, side)
            self.assertEqual(tm[f"thumb.01.{side}"], f"親指０.{side}")
            self.assertIn(tm[f"f_index.01.{side}"], (f"人指１.{side}", f"人差指１.{side}"))
        # Verify full 人差指１..３ naming variant also resolves cleanly.
        alt_index_src = set(parse_hierarchy_text(_mock_mmd_text(None, True, True, True, "人差指")))
        for side in SIDES:
            alt_map = self._thumb_map(alt_index_src, side)
            self.assertEqual(alt_map[f"f_index.01.{side}"], f"人差指１.{side}")

    def test_without_thumb0(self):
        report, src = self._check_common("no_thumb0")
        for side in SIDES:
            tm = self._thumb_map(src, side)
            self.assertEqual([tm[f"thumb.0{j}.{side}"] for j in (1, 2, 3)],
                             [f"親指１.{side}", f"親指２.{side}", f"親指先.{side}"])

    def test_name_helpers(self):
        self.assertEqual(bone_map.normalize_mmd_name(" 左腕 "), "腕.L")
        self.assertEqual(bone_map.normalize_mmd_name("右親指２"), "親指2.R")
        self.assertIn("人差指1.L", bone_map.mmd_name_variants("人指１.L"))
        self.assertIsNone(bone_map.resolve_source_bone({"_dummy_腕.L", "_shadow_腕.L"}, ("腕.L",)))
        self.assertEqual(bone_map.resolve_source_bone({"腕.L"}, ("左腕",)), "腕.L")
        self.assertEqual(bone_map.ARM_SWITCH_BONES, {"L": "upper_arm_parent.L", "R": "upper_arm_parent.R"})
        self.assertEqual(bone_map.LEG_SWITCH_BONES, {"L": "thigh_parent.L", "R": "thigh_parent.R"})
        self.assertEqual(tuple(bone_map.FINGER_NAMES), FINGERS)
        self.assertEqual(bone_map.FINGER_IK_SWITCH_BONES[("thumb", "L")], "thumb.01_ik.L")
        self.assertEqual(bone_map.FINGER_MASTER_BONES[("f_pinky", "R")], "f_pinky.01_master.R")


class SceneTestCase(unittest.TestCase):
    def setUp(self):
        clear_scene_objects()
        bpy.context.scene.frame_set(1)
        self.src = build_mmd_armature()
        self.rig = build_mmr_rig()
        bpy.context.view_layer.update()

    def tearDown(self):
        clear_scene_objects()

    def snap(self, arms, legs):
        set_settings(self.src, self.rig, arms, legs)
        for side in SIDES:
            for f in FINGERS:
                self.rig.pose.bones[f"{f}.01_master.{side}"].rotation_quaternion = \
                    Quaternion((0, 0, 1), 0.4)
        pose_source(self.src)
        result = bpy.ops.posesnap.snap_pose()
        self.assertEqual(result, {"FINISHED"})
        bpy.context.view_layer.update()

    def assert_fingers_fk(self):
        for side in SIDES:
            for s_name, t_name in finger_pairs(side):
                pb = self.rig.pose.bones[t_name]
                self.assertGreater(pb.matrix_basis.to_quaternion().angle, 1e-3, f"{t_name} not posed")
                self.assertLess(rot_angle(world_matrix(self.rig, t_name), world_matrix(self.src, s_name)),
                                ANGLE_TOL, t_name)
            for f in FINGERS:
                self.assertEqual(float(self.rig.pose.bones[f"{f}.01_ik.{side}"]["FK_IK"]), 0.0)
                master = self.rig.pose.bones[f"{f}.01_master.{side}"]
                self.assertLess(master.matrix_basis.to_quaternion().angle, 1e-5, f"{master.name} posed")


class TestRetargetIK(SceneTestCase):
    def test_ik_mode(self):
        self.snap("IK", "IK")
        src, rig = self.src, self.rig
        for side in SIDES:
            hand = world_matrix(rig, f"hand_ik.{side}")
            wrist = world_matrix(src, f"手首.{side}")
            self.assertLess((hand.translation - wrist.translation).length, POS_TOL, f"hand_ik.{side}")
            self.assertLess(rot_angle(hand, wrist), ANGLE_TOL)
            foot = world_matrix(rig, f"foot_ik.{side}")
            ankle = world_matrix(src, f"足首.{side}")
            self.assertLess((foot.translation - ankle.translation).length, POS_TOL, f"foot_ik.{side}")
            for kind, target, chain in (
                ("ARM", f"upper_arm_ik_target.{side}", ("腕", "ひじ", "手首")),
                ("LEG", f"thigh_ik_target.{side}", ("足", "ひざ", "足首")),
            ):
                a, b, c = (world_matrix(src, f"{n}.{side}").translation for n in chain)
                expected = math_retarget.compute_ik_pole_world(a, b, c, kind)
                got = world_matrix(rig, target).translation
                self.assertLess((got - expected).length, 1e-2, f"{target}: {got} vs {expected}")
                ac = (c - a).normalized()
                proj = a + ac * (b - a).dot(ac)
                self.assertGreater((got - proj).dot(b - proj), 0.0, f"{target} on wrong side")
            for switch in (f"upper_arm_parent.{side}", f"thigh_parent.{side}"):
                self.assertEqual(float(rig.pose.bones[switch]["IK_FK"]), 0.0)
                self.assertTrue(bool(rig.pose.bones[switch]["pole_vector"]))
        torso = world_matrix(rig, "torso").translation
        self.assertGreater((torso - Vector((0, 0, 1.0))).length, 0.01, "torso did not follow center")
        self.assert_fingers_fk()


class TestRetargetFK(SceneTestCase):
    def test_fk_mode(self):
        self.snap("FK", "FK")
        src, rig = self.src, self.rig
        for side in SIDES:
            pairs = (
                ("upper_arm_fk", "腕捩"), ("forearm_fk", "手捩"), ("hand_fk", "手首"),
                ("thigh_fk", "足"), ("shin_fk", "ひざ"), ("foot_fk", "足首"), ("toe", "つま先"),
            )
            for tgt, mmd in pairs:
                angle = rot_angle(world_matrix(rig, f"{tgt}.{side}"), world_matrix(src, f"{mmd}.{side}"))
                self.assertLess(angle, ANGLE_TOL, f"{tgt}.{side} off by {math.degrees(angle):.2f} deg")
            twist_free = rot_angle(world_matrix(rig, f"upper_arm_fk.{side}"), world_matrix(src, f"腕.{side}"))
            self.assertGreater(twist_free, math.radians(10), "arm twist not merged")
            hand_head = world_matrix(rig, f"hand_fk.{side}").translation
            self.assertLess((hand_head - world_matrix(src, f"手首.{side}").translation).length, POS_TOL)
            for switch in (f"upper_arm_parent.{side}", f"thigh_parent.{side}"):
                self.assertEqual(float(rig.pose.bones[switch]["IK_FK"]), 1.0)
        self.assert_fingers_fk()


class TestPoleAndCalibration(unittest.TestCase):
    def test_pole_bent(self):
        a, b, c = Vector((0, 0, 1)), Vector((0, 0.3, 0.5)), Vector((0, 0, 0))
        got = math_retarget.compute_ik_pole_world(a, b, c, "LEG")
        ac = c - a
        proj = a + ac.normalized() * (b - a).dot(ac.normalized())
        expected = b + (b - proj).normalized() * (0.45 * ac.length)
        self.assertLess((got - expected).length, 1e-5)

    def test_pole_collinear(self):
        a, c = Vector((0.15, 0, 1.4)), Vector((0.55, 0, 1.0))
        b = (a + c) / 2
        for kind in ("ARM", "LEG"):
            got = math_retarget.compute_ik_pole_world(a, b, c, kind)
            self.assertTrue(all(math.isfinite(x) for x in got))
            offset = got - b
            self.assertGreater(offset.length, 1e-4)
            self.assertLess(abs(offset.normalized().dot((c - a).normalized())), 1e-3)
        hint = Vector((0, 1, 0))
        got = math_retarget.compute_ik_pole_world(a, b, c, "ARM", fallback_dir=hint)
        self.assertGreater((got - b).dot(hint), 0.0)
        degenerate = math_retarget.compute_ik_pole_world(a, a, a, "ARM")
        self.assertTrue(all(math.isfinite(x) for x in degenerate))

    def test_hip_height_ratio(self):
        clear_scene_objects()
        try:
            src = build_mmd_armature()
            same = build_mmr_rig("RigSame", 1.0)
            big = build_mmr_rig("RigBig", 1.5)
            bpy.context.view_layer.update()
            r1 = math_retarget.compute_hip_height_ratio(src, same)
            r2 = math_retarget.compute_hip_height_ratio(src, big)
            self.assertAlmostEqual(r1, 1.0, delta=0.06)
            self.assertAlmostEqual(r2, 1.5, delta=0.1)
            st = set_settings(src, big)
            st.hip_scale_ratio = 1.0
            self.assertEqual(bpy.ops.posesnap.auto_calibrate_height(), {"FINISHED"})
            self.assertAlmostEqual(st.hip_scale_ratio, r2, delta=1e-4)
        finally:
            clear_scene_objects()


class TestBake(SceneTestCase):
    def _animate_source(self):
        for frame in range(1, 6):
            for side in SIDES:
                pb = self.src.pose.bones[f"腕.{side}"]
                pb.rotation_quaternion = Quaternion((1, 0, 0), math.radians(8 * frame))
                pb.keyframe_insert("rotation_quaternion", frame=frame)
            center = self.src.pose.bones["センター"]
            center.location = (0, 0.01 * frame, 0)
            center.keyframe_insert("location", frame=frame)

    def _key_frames(self, action):
        frames = set()
        for fc in compat.get_action_fcurves(action, getattr(self.rig.animation_data, "action_slot", None)):
            frames.update(round(kp.co[0]) for kp in fc.keyframe_points)
        return frames

    def test_bake_steps(self):
        self._animate_source()
        for step, expected in ((1, {1, 2, 3, 4, 5}), (2, {1, 3, 5})):
            self.rig.animation_data_clear()
            st = set_settings(self.src, self.rig, "IK", "IK")
            st.frame_start, st.frame_end, st.frame_step = 1, 5, step
            self.assertEqual(bpy.ops.posesnap.bake_animation(), {"FINISHED"})
            ad = self.rig.animation_data
            self.assertIsNotNone(ad)
            self.assertIsNotNone(ad.action)
            count = compat.count_action_fcurves(ad.action, getattr(ad, "action_slot", None))
            print(f"\n[bake] step={step} fcurves={count}")
            self.assertGreater(count, 0)
            self.assertEqual(self._key_frames(ad.action), expected)

    def test_compat_keyframes(self):
        action, slot = compat.ensure_action_and_slot(self.rig, "PoseSnapCompatTest")
        self.assertIs(self.rig.animation_data.action, action)
        pb = self.rig.pose.bones["torso"]
        compat.insert_pose_keyframes(pb, 3.0, group_name="torso")
        compat.insert_custom_prop_keyframe(self.rig.pose.bones["upper_arm_parent.L"], "IK_FK", 3.0)
        self.assertGreaterEqual(compat.count_action_fcurves(action, slot), 8)
        self.assertGreater(len(list(compat.get_action_fcurves(action, slot))), 0)


VPD_SAMPLE = """Vocaloid Pose Data file

sample.osm;\t\t// parent file
2;\t\t\t\t// bone count

Bone0{センター
  1.000000,2.000000,3.000000;\t\t// trans x,y,z
  0.000000,0.000000,0.000000,1.000000;\t// Quaternion x,y,z,w
}

Bone1{左腕
  0.000000,0.000000,0.000000;
  0.000000,0.000000,0.382683,0.923880;
}

Morph0{まばたき
  0.750000;
}
"""


class TestVpdParsing(unittest.TestCase):
    unwind = REPO_ROOT / "Unwind.vpd"

    def _entry(self, data, raw_name):
        for e in data.raw_bones:
            if e.name == raw_name:
                return e
        self.fail(f"{raw_name} missing")

    def test_conversions(self):
        loc = vpd_parser.mmd_to_blender_location((1.0, 2.0, 3.0))
        self.assertLess((loc - Vector((0.08, 0.24, 0.16))).length, 1e-6)
        loc2 = vpd_parser.mmd_to_blender_location((1.0, 2.0, 3.0), scale=1.0)
        self.assertLess((loc2 - Vector((1.0, 3.0, 2.0))).length, 1e-6)
        q = vpd_parser.mmd_to_blender_quaternion((0.0, 0.717356, 0.0, 0.696707))
        self.assertAlmostEqual(q.magnitude, 1.0, places=4)
        self.assertAlmostEqual(abs(q.w), 0.696707, places=4)
        self.assertAlmostEqual(q.angle, 2 * math.acos(0.696707), places=4)

    def test_unwind(self):
        if not self.unwind.is_file():
            self.skipTest("Unwind.vpd not in repo root")
        data = vpd_parser.parse_vpd_file(str(self.unwind))
        self.assertEqual(data.declared_bone_count, 116)
        self.assertEqual(len(data.raw_bones), 116)
        self.assertEqual(data.osm_name, "TDA Yume Miyuki City-Day.osm")
        for name in ("全ての親", "グルーブ", "左腕", "右親指２", "左小指３"):
            self._entry(data, name)
        root = self._entry(data, "全ての親")
        self.assertEqual(tuple(round(x, 6) for x in root.rotation_mmd_xyzw), (0.0, 0.717356, 0.0, 0.696707))
        self.assertAlmostEqual(root.rotation_bl.angle, 2 * math.acos(0.696707), places=4)
        groove = self._entry(data, "グルーブ")
        self.assertLess((groove.location_bl - Vector((0.0, 7.950006 * 0.08, -11.250012 * 0.08))).length, 1e-5)
        for e in data.raw_bones:
            self.assertAlmostEqual(e.rotation_bl.magnitude, 1.0, places=3)
        self.assertGreaterEqual(len(data.bones), 100)

    def test_embedded_sample(self):
        data = vpd_parser.parse_vpd_text(VPD_SAMPLE, "sample.vpd")
        self.assertEqual(data.osm_name, "sample.osm")
        self.assertEqual(data.declared_bone_count, 2)
        self.assertEqual(len(data.raw_bones), 2)
        self.assertAlmostEqual(data.morphs["まばたき"].weight, 0.75)
        center = self._entry(data, "センター")
        self.assertLess((center.location_bl - Vector((0.08, 0.24, 0.16))).length, 1e-6)
        arm = self._entry(data, "左腕")
        self.assertAlmostEqual(arm.rotation_bl.angle, math.radians(45), places=4)
        with tempfile.TemporaryDirectory() as tmp:
            utf8 = Path(tmp) / "utf8.vpd"
            utf8.write_bytes(VPD_SAMPLE.encode("utf-8"))
            self.assertEqual(len(vpd_parser.parse_vpd_file(str(utf8)).raw_bones), 2)
            sjis = Path(tmp) / "sjis.vpd"
            sjis.write_bytes(VPD_SAMPLE.encode("cp932"))
            self.assertIn("まばたき", vpd_parser.parse_vpd_file(str(sjis)).morphs)
            broken = Path(tmp) / "broken.vpd"
            broken.write_bytes(VPD_SAMPLE.encode("cp932").replace(b"// parent file", b"// \xff\xfe\x80\xa0"))
            self.assertEqual(len(vpd_parser.parse_vpd_file(str(broken)).raw_bones), 2)


class TestVpdApply(unittest.TestCase):
    unwind = REPO_ROOT / "Unwind.vpd"

    def setUp(self):
        clear_scene_objects()
        self.rig = build_mmr_rig()
        bpy.context.view_layer.update()

    def tearDown(self):
        clear_scene_objects()

    def _apply(self, mode, filepath):
        reset_pose(self.rig)
        bpy.context.view_layer.update()
        rest = {pb.name: world_matrix(self.rig, pb.name).copy() for pb in self.rig.pose.bones}
        st = set_settings(None, self.rig, mode, mode)
        before = (len(bpy.data.objects), len(bpy.data.armatures), len(bpy.data.meshes))
        self.assertEqual(bpy.ops.posesnap.import_vpd(filepath=str(filepath)), {"FINISHED"}, st.last_status)
        after = (len(bpy.data.objects), len(bpy.data.armatures), len(bpy.data.meshes))
        self.assertEqual(before, after, "VPD import left helper data behind")
        bpy.context.view_layer.update()
        return rest

    def _moved(self, rest, name):
        now = world_matrix(self.rig, name)
        return (now.translation - rest[name].translation).length > 1e-4 or rot_angle(now, rest[name]) > 1e-3

    def test_unwind_ik_fk(self):
        if not self.unwind.is_file():
            self.skipTest("Unwind.vpd not in repo root")
        rest = self._apply("IK", self.unwind)
        for name in ("torso", "hand_ik.L", "hand_ik.R", "foot_ik.L", "foot_ik.R", "thigh_ik_target.L",
                     "upper_arm_ik_target.L", "thumb.03.R", "f_pinky.03.L"):
            self.assertTrue(self._moved(rest, name), f"IK: {name} unchanged")
        self.assertLess(world_matrix(self.rig, "torso").translation.z, rest["torso"].translation.z,
                        "groove drop not applied to torso")
        for side in SIDES:
            self.assertEqual(float(self.rig.pose.bones[f"upper_arm_parent.{side}"]["IK_FK"]), 0.0)
            self.assertEqual(float(self.rig.pose.bones[f"thumb.01_ik.{side}"]["FK_IK"]), 0.0)
        rest = self._apply("FK", self.unwind)
        for name in ("torso", "upper_arm_fk.L", "forearm_fk.R", "thigh_fk.L", "shin_fk.R",
                     "thumb.03.R", "f_pinky.03.L"):
            self.assertTrue(self._moved(rest, name), f"FK: {name} unchanged")
        for side in SIDES:
            self.assertEqual(float(self.rig.pose.bones[f"upper_arm_parent.{side}"]["IK_FK"]), 1.0)
            self.assertEqual(float(self.rig.pose.bones[f"thigh_parent.{side}"]["IK_FK"]), 1.0)

    def test_embedded_sample_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.vpd"
            path.write_bytes(VPD_SAMPLE.encode("cp932"))
            rest = self._apply("FK", path)
            self.assertTrue(self._moved(rest, "upper_arm_fk.L"))
            self.assertTrue(self._moved(rest, "torso"))


class TestMorphRigIntegration(unittest.TestCase):
    def setUp(self):
        clear_scene_objects()
        self.rig = build_mmr_rig("RIG-MMR-FaceTest")
        self._add_morphrig_bones(self.rig)
        self.face_mesh = self._build_face_mesh("Face_Mesh", self.rig)
        bpy.context.view_layer.update()

    def tearDown(self):
        bpy.context.scene.tool_settings.use_keyframe_insert_auto = False
        clear_scene_objects()

    @staticmethod
    def _add_morphrig_bones(arm_obj):
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm_obj.data.edit_bones
        for name, x_pos in (
            ("Fader_Blink", 0.30),
            ("Fader_Vowel_A", 0.45),
            ("Fader_Vowel_UIEO", 0.60),
        ):
            bone = eb.new(name)
            bone.head = Vector((x_pos, 0.0, 1.40))
            bone.tail = Vector((x_pos, 0.0, 1.44))
        bpy.ops.object.mode_set(mode="OBJECT")

        def _limit(pbone, min_xyz, max_xyz):
            con = pbone.constraints.new("LIMIT_LOCATION")
            con.owner_space = "LOCAL"
            con.use_min_x = con.use_max_x = True
            con.use_min_y = con.use_max_y = True
            con.use_min_z = con.use_max_z = True
            con.min_x, con.min_y, con.min_z = min_xyz
            con.max_x, con.max_y, con.max_z = max_xyz

        _limit(arm_obj.pose.bones["Fader_Blink"], (0.0, 0.0, 0.0), (0.0, 0.10, 0.0))
        _limit(arm_obj.pose.bones["Fader_Vowel_A"], (0.0, 0.0, 0.0), (0.08, 0.0, 0.0))
        _limit(arm_obj.pose.bones["Fader_Vowel_UIEO"], (-0.04, -0.04, 0.0), (0.04, 0.04, 0.0))

    @staticmethod
    def _build_face_mesh(name, arm_obj):
        mesh_data = bpy.data.meshes.new(name)
        mesh_data.from_pydata([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)], [], [(0, 1, 2)])
        mesh_obj = bpy.data.objects.new(name, mesh_data)
        bpy.context.scene.collection.objects.link(mesh_obj)
        mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj

        mesh_obj.shape_key_add(name="Basis")
        for sk_name in ("まばたき", "あ", "い", "う", "え", "お", "ウィンク"):
            mesh_obj.shape_key_add(name=sk_name)
        return mesh_obj

    @staticmethod
    def _bind_morphrig_driver(mesh_obj, sk_name, arm_obj, bone_name, travel, axis="LOC_Y", direction=1.0):
        kb = mesh_obj.data.shape_keys.key_blocks[sk_name]
        fcurve = kb.driver_add("value")
        driver = fcurve.driver
        driver.type = "AVERAGE"
        var = driver.variables.new()
        var.name = "val"
        var.type = "TRANSFORMS"
        tgt = var.targets[0]
        tgt.id = arm_obj
        tgt.bone_target = bone_name
        tgt.transform_type = axis
        tgt.transform_space = "LOCAL_SPACE"

        while fcurve.modifiers:
            fcurve.modifiers.remove(fcurve.modifiers[0])
        while fcurve.keyframe_points:
            fcurve.keyframe_points.remove(fcurve.keyframe_points[0])
        gen = fcurve.modifiers.new(type="GENERATOR")
        gen.mode = "POLYNOMIAL"
        gen.poly_order = 1
        gen.coefficients[0] = 0.0
        gen.coefficients[1] = float(direction) / float(travel)
        fcurve.update()

    def _bind_all_morphrig_drivers(self):
        self._bind_morphrig_driver(self.face_mesh, "まばたき", self.rig, "Fader_Blink", 0.10, "LOC_Y", 1.0)
        self._bind_morphrig_driver(self.face_mesh, "あ", self.rig, "Fader_Vowel_A", 0.08, "LOC_X", 1.0)
        self._bind_morphrig_driver(self.face_mesh, "い", self.rig, "Fader_Vowel_UIEO", 0.04, "LOC_X", 1.0)
        self._bind_morphrig_driver(self.face_mesh, "う", self.rig, "Fader_Vowel_UIEO", 0.04, "LOC_X", -1.0)
        self._bind_morphrig_driver(self.face_mesh, "お", self.rig, "Fader_Vowel_UIEO", 0.04, "LOC_Y", 1.0)
        self._bind_morphrig_driver(self.face_mesh, "え", self.rig, "Fader_Vowel_UIEO", 0.04, "LOC_Y", -1.0)

    def test_vpd_drives_morphrig_sliders_and_2d_joystick(self):
        self._bind_all_morphrig_drivers()
        vpd_text = (
            "Vocaloid Pose Data file\n\n"
            "sample.osm;\n"
            "1;\n\n"
            "Bone0{センター\n"
            "  0.000000,0.000000,0.000000;\n"
            "  0.000000,0.000000,0.000000,1.000000;\n"
            "}\n\n"
            "Morph0{まばたき\n  0.75;\n}\n"
            "Morph1{あ\n  0.50;\n}\n"
            "Morph2{い\n  0.80;\n}\n"
            "Morph3{お\n  0.60;\n}\n"
            "Morph4{ウィンク\n  0.40;\n}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            vpd_path = Path(tmp) / "facial.vpd"
            vpd_path.write_bytes(vpd_text.encode("cp932"))
            bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
            stats = vpd_parser.apply_vpd_to_mmr(
                bpy.context,
                str(vpd_path),
                self.rig,
                math_retarget.RetargetOptions(frame=5),
            )
            self.assertEqual(stats.get("morphs_driven"), 4)
            self.assertEqual(stats.get("morphs_direct"), 1)
            self.assertEqual(stats.get("morph_bones"), 3)

        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.075, places=5)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_A"].location.x, 0.04, places=5)
        joy = self.rig.pose.bones["Fader_Vowel_UIEO"]
        self.assertAlmostEqual(joy.location.x, 0.032, places=5)
        self.assertAlmostEqual(joy.location.y, 0.024, places=5)
        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["ウィンク"].value, 0.40, places=5)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_sk = self.face_mesh.evaluated_get(depsgraph).data.shape_keys.key_blocks
        self.assertAlmostEqual(eval_sk["まばたき"].value, 0.75, places=4)
        self.assertAlmostEqual(eval_sk["あ"].value, 0.50, places=4)
        self.assertAlmostEqual(eval_sk["い"].value, 0.80, places=4)
        self.assertAlmostEqual(eval_sk["お"].value, 0.60, places=4)

        action = self.rig.animation_data.action
        fcurves = compat.get_action_fcurves(action, getattr(self.rig.animation_data, "action_slot", None))
        paths = {fc.data_path for fc in fcurves}
        self.assertIn('pose.bones["Fader_Blink"].location', paths)
        self.assertIn('pose.bones["Fader_Vowel_UIEO"].location', paths)

        sk_anim = self.face_mesh.data.shape_keys.animation_data
        sk_fcurves = compat.get_action_fcurves(sk_anim.action, getattr(sk_anim, "action_slot", None))
        sk_paths = {fc.data_path for fc in sk_fcurves}
        self.assertIn('key_blocks["ウィンク"].value', sk_paths)
        self.assertNotIn('key_blocks["まばたき"].value', sk_paths)
        self.assertNotIn('key_blocks["い"].value', sk_paths)

    def test_vmd_bake_morphrig_and_negative_joystick_quadrant(self):
        self._bind_all_morphrig_drivers()
        src_arm = build_mmd_armature("MMD_Source_Morph")
        src_mesh = self._build_face_mesh("MMD_Face_Mesh", src_arm)
        src_kbs = src_mesh.data.shape_keys.key_blocks

        # Frame 1: う=0.5, え=0.75; Frame 2: い=0.8, う=0.3 (opposing poles -> +0.5*half).
        frame_values = {
            1: {"まばたき": 0.25, "い": 0.0, "う": 0.50, "お": 0.0, "え": 0.75, "ウィンク": 0.20},
            2: {"まばたき": 0.90, "い": 0.80, "う": 0.30, "お": 0.40, "え": 0.0, "ウィンク": 0.60},
        }
        for frame, values in frame_values.items():
            for name, val in values.items():
                src_kbs[name].value = val
                compat.insert_shape_key_keyframe(src_kbs[name], frame=frame)

        math_retarget.bake_armature_range(
            bpy.context,
            src_arm,
            self.rig,
            math_retarget.RetargetOptions(),
            frame_start=1,
            frame_end=2,
            frame_step=1,
        )

        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.025, places=5)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.x, -0.02, places=5)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.y, -0.03, places=5)

        bpy.context.scene.frame_set(2)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.09, places=5)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.x, 0.02, places=5)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.y, 0.016, places=5)
        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["ウィンク"].value, 0.60, places=5)

    def test_without_morphrig_drivers_falls_back_cleanly(self):
        vpd_text = (
            "Vocaloid Pose Data file\n\n"
            "sample.osm;\n"
            "0;\n\n"
            "Morph0{まばたき\n  0.65;\n}\n"
            "Morph1{い\n  0.45;\n}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            vpd_path = Path(tmp) / "plain.vpd"
            vpd_path.write_bytes(vpd_text.encode("utf-8"))
            stats = vpd_parser.apply_vpd_to_mmr(bpy.context, str(vpd_path), self.rig)
            self.assertEqual(stats.get("morphs_driven"), 0)
            self.assertEqual(stats.get("morphs_direct"), 2)

        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["まばたき"].value, 0.65, places=5)
        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["い"].value, 0.45, places=5)


class TestVmdImport(unittest.TestCase):
    def setUp(self):
        clear_scene_objects()
        self.rig = build_mmr_rig("RIG-MMR-VmdTest")
        TestMorphRigIntegration._add_morphrig_bones(self.rig)
        self.face_mesh = TestMorphRigIntegration._build_face_mesh("Face_Vmd", self.rig)
        TestMorphRigIntegration._bind_morphrig_driver(
            self.face_mesh, "まばたき", self.rig, "Fader_Blink", 0.10, "LOC_Y", 1.0
        )
        TestMorphRigIntegration._bind_morphrig_driver(
            self.face_mesh, "い", self.rig, "Fader_Vowel_UIEO", 0.04, "LOC_X", 1.0
        )
        TestMorphRigIntegration._bind_morphrig_driver(
            self.face_mesh, "お", self.rig, "Fader_Vowel_UIEO", 0.04, "LOC_Y", 1.0
        )
        bpy.context.view_layer.update()

    def tearDown(self):
        clear_scene_objects()

    @staticmethod
    def _build_vmd_bytes(bone_keys, morph_keys, model_name="TestModel", version=2):
        import struct

        magic = b"Vocaloid Motion Data 0002" if version == 2 else b"Vocaloid Motion Data file"
        name_len = 20 if version == 2 else 10
        out = bytearray(magic.ljust(30, b"\x00"))
        out.extend(model_name.encode("cp932")[:name_len].ljust(name_len, b"\x00"))

        out.extend(struct.pack("<I", len(bone_keys)))
        linear_interp = bytes([20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107] * 4)
        for bname, frame, loc, rot in bone_keys:
            raw_b = bname.encode("cp932")[:15].ljust(15, b"\x00")
            out.extend(struct.pack("<15sI3f4f", raw_b, frame, *loc, *rot))
            out.extend(linear_interp)

        out.extend(struct.pack("<I", len(morph_keys)))
        for mname, frame, weight in morph_keys:
            raw_m = mname.encode("cp932")[:15].ljust(15, b"\x00")
            out.extend(struct.pack("<15sIf", raw_m, frame, float(weight)))
        return bytes(out)

    def test_vmd_parse_and_direct_operator_import(self):
        from posesnap import vmd_parser

        q_id = (0.0, 0.0, 0.0, 1.0)
        q_arm = (0.0, 0.0, math.sin(math.radians(20)), math.cos(math.radians(20)))
        vmd_bytes = self._build_vmd_bytes(
            bone_keys=[
                ("センター", 0, (0.0, 0.0, 0.0), q_id),
                ("センター", 4, (0.0, 2.0, 0.0), q_id),
                ("左腕", 0, (0.0, 0.0, 0.0), q_id),
                ("左腕", 4, (0.0, 0.0, 0.0), q_arm),
            ],
            morph_keys=[
                ("まばたき", 0, 0.0),
                ("まばたき", 4, 0.8),
                ("い", 0, 0.0),
                ("い", 4, 0.5),
                ("お", 0, 0.0),
                ("お", 4, 1.0),
                ("ウィンク", 0, 0.0),
                ("ウィンク", 4, 0.6),
            ],
        )

        parsed = vmd_parser.parse_vmd_bytes(vmd_bytes, "memory.vmd")
        self.assertEqual(parsed.version, 2)
        self.assertEqual(parsed.frame_min, 0)
        self.assertEqual(parsed.frame_max, 4)

        with tempfile.TemporaryDirectory() as tmp:
            vmd_path = Path(tmp) / "motion.vmd"
            vmd_path.write_bytes(vmd_bytes)
            st = set_settings(None, self.rig, "FK", "IK")
            before = (len(bpy.data.objects), len(bpy.data.armatures), len(bpy.data.meshes))
            res = bpy.ops.posesnap.import_vmd(
                filepath=str(vmd_path),
                frame_offset=1,
                frame_step=2,
                set_timeline_range=True,
            )
            self.assertEqual(res, {"FINISHED"}, st.last_status)
            after = (len(bpy.data.objects), len(bpy.data.armatures), len(bpy.data.meshes))
            self.assertEqual(before, after, "VMD import left temporary data behind")

        self.assertEqual(bpy.context.scene.frame_start, 1)
        self.assertEqual(bpy.context.scene.frame_end, 5)

        bpy.context.scene.frame_set(3)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.04, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.x, 0.01, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.y, 0.02, places=4)
        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["ウィンク"].value, 0.3, places=4)

        bpy.context.scene.frame_set(5)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.08, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.x, 0.02, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.y, 0.04, places=4)
        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["ウィンク"].value, 0.6, places=4)

        # Verify sparse keyframes: only frames 1 and 5 (no dense intermediate clutter).
        action = self.rig.animation_data.action
        fcurves = compat.get_action_fcurves(action, getattr(self.rig.animation_data, "action_slot", None))
        blink_fc = next(fc for fc in fcurves if fc.data_path == 'pose.bones["Fader_Blink"].location' and fc.array_index == 1)
        self.assertEqual([int(round(kp.co.x)) for kp in blink_fc.keyframe_points], [1, 5])
        self.assertTrue(all(kp.interpolation == "LINEAR" for kp in blink_fc.keyframe_points))

    def test_vmd_invalid_header_raises(self):
        from posesnap import vmd_parser

        with self.assertRaises(ValueError):
            vmd_parser.parse_vmd_bytes(b"Not a VMD file" * 5)

    def test_vmd_morph_only_with_standalone_morphrig_board(self):
        # Mesh is deformed by a separate MMD armature, while standalone Face & Mouth boards drive its shape keys.
        deform_arm = build_mmd_armature("Deform_MMD")
        face_board = bpy.data.objects.new("MorphRig_Board_Face", bpy.data.armatures.new("BoardData"))
        mouth_board = bpy.data.objects.new("MorphRig_Mouth_Face", bpy.data.armatures.new("MouthData"))
        bpy.context.scene.collection.objects.link(face_board)
        bpy.context.scene.collection.objects.link(mouth_board)
        TestMorphRigIntegration._add_morphrig_bones(face_board)
        TestMorphRigIntegration._add_morphrig_bones(mouth_board)

        char_mesh = TestMorphRigIntegration._build_face_mesh("Char_Mesh", deform_arm)
        TestMorphRigIntegration._bind_morphrig_driver(
            char_mesh, "まばたき", face_board, "Fader_Blink", 0.10, "LOC_Y", 1.0
        )
        TestMorphRigIntegration._bind_morphrig_driver(
            char_mesh, "い", mouth_board, "Fader_Vowel_UIEO", 0.04, "LOC_X", 1.0
        )
        TestMorphRigIntegration._bind_morphrig_driver(
            char_mesh, "え", mouth_board, "Fader_Vowel_UIEO", 0.04, "LOC_Y", -1.0
        )
        bpy.context.view_layer.update()

        vmd_bytes = self._build_vmd_bytes(
            bone_keys=[],
            morph_keys=[
                ("まばたき", 0, 0.0),
                ("まばたき", 2, 0.9),
                ("い", 0, 0.0),
                ("い", 2, 0.75),
                ("え", 0, 0.0),
                ("え", 2, 0.50),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            vmd_path = Path(tmp) / "facial_only.vmd"
            vmd_path.write_bytes(vmd_bytes)
            st = set_settings(deform_arm, face_board, "IK", "IK")
            res = bpy.ops.posesnap.import_vmd(filepath=str(vmd_path), frame_offset=1, frame_step=1)
            self.assertEqual(res, {"FINISHED"}, st.last_status)

        bpy.context.scene.frame_set(3)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(face_board.pose.bones["Fader_Blink"].location.y, 0.09, places=4)
        self.assertAlmostEqual(mouth_board.pose.bones["Fader_Vowel_UIEO"].location.x, 0.03, places=4)
        self.assertAlmostEqual(mouth_board.pose.bones["Fader_Vowel_UIEO"].location.y, -0.02, places=4)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_sk = char_mesh.evaluated_get(depsgraph).data.shape_keys.key_blocks
        self.assertAlmostEqual(eval_sk["まばたき"].value, 0.9, places=4)
        self.assertAlmostEqual(eval_sk["い"].value, 0.75, places=4)
        self.assertAlmostEqual(eval_sk["え"].value, 0.50, places=4)

    def test_vmd_morph_only_with_constrained_deform_arm_preserves_body_pose(self):
        # Target MMR rig constrains a separate deform armature whose mesh holds driven shape keys.
        deform_arm = build_mmd_armature("Deform_Constrained")
        con = deform_arm.pose.bones["センター"].constraints.new("COPY_TRANSFORMS")
        con.target = self.rig
        con.subtarget = "torso"

        # Replace default face_mesh modifier to point to deform_arm instead of self.rig.
        self.face_mesh.modifiers["Armature"].object = deform_arm
        self.rig.pose.bones["torso"].location = Vector((0.0, 0.0, 0.25))
        bpy.context.view_layer.update()

        vmd_bytes = self._build_vmd_bytes(
            bone_keys=[],
            morph_keys=[("まばたき", 0, 0.6)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            vmd_path = Path(tmp) / "blink_only.vmd"
            vmd_path.write_bytes(vmd_bytes)
            st = set_settings(None, self.rig, "IK", "IK")
            res = bpy.ops.posesnap.import_vmd(filepath=str(vmd_path), frame_offset=1, frame_step=1)
            self.assertEqual(res, {"FINISHED"}, st.last_status)

        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.06, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["torso"].location.z, 0.25, places=4)

    def test_vmd_multi_file_merge_body_and_morph(self):
        q_id = (0.0, 0.0, 0.0, 1.0)
        q_arm = (0.0, 0.0, math.sin(math.radians(25)), math.cos(math.radians(25)))
        body_vmd = self._build_vmd_bytes(
            bone_keys=[
                ("センター", 0, (0.0, 0.0, 0.0), q_id),
                ("センター", 4, (0.0, 3.0, 0.0), q_id),
                ("左腕", 0, (0.0, 0.0, 0.0), q_id),
                ("左腕", 4, (0.0, 0.0, 0.0), q_arm),
            ],
            morph_keys=[],
        )
        face_vmd = self._build_vmd_bytes(
            bone_keys=[],
            morph_keys=[
                ("まばたき", 0, 0.0),
                ("まばたき", 6, 1.0),
                ("い", 0, 0.0),
                ("い", 6, 0.5),
            ],
        )
        lip_vmd = self._build_vmd_bytes(
            bone_keys=[],
            morph_keys=[
                ("お", 0, 0.0),
                ("お", 6, 0.75),
                ("ウィンク", 0, 0.0),
                ("ウィンク", 6, 0.4),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "body.vmd").write_bytes(body_vmd)
            (Path(tmp) / "face.vmd").write_bytes(face_vmd)
            (Path(tmp) / "lip.vmd").write_bytes(lip_vmd)
            st = set_settings(None, self.rig, "FK", "IK")
            res = bpy.ops.posesnap.import_vmd(
                directory=tmp,
                files=[{"name": "body.vmd"}, {"name": "face.vmd"}, {"name": "lip.vmd"}],
                frame_offset=1,
                set_timeline_range=True,
            )
            self.assertEqual(res, {"FINISHED"}, st.last_status)
            self.assertIn("3 .vmd files", st.last_status)

        self.assertEqual(bpy.context.scene.frame_start, 1)
        self.assertEqual(bpy.context.scene.frame_end, 7)

        bpy.context.scene.frame_set(7)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Blink"].location.y, 0.10, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.x, 0.02, places=4)
        self.assertAlmostEqual(self.rig.pose.bones["Fader_Vowel_UIEO"].location.y, 0.03, places=4)
        self.assertAlmostEqual(self.face_mesh.data.shape_keys.key_blocks["ウィンク"].value, 0.4, places=4)

        action = self.rig.animation_data.action
        fcurves = compat.get_action_fcurves(action, getattr(self.rig.animation_data, "action_slot", None))
        blink_fc = next(fc for fc in fcurves if fc.data_path == 'pose.bones["Fader_Blink"].location' and fc.array_index == 1)
        self.assertEqual([int(round(kp.co.x)) for kp in blink_fc.keyframe_points], [1, 7])


class TestUnregister(unittest.TestCase):
    def test_unregister_register_cycle(self):
        posesnap.unregister()
        self.assertFalse(hasattr(bpy.types, "POSESNAP_PT_main"))
        self.assertFalse(hasattr(bpy.types.Scene, "posesnap"))
        posesnap.register()
        self.assertTrue(hasattr(bpy.types, "POSESNAP_PT_main"))


ORDERED_CASES = (
    TestStartupAndRegistration, TestHierarchyMapping, TestPoleAndCalibration, TestRetargetIK,
    TestRetargetFK, TestBake, TestVpdParsing, TestVpdApply, TestMorphRigIntegration,
    TestVmdImport, TestUnregister,
)


def run():
    print(f"\nPoseSnap tests on Blender {bpy.app.version_string}, files from {REPO_ROOT}")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(c) for c in ORDERED_CASES)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    print(f"[summary] ran={result.testsRun} passed={passed} failures={len(result.failures)} "
          f"errors={len(result.errors)} skipped={len(result.skipped)}")
    try:
        posesnap.unregister()
    except Exception as exc:  # noqa: BLE001
        print(f"[cleanup] unregister failed: {exc}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        EXIT_CODE = run()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        EXIT_CODE = 1
    sys.exit(EXIT_CODE)
