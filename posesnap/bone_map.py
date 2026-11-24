# SPDX-License-Identifier: GPL-3.0-or-later
"""MMD to MikuMikuRig bone mapping tables, name normalization, and rig detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

_FULLWIDTH_TO_HALF = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ．＿－＋　",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz._-+ ",
)
_HALF_DIGITS_TO_FULL = str.maketrans("0123456789", "０１２３４５６７８９")
_SIDE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?P<sep>[._])(?P<side>[LRlr])$")
_IGNORED_BONE_PREFIXES = ("_dummy_", "_shadow_")

ARM_SWITCH_BONES: dict[str, str] = {
    "L": "upper_arm_parent.L",
    "R": "upper_arm_parent.R",
}

LEG_SWITCH_BONES: dict[str, str] = {
    "L": "thigh_parent.L",
    "R": "thigh_parent.R",
}

FINGER_NAMES: tuple[str, ...] = (
    "thumb",
    "f_index",
    "f_middle",
    "f_ring",
    "f_pinky",
)

FINGER_IK_SWITCH_BONES: dict[tuple[str, str], str] = {
    (finger, side): f"{finger}.01_ik.{side}"
    for side in ("L", "R")
    for finger in FINGER_NAMES
}

FINGER_MASTER_BONES: dict[tuple[str, str], str] = {
    (finger, side): f"{finger}.01_master.{side}"
    for side in ("L", "R")
    for finger in FINGER_NAMES
}

TORSO_SPINE_HEAD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "root": ("全ての親",),
    "torso": ("センター", "グルーブ", "腰", "全ての親"),
    "hips": ("腰", "下半身"),
    "spine_fk": ("上半身",),
    "spine_fk.001": ("上半身",),
    "spine_fk.002": ("上半身3", "上半身1"),
    "spine_fk.003": ("上半身2",),
    "chest": ("上半身2", "上半身3", "上半身1", "上半身"),
    "neck": ("首",),
    "head": ("頭",),
    "eyes": ("両目",),
    "eye.L": ("目.L",),
    "eye.R": ("目.R",),
}

SHOULDER_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    side: {
        "target": (f"shoulder.{side}",),
        "shoulder": (f"肩.{side}",),
        "shoulder_p": (f"肩P.{side}",),
        "shoulder_c": (f"肩C.{side}",),
    }
    for side in ("L", "R")
}

ARM_SPEC: dict[str, dict[str, Any]] = {
    side: {
        "switch_bone": ARM_SWITCH_BONES[side],
        "fk": {
            f"upper_arm_fk.{side}": (f"腕.{side}",),
            f"forearm_fk.{side}": (f"ひじ.{side}",),
            f"hand_fk.{side}": (f"手首.{side}",),
        },
        "ik": {
            f"hand_ik.{side}": (f"手首.{side}",),
            "pole_target": f"upper_arm_ik_target.{side}",
        },
        "twist": {
            "upper_twist": (f"腕捩.{side}",),
            "lower_twist": (f"手捩.{side}",),
        },
        "pole_chain": (
            (f"腕.{side}",),
            (f"ひじ.{side}",),
            (f"手首.{side}",),
        ),
    }
    for side in ("L", "R")
}

LEG_SPEC: dict[str, dict[str, Any]] = {
    side: {
        "switch_bone": LEG_SWITCH_BONES[side],
        "fk": {
            f"thigh_fk.{side}": (f"足.{side}", f"足D.{side}"),
            f"shin_fk.{side}": (f"ひざ.{side}", f"ひざD.{side}"),
            f"foot_fk.{side}": (f"足首.{side}", f"足首D.{side}"),
            f"toe.{side}": (f"つま先.{side}", f"足先EX.{side}"),
        },
        "ik": {
            f"foot_ik.{side}": (f"足首.{side}", f"足首D.{side}", f"足ＩＫ.{side}"),
            f"toe.{side}": (f"つま先.{side}", f"足先EX.{side}", f"つま先ＩＫ.{side}"),
            "pole_target": f"thigh_ik_target.{side}",
        },
        "hip_cancel": (f"腰キャンセル.{side}",),
        "pole_chain": (
            (f"足.{side}", f"足D.{side}"),
            (f"ひざ.{side}", f"ひざD.{side}"),
            (f"足首.{side}", f"足首D.{side}"),
        ),
    }
    for side in ("L", "R")
}

_FINGER_BASE_JP: dict[str, str] = {
    "f_index": "人指",
    "f_middle": "中指",
    "f_ring": "薬指",
    "f_pinky": "小指",
}


def _is_ignored_bone_name(name: str) -> bool:
    return name.startswith(_IGNORED_BONE_PREFIXES)


def _split_side(text: str) -> tuple[str, str | None]:
    match = _SIDE_SUFFIX_RE.match(text)
    if match:
        return match.group("base"), match.group("side").upper()
    if len(text) > 1 and text[0] in ("左", "右"):
        return text[1:], "L" if text[0] == "左" else "R"
    return text, None


def normalize_mmd_name(name: str) -> str:
    """Normalize full-width characters, whitespace, and left/right tags to canonical MMD form."""
    cleaned = name.strip().translate(_FULLWIDTH_TO_HALF)
    if not cleaned:
        return ""
    base, side = _split_side(cleaned)
    # Strip redundant leading 左/右 if the bone also carries a .L/.R suffix.
    if side is not None and len(base) > 1 and base[0] in ("左", "右"):
        prefix_side = "L" if base[0] == "左" else "R"
        if prefix_side == side:
            base = base[1:]
    if side is not None:
        return f"{base}.{side}"
    return cleaned


def _expand_text_variants(text: str) -> list[str]:
    half = text.translate(_FULLWIDTH_TO_HALF)
    full = half.translate(_HALF_DIGITS_TO_FULL).replace("IK", "ＩＫ")
    items = [text, half, full]
    if "人差指" in half:
        items.append(half.replace("人差指", "人指"))
        items.append(full.replace("人差指", "人指"))
    elif "人指" in half:
        items.append(half.replace("人指", "人差指"))
        items.append(full.replace("人指", "人差指"))
    return items


def mmd_name_variants(name: str) -> tuple[str, ...]:
    """Generate ordered name variants across .L/.R, 左/右, full-width, and index finger spellings."""
    stripped = name.strip()
    if not stripped:
        return ()

    variants: list[str] = [stripped]
    canonical = normalize_mmd_name(stripped)
    base_half, side = _split_side(stripped.translate(_FULLWIDTH_TO_HALF))

    if side is None:
        for item in _expand_text_variants(stripped):
            if item not in variants:
                variants.append(item)
        if canonical and canonical not in variants:
            variants.append(canonical)
        return tuple(variants)

    prefix = "左" if side == "L" else "右"
    base_variants = _expand_text_variants(base_half)
    if canonical and canonical not in variants:
        variants.append(canonical)
    for base_var in base_variants:
        for candidate in (
            f"{base_var}.{side}",
            f"{prefix}{base_var}",
            f"{base_var}_{side}",
            f"{base_var}.{side.lower()}",
        ):
            if candidate not in variants:
                variants.append(candidate)

    return tuple(variants)


def _extract_bone_entries(source: Any) -> list[tuple[str, str | None]]:
    """Extract (actual_bone_name, mmd_name_j) pairs from any Blender or Python bone container."""
    if source is None:
        return []

    pose_bones = getattr(getattr(source, "pose", None), "bones", None)
    data_bones = getattr(getattr(source, "data", None), "bones", None)

    entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    def _add_bone_obj(bone_obj: Any, fallback_name: str | None = None) -> None:
        bname = getattr(bone_obj, "name", fallback_name)
        if not isinstance(bname, str) or not bname or bname in seen:
            return
        if _is_ignored_bone_name(bname):
            return
        seen.add(bname)
        mmd_bone = getattr(bone_obj, "mmd_bone", None)
        name_j = getattr(mmd_bone, "name_j", None) if mmd_bone is not None else None
        if isinstance(name_j, str) and name_j.strip() and not _is_ignored_bone_name(name_j.strip()):
            entries.append((bname, name_j.strip()))
        else:
            entries.append((bname, None))

    if pose_bones is not None:
        for bone_obj in pose_bones:
            _add_bone_obj(bone_obj)
    if data_bones is not None:
        for bone_obj in data_bones:
            _add_bone_obj(bone_obj)
    if pose_bones is not None or data_bones is not None:
        return entries

    if isinstance(source, Mapping):
        for key, value in source.items():
            if isinstance(key, str) and key and not _is_ignored_bone_name(key):
                if hasattr(value, "mmd_bone") or hasattr(value, "name"):
                    _add_bone_obj(value, fallback_name=key)
                elif key not in seen:
                    seen.add(key)
                    entries.append((key, None))
        return entries

    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        for item in source:
            if isinstance(item, str):
                if item and item not in seen and not _is_ignored_bone_name(item):
                    seen.add(item)
                    entries.append((item, None))
            else:
                _add_bone_obj(item)
    return entries


def resolve_source_bone(source: Any, candidates: tuple[str, ...]) -> str | None:
    """Resolve the first matching source bone name from candidate names, skipping dummy/shadow."""
    entries = _extract_bone_entries(source)
    if not entries or not candidates:
        return None

    exact_map: dict[str, str] = {}
    norm_map: dict[str, str] = {}
    name_j_exact: dict[str, str] = {}
    name_j_norm: dict[str, str] = {}

    for actual_name, name_j in entries:
        exact_map.setdefault(actual_name, actual_name)
        norm_actual = normalize_mmd_name(actual_name)
        if norm_actual:
            norm_map.setdefault(norm_actual, actual_name)
        if name_j:
            name_j_exact.setdefault(name_j, actual_name)
            norm_j = normalize_mmd_name(name_j)
            if norm_j:
                name_j_norm.setdefault(norm_j, actual_name)

    for candidate in candidates:
        variants = mmd_name_variants(candidate)
        for variant in variants:
            if variant in exact_map:
                return exact_map[variant]
        for variant in variants:
            if variant in name_j_exact:
                return name_j_exact[variant]
        norm_cand = normalize_mmd_name(candidate)
        if norm_cand in norm_map:
            return norm_map[norm_cand]
        if norm_cand in name_j_norm:
            return name_j_norm[norm_cand]

    return None


def _extract_target_bone_names(target: Any) -> set[str] | None:
    if target is None:
        return None
    return {name for name, _ in _extract_bone_entries(target)}


def _is_valid_finger_fk_target(bone_name: str, target_names: set[str] | None) -> bool:
    if "_master" in bone_name or "_ik" in bone_name:
        return False
    if target_names is None:
        return True
    return bone_name in target_names


def resolve_finger_fk_pairs(
    source: Any,
    target: Any,
    side: str,
) -> list[tuple[str, str]]:
    """Resolve MMD finger segment pairs strictly to Rigify FK finger bones."""
    side_norm = side.upper()
    if side_norm not in ("L", "R"):
        raise ValueError(f"Expected side 'L' or 'R', got {side!r}")

    target_names = _extract_target_bone_names(target)
    pairs: list[tuple[str, str]] = []

    has_thumb_zero = resolve_source_bone(source, (f"親指０.{side_norm}",)) is not None
    if has_thumb_zero:
        thumb_sources = (
            (f"親指０.{side_norm}",),
            (f"親指１.{side_norm}",),
            (f"親指２.{side_norm}",),
        )
    else:
        thumb_sources = (
            (f"親指１.{side_norm}",),
            (f"親指２.{side_norm}",),
            (f"親指先.{side_norm}",),
        )

    for seg_idx, src_cands in enumerate(thumb_sources, start=1):
        tgt_name = f"thumb.{seg_idx:02d}.{side_norm}"
        if not _is_valid_finger_fk_target(tgt_name, target_names):
            continue
        src_name = resolve_source_bone(source, src_cands)
        if src_name is not None:
            pairs.append((src_name, tgt_name))

    for finger, jp_prefix in _FINGER_BASE_JP.items():
        for seg_idx in (1, 2, 3):
            tgt_name = f"{finger}.{seg_idx:02d}.{side_norm}"
            if not _is_valid_finger_fk_target(tgt_name, target_names):
                continue
            src_name = resolve_source_bone(source, (f"{jp_prefix}{seg_idx}.{side_norm}",))
            if src_name is not None:
                pairs.append((src_name, tgt_name))

    return pairs


def inspect_hierarchy_mapping(source: Any, target: Any) -> dict[str, Any]:
    """Inspect and resolve the full MMD-to-MMR bone mapping for diagnostics and tests."""
    target_names = _extract_target_bone_names(target)

    def _has_target(bone_name: str) -> bool:
        return target_names is None or bone_name in target_names

    torso_spine_head: dict[str, str] = {}
    for tgt_bone, cands in TORSO_SPINE_HEAD_CANDIDATES.items():
        if not _has_target(tgt_bone):
            continue
        src_bone = resolve_source_bone(source, cands)
        if src_bone is not None:
            torso_spine_head[tgt_bone] = src_bone

    shoulders: dict[str, dict[str, str | None]] = {}
    for side in ("L", "R"):
        spec = SHOULDER_CANDIDATES[side]
        tgt_shoulder = spec["target"][0]
        shoulders[side] = {
            "target": tgt_shoulder if _has_target(tgt_shoulder) else None,
            "shoulder": resolve_source_bone(source, spec["shoulder"]),
            "shoulder_p": resolve_source_bone(source, spec["shoulder_p"]),
            "shoulder_c": resolve_source_bone(source, spec["shoulder_c"]),
        }

    arms: dict[str, dict[str, Any]] = {}
    for side in ("L", "R"):
        spec = ARM_SPEC[side]
        fk_map: dict[str, str] = {}
        for tgt_bone, cands in spec["fk"].items():
            if not _has_target(tgt_bone):
                continue
            src_bone = resolve_source_bone(source, cands)
            if src_bone is not None:
                fk_map[tgt_bone] = src_bone

        hand_ik = f"hand_ik.{side}"
        ik_map: dict[str, str | None] = {
            hand_ik: resolve_source_bone(source, spec["ik"][hand_ik]) if _has_target(hand_ik) else None,
            "pole_target": spec["ik"]["pole_target"] if _has_target(spec["ik"]["pole_target"]) else None,
        }
        twist_map = {
            "upper_twist": resolve_source_bone(source, spec["twist"]["upper_twist"]),
            "lower_twist": resolve_source_bone(source, spec["twist"]["lower_twist"]),
        }
        arms[side] = {
            "switch_bone": spec["switch_bone"] if _has_target(spec["switch_bone"]) else None,
            "fk": fk_map,
            "ik": ik_map,
            "twist": twist_map,
        }

    legs: dict[str, dict[str, Any]] = {}
    for side in ("L", "R"):
        spec = LEG_SPEC[side]
        fk_map: dict[str, str] = {}
        for tgt_bone, cands in spec["fk"].items():
            if not _has_target(tgt_bone):
                continue
            src_bone = resolve_source_bone(source, cands)
            if src_bone is not None:
                fk_map[tgt_bone] = src_bone

        foot_ik = f"foot_ik.{side}"
        toe = f"toe.{side}"
        ik_map: dict[str, str | None] = {
            foot_ik: resolve_source_bone(source, spec["ik"][foot_ik]) if _has_target(foot_ik) else None,
            toe: resolve_source_bone(source, spec["ik"][toe]) if _has_target(toe) else None,
            "pole_target": spec["ik"]["pole_target"] if _has_target(spec["ik"]["pole_target"]) else None,
        }
        legs[side] = {
            "switch_bone": spec["switch_bone"] if _has_target(spec["switch_bone"]) else None,
            "fk": fk_map,
            "ik": ik_map,
            "hip_cancel": resolve_source_bone(source, spec["hip_cancel"]),
        }

    fingers_fk = {
        "L": resolve_finger_fk_pairs(source, target, "L"),
        "R": resolve_finger_fk_pairs(source, target, "R"),
    }
    finger_switches = {
        key: bone_name
        for key, bone_name in FINGER_IK_SWITCH_BONES.items()
        if _has_target(bone_name)
    }

    features = {
        "has_upper_body_3": resolve_source_bone(source, ("上半身3",)) is not None,
        "has_upper_body_1": resolve_source_bone(source, ("上半身1",)) is not None,
        "has_shoulder_p": any(shoulders[s]["shoulder_p"] is not None for s in ("L", "R")),
        "has_shoulder_c": any(shoulders[s]["shoulder_c"] is not None for s in ("L", "R")),
        "has_hip_cancel": any(legs[s]["hip_cancel"] is not None for s in ("L", "R")),
        "has_thumb_0": any(
            resolve_source_bone(source, (f"親指０.{s}",)) is not None for s in ("L", "R")
        ),
        "has_arm_twist": any(arms[s]["twist"]["upper_twist"] is not None for s in ("L", "R")),
        "has_hand_twist": any(arms[s]["twist"]["lower_twist"] is not None for s in ("L", "R")),
        "has_leg_ik": any(
            resolve_source_bone(source, (f"足ＩＫ.{s}",)) is not None for s in ("L", "R")
        ),
        "has_leg_deform": any(
            resolve_source_bone(source, (f"足D.{s}",)) is not None for s in ("L", "R")
        ),
    }

    return {
        "torso_spine_head": torso_spine_head,
        "shoulders": shoulders,
        "arms": arms,
        "legs": legs,
        "fingers_fk": fingers_fk,
        "finger_switches": finger_switches,
        "features": features,
    }


def score_mmd_armature(arm_obj: Any) -> int:
    """Score how strongly an object matches a standard MMD source armature."""
    if arm_obj is None or getattr(arm_obj, "type", "ARMATURE") != "ARMATURE":
        return 0

    score = 0
    for marker in ("センター", "上半身", "下半身", "首", "頭"):
        if resolve_source_bone(arm_obj, (marker,)) is not None:
            score += 2
    for marker in (
        "腕.L", "腕.R", "ひじ.L", "ひじ.R", "手首.L", "手首.R",
        "足.L", "足.R", "ひざ.L", "ひざ.R", "足首.L", "足首.R",
    ):
        if resolve_source_bone(arm_obj, (marker,)) is not None:
            score += 2
    for marker in (
        "全ての親", "グルーブ", "上半身2", "肩.L", "肩.R",
        "足ＩＫ.L", "足ＩＫ.R", "親指１.L", "人指１.L",
    ):
        if resolve_source_bone(arm_obj, (marker,)) is not None:
            score += 1

    if hasattr(arm_obj, "mmd_root") or getattr(arm_obj, "mmd_type", None) == "ROOT":
        score += 5

    entries = {name for name, _ in _extract_bone_entries(arm_obj)}
    if {"torso", "upper_arm_parent.L", "thigh_parent.L"}.issubset(entries):
        score -= 20

    return max(score, 0)


def score_mmr_armature(arm_obj: Any) -> int:
    """Score how strongly an object matches a MikuMikuRig / Rigify target control rig."""
    if arm_obj is None or getattr(arm_obj, "type", "ARMATURE") != "ARMATURE":
        return 0

    entries = {name for name, _ in _extract_bone_entries(arm_obj)}
    if not entries:
        return 0

    score = 0
    for marker in ("torso", "hips", "chest", "spine_fk", "neck", "head"):
        if marker in entries:
            score += 3
    for marker in (
        "upper_arm_parent.L", "upper_arm_parent.R",
        "thigh_parent.L", "thigh_parent.R",
    ):
        if marker in entries:
            score += 4
    for marker in (
        "hand_ik.L", "hand_ik.R", "foot_ik.L", "foot_ik.R",
        "upper_arm_fk.L", "upper_arm_fk.R", "thigh_fk.L", "thigh_fk.R",
        "upper_arm_ik_target.L", "upper_arm_ik_target.R",
        "thigh_ik_target.L", "thigh_ik_target.R",
    ):
        if marker in entries:
            score += 2
    for marker in ("thumb.01.L", "f_index.01.L", "thumb.01.R", "f_index.01.R"):
        if marker in entries:
            score += 1

    data = getattr(arm_obj, "data", None)
    if data is not None and hasattr(data, "get") and data.get("rig_id"):
        score += 5
    mmr_prop = getattr(arm_obj, "mmr", None)
    if mmr_prop is not None and getattr(mmr_prop, "MMR_Arm", False):
        score += 5
    if getattr(arm_obj, "name", "").startswith("RIG-"):
        score += 3

    if {"腕.L", "ひじ.L", "足.L"}.issubset(entries) and "upper_arm_parent.L" not in entries:
        score -= 20

    return max(score, 0)


def auto_detect_rig_pair(scene: Any) -> tuple[Any, Any]:
    """Find the highest-scoring (source MMD armature, target MMR rig) pair in a scene."""
    objects = getattr(scene, "objects", scene)
    if objects is None:
        return None, None

    armatures = [obj for obj in objects if getattr(obj, "type", None) == "ARMATURE"]
    if not armatures:
        return None, None

    best_src = None
    best_src_score = 5
    for obj in armatures:
        score = score_mmd_armature(obj)
        if score > best_src_score:
            best_src_score = score
            best_src = obj

    best_tgt = None
    best_tgt_score = 5
    for obj in armatures:
        if obj is best_src:
            continue
        score = score_mmr_armature(obj)
        if score > best_tgt_score:
            best_tgt_score = score
            best_tgt = obj

    return best_src, best_tgt
