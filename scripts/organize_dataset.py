#!/usr/bin/env python3
"""Organize lab_dateset/ into scene-based structure for MultimodalRAG experiments.

Creates a non-destructive organized/ overlay using symlinks, extracts video frames,
filters COCO annotations per scene, and generates a master manifest.

Usage:
    python scripts/organize_dataset.py              # full run
    python scripts/organize_dataset.py --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------
SCENES: dict[str, dict[str, Any]] = {
    "S01_fish_health_tilapia": {
        "description": "Smartphone photos of tilapia in aquaculture ponds",
        "rag_domain": "fish",
        "source": "smartphone",
    },
    "S02_fish_health_grouper": {
        "description": "Brackish dataset images containing only fish annotations",
        "rag_domain": "fish",
        "source": "brackish",
        "category_filter": {"include": [2], "exclusive": True},
        "max_images": 200,
    },
    "S03_disease_white_spot": {
        "description": "White spot disease reference (seed data only, no images yet)",
        "rag_domain": "fish",
        "source": "seed_data",
        "disease_ids": ["WSD"],
    },
    "S04_disease_general": {
        "description": "General disease reference (seed data only, no images yet)",
        "rag_domain": "fish",
        "source": "seed_data",
        "disease_ids": None,  # all diseases
    },
    "S05_environment_net_cage": {
        "description": "Structural background: crab and starfish (net cage environment)",
        "rag_domain": "aquaculture_env",
        "source": "brackish",
        "category_filter": {"include": [1, 6], "exclusive": False},
        "max_images": 100,
    },
    "S06_environment_pond_tank": {
        "description": "Sonar maps, depth charts, and ROV equipment photos",
        "rag_domain": "aquaculture_env",
        "source": "drive_download",
    },
    "S07_underwater_survey_rov": {
        "description": "Extracted frames from ROV and underwater survey videos",
        "rag_domain": "aquaculture_env",
        "source": "video_frames",
    },
    "S08_water_quality_degraded": {
        "description": "UFO-120 high-res / low-res-degraded image pairs",
        "rag_domain": "water_quality",
        "source": "ufo120",
    },
    "S09_multi_species_detection": {
        "description": "Brackish images with 2+ distinct annotation categories",
        "rag_domain": "fish",
        "source": "brackish",
        "category_filter": {"min_categories": 2},
        "max_images": 200,
    },
    "S10_edge_cases_turbidity": {
        "description": "Brackish jellyfish images (edge-case / turbidity challenge)",
        "rag_domain": "fish",
        "source": "brackish",
        "category_filter": {"include": [3], "exclusive": False},
        "max_images": 100,
    },
}

# ---------------------------------------------------------------------------
# Video definitions for frame extraction
# ---------------------------------------------------------------------------
VIDEO_PROFILES: dict[str, dict[str, Any]] = {
    # short clips (<100MB) -> 1 fps, max 30
    "000.avi": {"fps": 1.0, "max_frames": 30},
    "fish_mjpeg.avi": {"fps": 1.0, "max_frames": 30},
    "RoV 碼頭水下攝影加聲納.mp4": {"fps": 0.5, "max_frames": 30},
    "RoV_碼頭水下攝影加聲納_水下10m.mp4": {"fps": 0.5, "max_frames": 30},
    # medium clips (100MB-1GB) -> 0.1 fps, max 50
    "7DF3B0FFB35FEC7572361CDBA53A050B69C09326.mp4": {"fps": 0.1, "max_frames": 50},
    "62BA91ED674B923656BDA07E8A52F43F080637F6.mp4": {"fps": 0.1, "max_frames": 50},
    "2026.01.07-仕國衛1號(TRITON 1) 水下攝影紀錄.wmv": {"fps": 0.1, "max_frames": 50},
    "RoV海巡任務.mp4": {"fps": 0.1, "max_frames": 50},
    "RoV水下攝影_驗證無人船聲納找到的淤積點.mp4": {"fps": 0.1, "max_frames": 50},
    "RoV 船底攝影加聲納.mp4": {"fps": 0.1, "max_frames": 50},
    # long clips (>1GB) -> scene change detection, max 50
    "AAS自動巡檢系統 2022-02-11 14-17-43.mp4": {"scene_detect": 0.3, "max_frames": 50},
    "RoV海岸測試.mp4": {"scene_detect": 0.3, "max_frames": 50},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def relative_symlink(target: Path, link: Path) -> None:
    """Create a relative symlink from *link* pointing to *target*."""
    rel = os.path.relpath(target, link.parent)
    link.symlink_to(rel)


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------
class COCOIndex:
    """Lightweight index over COCO annotation JSON."""

    def __init__(self, json_path: Path):
        with open(json_path) as f:
            data = json.load(f)
        self.info = data.get("info", {})
        self.licenses = data.get("licenses", [])
        self.categories = data.get("categories", [])
        self.images: dict[int, dict] = {img["id"]: img for img in data["images"]}
        # image_id -> list[annotation]
        self.img_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            self.img_anns[ann["image_id"]].append(ann)
        # image_id -> set of category_ids
        self.img_cats: dict[int, set[int]] = {}
        for img_id, anns in self.img_anns.items():
            self.img_cats[img_id] = {a["category_id"] for a in anns}

    def filter_images(
        self,
        include_cats: list[int] | None = None,
        exclusive: bool = False,
        min_categories: int | None = None,
        max_images: int | None = None,
    ) -> list[int]:
        """Return image IDs matching the filter criteria."""
        result: list[int] = []
        for img_id in sorted(self.images.keys()):
            cats = self.img_cats.get(img_id, set())
            if not cats:
                continue
            if include_cats is not None:
                has_target = bool(cats & set(include_cats))
                if not has_target:
                    continue
                if exclusive:
                    # ONLY the target categories (no others)
                    if cats - set(include_cats):
                        continue
            if min_categories is not None:
                if len(cats) < min_categories:
                    continue
            result.append(img_id)
            if max_images and len(result) >= max_images:
                break
        return result

    def export_subset(self, image_ids: list[int]) -> dict[str, Any]:
        """Export a filtered COCO JSON for the given image IDs."""
        id_set = set(image_ids)
        images = [self.images[i] for i in image_ids if i in self.images]
        annotations = []
        ann_id = 0
        for img_id in image_ids:
            for ann in self.img_anns.get(img_id, []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id
                annotations.append(new_ann)
                ann_id += 1
        return {
            "info": self.info,
            "licenses": self.licenses,
            "categories": self.categories,
            "images": images,
            "annotations": annotations,
        }


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float | None = None,
    scene_detect: float | None = None,
    max_frames: int = 50,
    dry_run: bool = False,
) -> int:
    """Extract frames from a video using ffmpeg.

    Returns number of frames extracted (or estimated for dry_run).
    """
    if dry_run:
        log.info("  [DRY-RUN] Would extract up to %d frames from %s", max_frames, video_path.name)
        return max_frames

    ensure_dir(output_dir)
    prefix = output_dir / "frame_%04d.jpg"

    if scene_detect is not None:
        # Scene-change detection
        vf = f"select='gt(scene\\,{scene_detect})',setpts=N/FRAME_RATE/TB"
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", vf,
            "-vsync", "vfr",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            str(prefix),
            "-y", "-loglevel", "warning",
        ]
    else:
        # Fixed FPS extraction
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            str(prefix),
            "-y", "-loglevel", "warning",
        ]

    log.info("  Extracting frames: %s", video_path.name)
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("  Frame extraction timed out for %s", video_path.name)
    except subprocess.CalledProcessError as e:
        log.warning("  Frame extraction failed for %s: %s", video_path.name, e)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    log.info("  Extracted %d frames from %s", len(frames), video_path.name)
    return len(frames)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
KNOWN_DUPLICATE_PAIRS: list[tuple[str, str]] = [
    ("000.avi", "fish_input.avi"),
    ("S__82722824_0.jpg", "S__82722824_1.jpg"),
]

# The AAS duplicate has a (1) in the name
AAS_ORIGINAL = "AAS自動巡檢系統 2022-02-11 14-17-43.mp4"
AAS_DUPLICATE = "AAS自動巡檢系統 2022-02-11 14-17-43 (1).mp4"


def verify_and_remove_duplicates(lab_dir: Path, dry_run: bool = False) -> list[Path]:
    """Verify known duplicates by MD5 and remove them. Returns list of removed files."""
    removed: list[Path] = []

    # Check small file pairs
    for keep_name, dup_name in KNOWN_DUPLICATE_PAIRS:
        keep = lab_dir / keep_name
        dup = lab_dir / dup_name
        if not dup.exists():
            log.info("  Duplicate already removed: %s", dup_name)
            continue
        if not keep.exists():
            log.warning("  Original missing: %s (keeping duplicate)", keep_name)
            continue
        log.info("  Checking MD5: %s vs %s", keep_name, dup_name)
        if md5_file(keep) == md5_file(dup):
            size_mb = dup.stat().st_size / (1024 * 1024)
            if dry_run:
                log.info("  [DRY-RUN] Would remove duplicate: %s (%.1f MB)", dup_name, size_mb)
            else:
                dup.unlink()
                log.info("  Removed duplicate: %s (%.1f MB freed)", dup_name, size_mb)
            removed.append(dup)
        else:
            log.warning("  MD5 mismatch — keeping both: %s, %s", keep_name, dup_name)

    # Check large AAS duplicate (use file size comparison first, then partial MD5)
    aas_keep = lab_dir / AAS_ORIGINAL
    aas_dup = lab_dir / AAS_DUPLICATE
    if aas_dup.exists() and aas_keep.exists():
        size_keep = aas_keep.stat().st_size
        size_dup = aas_dup.stat().st_size
        if size_keep == size_dup:
            # For large files, compare first and last 10MB
            log.info("  Checking partial MD5 for AAS duplicate (%.1f GB)...", size_dup / 1e9)
            match = True
            for offset in [0, max(0, size_keep - 10 * 1024 * 1024)]:
                h1, h2 = hashlib.md5(), hashlib.md5()
                with open(aas_keep, "rb") as f1, open(aas_dup, "rb") as f2:
                    f1.seek(offset)
                    f2.seek(offset)
                    h1.update(f1.read(10 * 1024 * 1024))
                    h2.update(f2.read(10 * 1024 * 1024))
                if h1.hexdigest() != h2.hexdigest():
                    match = False
                    break
            if match:
                size_gb = size_dup / (1024 ** 3)
                if dry_run:
                    log.info("  [DRY-RUN] Would remove AAS duplicate (%.2f GB)", size_gb)
                else:
                    aas_dup.unlink()
                    log.info("  Removed AAS duplicate (%.2f GB freed)", size_gb)
                removed.append(aas_dup)
            else:
                log.warning("  AAS files differ — keeping both")
    elif not aas_dup.exists():
        log.info("  AAS duplicate already removed")

    return removed


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------
class DatasetOrganizer:
    """Orchestrates the full organization pipeline."""

    def __init__(self, lab_dir: Path, organized_dir: Path, dry_run: bool = False):
        self.lab_dir = lab_dir
        self.organized_dir = organized_dir
        self.scenes_dir = organized_dir / "scenes"
        self.dry_run = dry_run
        self.manifest: dict[str, Any] = {
            "version": "1.0",
            "lab_dir": str(lab_dir),
            "scenes": {},
            "statistics": {},
        }
        self._coco_index: COCOIndex | None = None

    @property
    def coco_index(self) -> COCOIndex:
        if self._coco_index is None:
            coco_path = self.lab_dir / "Brackish Underwater.v1-1920x1080.coco" / "train" / "_annotations.coco.json"
            log.info("Loading COCO annotations from %s", coco_path)
            self._coco_index = COCOIndex(coco_path)
        return self._coco_index

    @property
    def brackish_train_dir(self) -> Path:
        return self.lab_dir / "Brackish Underwater.v1-1920x1080.coco" / "train"

    def run(self) -> None:
        """Execute full organization pipeline."""
        log.info("=" * 60)
        log.info("Dataset Organization %s", "[DRY-RUN]" if self.dry_run else "[LIVE]")
        log.info("Lab dir: %s", self.lab_dir)
        log.info("Output:  %s", self.organized_dir)
        log.info("=" * 60)

        # Step 0: Remove duplicates
        log.info("\n--- Step 0: Duplicate Cleanup ---")
        removed = verify_and_remove_duplicates(self.lab_dir, self.dry_run)
        self.manifest["duplicates_removed"] = [str(p) for p in removed]

        # Create base structure
        if not self.dry_run:
            ensure_dir(self.scenes_dir)
            ensure_dir(self.organized_dir / "chromadb_seed")

        # Build each scene
        total_files = 0
        for scene_id, scene_def in SCENES.items():
            log.info("\n--- Building scene: %s ---", scene_id)
            source = scene_def["source"]
            builder = getattr(self, f"_build_{source}", None)
            if builder is None:
                log.error("  No builder for source: %s", source)
                continue
            count = builder(scene_id, scene_def)
            total_files += count
            self.manifest["scenes"][scene_id] = {
                "description": scene_def["description"],
                "rag_domain": scene_def["rag_domain"],
                "source": source,
                "file_count": count,
            }

        # Statistics
        self.manifest["statistics"] = {
            "total_scenes": len(SCENES),
            "total_organized_files": total_files,
        }

        # Build representative images for ChromaDB seed
        log.info("\n--- Building ChromaDB seed representative images ---")
        self._build_chromadb_seed()

        # Write manifest
        manifest_path = self.organized_dir / "dataset_manifest.json"
        if not self.dry_run:
            with open(manifest_path, "w") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
            log.info("\nManifest written: %s", manifest_path)
        else:
            log.info("\n[DRY-RUN] Would write manifest to %s", manifest_path)

        log.info("\n" + "=" * 60)
        log.info("Organization complete. Total files: %d across %d scenes",
                 total_files, len(SCENES))
        log.info("=" * 60)

    # ── S01: Smartphone photos ──────────────────────────
    def _build_smartphone(self, scene_id: str, scene_def: dict) -> int:
        scene_dir = self.scenes_dir / scene_id
        images_dir = scene_dir / "images"
        photos = sorted(self.lab_dir.glob("S__*.jpg"))
        # Exclude known duplicate
        photos = [p for p in photos if p.name != "S__82722824_1.jpg"]
        if not self.dry_run:
            ensure_dir(images_dir)
            self._write_scene_meta(scene_dir, scene_def, len(photos))
        for p in photos:
            link = images_dir / p.name
            if not self.dry_run:
                if not link.exists():
                    relative_symlink(p, link)
            log.info("  -> %s", p.name)
        log.info("  Total: %d smartphone photos", len(photos))
        return len(photos)

    # ── S02/S05/S09/S10: Brackish filtered ──────────────
    def _build_brackish(self, scene_id: str, scene_def: dict) -> int:
        scene_dir = self.scenes_dir / scene_id
        images_dir = scene_dir / "images"
        ann_dir = scene_dir / "annotations"
        cat_filter = scene_def.get("category_filter", {})
        max_images = scene_def.get("max_images")

        image_ids = self.coco_index.filter_images(
            include_cats=cat_filter.get("include"),
            exclusive=cat_filter.get("exclusive", False),
            min_categories=cat_filter.get("min_categories"),
            max_images=max_images,
        )

        if not self.dry_run:
            ensure_dir(images_dir)
            ensure_dir(ann_dir)
            self._write_scene_meta(scene_dir, scene_def, len(image_ids))

        # Symlink images
        count = 0
        for img_id in image_ids:
            img_info = self.coco_index.images[img_id]
            src = self.brackish_train_dir / img_info["file_name"]
            if src.exists():
                link = images_dir / img_info["file_name"]
                if not self.dry_run and not link.exists():
                    relative_symlink(src, link)
                count += 1

        # Export filtered COCO
        if not self.dry_run:
            filtered = self.coco_index.export_subset(image_ids)
            coco_out = ann_dir / "_annotations_filtered.coco.json"
            with open(coco_out, "w") as f:
                json.dump(filtered, f, indent=2)
            log.info("  Filtered COCO: %d images, %d annotations",
                     len(filtered["images"]), len(filtered["annotations"]))

        log.info("  Total: %d images for %s", count, scene_id)
        return count

    # ── S03/S04: Seed data (no images) ──────────────────
    def _build_seed_data(self, scene_id: str, scene_def: dict) -> int:
        scene_dir = self.scenes_dir / scene_id
        seed_src = PROJECT_ROOT / "knowledge_base" / "seed_data" / "fish_diseases" / "diseases.json"

        if not seed_src.exists():
            log.warning("  Seed data not found: %s", seed_src)
            return 0

        with open(seed_src) as f:
            diseases = json.load(f)

        target_ids = scene_def.get("disease_ids")
        if target_ids:
            diseases = [d for d in diseases if d.get("disease_id") in target_ids]

        if not self.dry_run:
            ensure_dir(scene_dir)
            self._write_scene_meta(scene_dir, scene_def, 0)
            # Write filtered diseases reference
            ref_path = scene_dir / "diseases_reference.json"
            with open(ref_path, "w") as f:
                json.dump(diseases, f, indent=2, ensure_ascii=False)

        log.info("  Seed data: %d disease entries for %s", len(diseases), scene_id)
        return 0  # no image files

    # ── S06: Drive-download (sonar/ROV photos) ──────────
    def _build_drive_download(self, scene_id: str, scene_def: dict) -> int:
        scene_dir = self.scenes_dir / scene_id
        images_dir = scene_dir / "images"
        sonar_dir = scene_dir / "sonar_maps"

        # Collect sonar PNGs from drive-download-002
        dd002 = self.lab_dir / "drive-download-20260207T192608Z-1-002"
        sonar_files: list[Path] = []
        rov_photos: list[Path] = []

        if dd002.exists():
            # Sonar maps from 龍洞遊艇港測試資料
            longdong = dd002 / "無人船相關" / "龍洞遊艇港測試資料"
            if longdong.exists():
                sonar_files.extend(sorted(longdong.glob("*.png")))
            # ROV photos
            rov_dir = dd002 / "RoV水下探測"
            if rov_dir.exists():
                rov_photos.extend(sorted(rov_dir.glob("*.jpg")))

        total = len(sonar_files) + len(rov_photos)
        if not self.dry_run:
            ensure_dir(images_dir)
            ensure_dir(sonar_dir)
            self._write_scene_meta(scene_dir, scene_def, total)

        for f in sonar_files:
            link = sonar_dir / f.name
            if not self.dry_run and not link.exists():
                relative_symlink(f, link)
            log.info("  -> sonar: %s", f.name)

        for f in rov_photos:
            link = images_dir / f.name
            if not self.dry_run and not link.exists():
                relative_symlink(f, link)
            log.info("  -> rov photo: %s", f.name)

        log.info("  Total: %d files (%d sonar + %d photos)", total, len(sonar_files), len(rov_photos))
        return total

    # ── S07: Video frame extraction ─────────────────────
    def _build_video_frames(self, scene_id: str, scene_def: dict) -> int:
        scene_dir = self.scenes_dir / scene_id
        frames_dir = scene_dir / "frames"

        if not self.dry_run:
            ensure_dir(frames_dir)

        total_frames = 0
        video_sources = self._discover_videos()

        for video_path, profile in video_sources:
            video_name = video_path.stem
            out_dir = frames_dir / video_name
            n = extract_frames(
                video_path=video_path,
                output_dir=out_dir,
                fps=profile.get("fps"),
                scene_detect=profile.get("scene_detect"),
                max_frames=profile.get("max_frames", 50),
                dry_run=self.dry_run,
            )
            total_frames += n

        if not self.dry_run:
            self._write_scene_meta(scene_dir, scene_def, total_frames)

        log.info("  Total: %d frames from %d videos", total_frames, len(video_sources))
        return total_frames

    def _discover_videos(self) -> list[tuple[Path, dict]]:
        """Find all videos and match to profiles."""
        results: list[tuple[Path, dict]] = []
        # Walk lab_dir for video files
        video_exts = {".avi", ".mp4", ".wmv", ".mov"}
        skip_names = {"fish_input.avi", AAS_DUPLICATE}

        for root, _dirs, files in os.walk(self.lab_dir):
            root_path = Path(root)
            # Skip the organized directory itself
            if str(root_path).startswith(str(self.organized_dir)):
                continue
            for fname in files:
                if fname in skip_names:
                    continue
                fpath = root_path / fname
                if fpath.suffix.lower() in video_exts:
                    # Find matching profile
                    profile = VIDEO_PROFILES.get(fname)
                    if profile is None:
                        # Try to match by looking at just the filename
                        for pname, pval in VIDEO_PROFILES.items():
                            if fname == pname:
                                profile = pval
                                break
                    if profile is None:
                        # Default profile based on file size
                        size = fpath.stat().st_size
                        if size > 1_000_000_000:
                            profile = {"scene_detect": 0.3, "max_frames": 50}
                        elif size > 100_000_000:
                            profile = {"fps": 0.1, "max_frames": 50}
                        else:
                            profile = {"fps": 1.0, "max_frames": 30}
                    results.append((fpath, profile))

        return results

    # ── S08: UFO-120 ────────────────────────────────────
    def _build_ufo120(self, scene_id: str, scene_def: dict) -> int:
        scene_dir = self.scenes_dir / scene_id
        ufo_root = self.lab_dir / "UFO-120"

        if not ufo_root.exists():
            log.warning("  UFO-120 directory not found")
            return 0

        total = 0
        # Symlink preserving structure: train_val/{hr,lrd}, TEST/{hr,lrd}
        for split in ["train_val", "TEST"]:
            for subdir in ["hr", "lrd"]:
                src_dir = ufo_root / split / subdir
                if not src_dir.exists():
                    continue
                dst_dir = scene_dir / split / subdir
                if not self.dry_run:
                    ensure_dir(dst_dir)
                images = sorted(src_dir.glob("*.jpg"))
                for img in images:
                    link = dst_dir / img.name
                    if not self.dry_run and not link.exists():
                        relative_symlink(img, link)
                    total += 1

        if not self.dry_run:
            self._write_scene_meta(scene_dir, scene_def, total)

        log.info("  Total: %d UFO-120 image symlinks", total)
        return total

    # ── ChromaDB seed ───────────────────────────────────
    def _build_chromadb_seed(self) -> None:
        """Build representative_images.json with 5-10 images per scene."""
        seed_dir = self.organized_dir / "chromadb_seed"
        rep: dict[str, list[str]] = {}

        for scene_id in SCENES:
            scene_dir = self.scenes_dir / scene_id
            images: list[str] = []

            # Collect from images/ subdirectory
            img_dir = scene_dir / "images"
            if img_dir.exists():
                for img in sorted(img_dir.glob("*.jpg"))[:10]:
                    images.append(str(img.resolve()))

            # Collect from frames/ subdirectory
            frames_dir = scene_dir / "frames"
            if frames_dir.exists():
                all_frames = sorted(frames_dir.rglob("*.jpg"))
                # Pick evenly spaced frames
                step = max(1, len(all_frames) // 10)
                images.extend(str(f.resolve()) for f in all_frames[::step][:10])

            # Collect from UFO-120 hr/
            hr_dir = scene_dir / "train_val" / "hr"
            if hr_dir.exists():
                for img in sorted(hr_dir.glob("*.jpg"))[:5]:
                    images.append(str(img.resolve()))
            hr_test_dir = scene_dir / "TEST" / "hr"
            if hr_test_dir.exists():
                for img in sorted(hr_test_dir.glob("*.jpg"))[:5]:
                    images.append(str(img.resolve()))

            rep[scene_id] = images[:10]  # cap at 10

        if not self.dry_run:
            ensure_dir(seed_dir)
            out = seed_dir / "representative_images.json"
            with open(out, "w") as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)
            total = sum(len(v) for v in rep.values())
            log.info("  Representative images: %d total across %d scenes", total, len(rep))
        else:
            total = sum(len(v) for v in rep.values())
            log.info("  [DRY-RUN] Would select %d representative images", total)

    # ── Scene metadata ──────────────────────────────────
    def _write_scene_meta(self, scene_dir: Path, scene_def: dict, file_count: int) -> None:
        meta = {
            "description": scene_def["description"],
            "rag_domain": scene_def["rag_domain"],
            "source": scene_def["source"],
            "file_count": file_count,
        }
        ensure_dir(scene_dir)
        with open(scene_dir / "scene_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Organize lab_dateset for MultimodalRAG experiments")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--lab-dir", type=Path, default=None, help="Override lab_dateset path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override organized/ output path")
    args = parser.parse_args()

    # Load config for paths
    cfg = load_config()

    lab_dir = args.lab_dir or Path(getattr(cfg, "dataset", None) and cfg.dataset.lab_dir or str(PROJECT_ROOT / "lab_dateset"))
    organized_dir = args.output_dir or Path(getattr(cfg, "dataset", None) and cfg.dataset.organized_dir or str(lab_dir / "organized"))

    if not lab_dir.exists():
        log.error("Lab directory does not exist: %s", lab_dir)
        sys.exit(1)

    organizer = DatasetOrganizer(lab_dir, organized_dir, dry_run=args.dry_run)
    organizer.run()


if __name__ == "__main__":
    main()
