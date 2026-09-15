"""
This processes the 66 recommended core datasets from the Multiverse MTSC archive [1].

[1] M. Middlehurst, A. Rushbrooke, A. Ismail-Fawaz, M. Devanne, G. Forestier,
    A. Dempster, G. I. Webb, C. Holder, and A. Bagnall. "The multiverse of time
    series machine learning: an archive for multivariate time series classification".
    arXiv preprint arXiv:2603.20352, 2026.
"""

import argparse
import sys
import time
from pathlib import Path
from aeon.datasets import load_classification

# 66 MTSC datasets recommended for general purpose algorithm comparison
mv_core = [
    "Alzheimers",
    "AppliancesEnergy_disc",
    "ArticularyWordRecognition",
    "AsphaltObstaclesCoordinates",
    "AsphaltRegularityCoordinates",
    "AtrialFibrillation",
    "AustraliaRainfall_disc",
    "AutomotiveRoadTrials",
    "BeijingPM10Quality_disc",
    "BeijingPM25Quality_disc",
    "BenzeneConcentration_disc",
    "BIDMC32HR_disc",
    "BIDMC32SpO2_disc",
    "Blink",
    "BoneIntensitiesAgeGroup",
    "BoneProbAgeGroup",
    "CharacterTrajectories",
    "CounterMovementJump",
    "Cricket",
    "CrowdSourced",
    "DuckDuckGeese",
    "EigenWorms",
    "EmoPain",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "EyesOpenShut",
    "FaceDetection",
    "FordChallenge",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "HouseholdPowerConsumption1_disc",
    "HouseholdPowerConsumption2_disc",
    "IEEEPPG_disc",
    "IRDS-SFL",
    "JapaneseVowels",
    "KERAAL-RTK",
    "KIMORE-PR-C",
    "KINECAL-QSEO",
    "Libras",
    "Locust2022",
    "LowCost",
    "LSST",
    "MindReading",
    "MotionSenseHAR",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "PhotoStimulation",
    "RacketSports",
    "SelfRegulationSCP1",
    "Skoda",
    "SpokenArabicDigits",
    "StandWalkJump",
    "STEW",
    "TactileTextureRecognition",
    "Tiselac",
    "UCDHE-Rowing-MC",
    "UCIActivity",
    "UIPRMD-DS-C",
    "USCActivity",
    "UWaveGestureLibrary",
    "WISDM",
]

# Suffixes marking corrected or alternate versions of base .ts files
OLD_VERSION_SUFFIXES = ("_nmv", "_eq", "_disc")


def get_multiverse_datasets() -> list[str]:
    """Return sorted and unique list of Multiverse core datasets."""
    return sorted(set(mv_core))


def is_already_downloaded(name: str, extract_path: str) -> bool:
    """Check whether a dataset's TRAIN and TEST .ts files already exist.

    Uses glob matching to handle variant suffixes like <name>_eq, <name>_nmv, etc.
    """
    ds_dir = Path(extract_path) / name

    if not ds_dir.is_dir():
        return False

    train_files = [
        f for f in ds_dir.glob("*_TRAIN.ts") if f.is_file() and f.stat().st_size > 0
    ]
    test_files = [
        f for f in ds_dir.glob("*_TEST.ts") if f.is_file() and f.stat().st_size > 0
    ]

    return len(train_files) > 0 and len(test_files) > 0


def _remove_non_ts_files(ds_dir: Path) -> int:
    """Delete every file in `ds_dir` that does not have a `.ts` extension."""
    deleted = 0
    for path in ds_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".ts":
            try:
                path.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted


def _remove_old_ts_duplicates(ds_dir: Path) -> int:
    """Delete plain `.ts` files that have a corrected/suffixed counterpart."""
    ts_files = {p.stem: p for p in ds_dir.glob("*.ts")}
    deleted = 0

    for stem, path in list(ts_files.items()):
        for split in ("_TRAIN", "_TEST"):
            if not stem.endswith(split):
                continue
            prefix = stem[: -len(split)]
            for suffix in OLD_VERSION_SUFFIXES:
                if prefix.endswith(suffix):
                    base_prefix = prefix[: -len(suffix)]
                    old_stem = base_prefix + split
                    old_path = ts_files.get(old_stem)
                    if old_path is not None and old_path.exists():
                        try:
                            old_path.unlink()
                            deleted += 1
                            ts_files.pop(old_stem, None)
                        except OSError:
                            pass
            break

    return deleted


def clean_dataset_folder(name: str, extract_path: str) -> int:
    """Clean up a dataset's subfolder after download/extraction."""
    ds_dir = Path(extract_path) / name
    if not ds_dir.is_dir():
        return 0

    deleted = _remove_non_ts_files(ds_dir)
    deleted += _remove_old_ts_duplicates(ds_dir)
    return deleted


def download_dataset(
    name: str, extract_path: str, max_retries: int = 3, force: bool = False
) -> tuple[str, bool, str]:
    """Download a single dataset with retries and verification."""
    if not force and is_already_downloaded(name, extract_path):
        clean_dataset_folder(name, extract_path)
        return name, True, "skipped (already downloaded)"

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            load_classification(name, split="TRAIN", extract_path=extract_path)
            load_classification(name, split="TEST", extract_path=extract_path)
            clean_dataset_folder(name, extract_path)
            return name, True, ""
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return name, False, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="./mv_core",
        help="Directory to download/extract datasets into (default: ./mv_core)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts per dataset (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if they already exist under --out",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = get_multiverse_datasets()
    print(f"Preparing to download {len(datasets)} Multiverse datasets to '{out_dir}'\n")

    failed = []
    skipped = 0
    for i, name in enumerate(datasets, start=1):
        print(f"[{i}/{len(datasets)}] '{name}' ...", end=" ", flush=True)
        _, ok, err = download_dataset(
            name, str(out_dir), max_retries=args.retries, force=args.force
        )
        if ok:
            if err:
                skipped += 1
                print(err)
            else:
                print("done.")
        else:
            print(f"FAILED ({err})")
            failed.append((name, err))

    print("\n--- Summary ---")
    print(f"Already had: {skipped}/{len(datasets)}")
    print(f"Newly downloaded: {len(datasets) - len(failed) - skipped}/{len(datasets)}")
    print(f"Successful overall: {len(datasets) - len(failed)}/{len(datasets)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All datasets processed successfully.")


if __name__ == "__main__":
    main()