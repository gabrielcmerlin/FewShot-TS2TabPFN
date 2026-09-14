"""
Download the 142 datasets used in the Bake Off Redux paper [1].

This is the union of:
  - the 112 original UCR univariate equal-length datasets
    (aeon.datasets.tsc_datasets.univariate_equal_length)
  - the 30 new univariate datasets introduced in the Redux paper
    (aeon.datasets.tsc_datasets.redux)

[1] M. Middlehurst, P. Schafer, and A. Bagnall. "Bake Off Redux: A Review
    and Experimental Evaluation of Recent Time Series Classification
    Algorithms". Data Mining and Knowledge Discovery, 38:1958-2031, 2024.
"""

import argparse
import sys
import time
from pathlib import Path
from aeon.datasets import load_classification

# 112 equal length/no missing univariate TSC datasets used in the HIVE-COTE
univariate_equal_length = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

# 30 new univariate TSC datasets introduced in the Bake Off Redux paper.
redux = [
    "AconityMINIPrinterLarge",
    "AconityMINIPrinterSmall",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "AsphaltObstacles",
    "AsphaltPavementType",
    "AsphaltRegularity",
    "Colposcopy",
    "Covid3Month",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "ElectricDeviceDetection",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "KeplerLightCurves",
    "MelbournePedestrian",
    "PhoneHeartbeatSound",
    "PickupGestureWiimoteZ",
    "PLAID",
    "ShakeGestureWiimoteZ",
    "SharePriceIncrease",
    "Tools",
]

# Suffixes that mark a "corrected"/alternate version of a base .ts file.
# If both `<stem>.ts` and `<stem><suffix>.ts` exist (e.g. `A_TRAIN.ts` and
# `A_TRAIN_nmv.ts`), the plain `<stem>.ts` is considered the old/uncorrected
# version and is removed, keeping only the suffixed one.
OLD_VERSION_SUFFIXES = ("_nmv", "_eq", "_disc")

def get_redux_142() -> list[str]:
    """Return the sorted, de-duplicated union of the two dataset lists."""
    combined = sorted(set(univariate_equal_length) | set(redux))
    return combined

def is_already_downloaded(name: str, extract_path: str) -> bool:
    """Check whether a dataset's TRAIN and TEST .ts files already exist.
    
    Handles variants like <name>_eq, <name>_nmv, etc.
    """
    ds_dir = Path(extract_path) / name
    
    if not ds_dir.is_dir():
        return False

    # Procura qualquer arquivo que termine com _TRAIN.ts e _TEST.ts dentro do diretório
    train_files = [f for f in ds_dir.glob("*_TRAIN.ts") if f.is_file() and f.stat().st_size > 0]
    test_files = [f for f in ds_dir.glob("*_TEST.ts") if f.is_file() and f.stat().st_size > 0]

    return len(train_files) > 0 and len(test_files) > 0

def _remove_non_ts_files(ds_dir: Path) -> int:
    """Delete every file in `ds_dir` that doesn't have a `.ts` extension."""
    deleted = 0
    for path in ds_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".ts":
            try:
                path.unlink()
                deleted += 1
            except OSError:
                pass  # best-effort cleanup, don't fail the whole run
    return deleted

def _remove_old_ts_duplicates(ds_dir: Path) -> int:
    """Delete plain `.ts` files that have a corrected/suffixed counterpart.

    Corrected files are named `<name>_<suffix>_<SPLIT>.ts` (suffix inserted
    right before the TRAIN/TEST marker), e.g. `AsphaltObstacles_eq_TEST.ts`
    is the corrected counterpart of the plain `AsphaltObstacles_TEST.ts`.
    When both exist, the plain file is deleted and only the suffixed one
    is kept.
    """
    ts_files = {p.stem: p for p in ds_dir.glob("*.ts")}
    deleted = 0

    for stem, path in list(ts_files.items()):
        for split in ("_TRAIN", "_TEST"):
            if not stem.endswith(split):
                continue
            prefix = stem[: -len(split)]  # e.g. "AsphaltObstacles_eq"
            for suffix in OLD_VERSION_SUFFIXES:
                if prefix.endswith(suffix):
                    base_prefix = prefix[: -len(suffix)]  # e.g. "AsphaltObstacles"
                    old_stem = base_prefix + split  # e.g. "AsphaltObstacles_TEST"
                    old_path = ts_files.get(old_stem)
                    if old_path is not None and old_path.exists():
                        try:
                            old_path.unlink()
                            deleted += 1
                            ts_files.pop(old_stem, None)
                        except OSError:
                            pass
            break  # matched TRAIN or TEST, no need to check the other

    return deleted

def clean_dataset_folder(name: str, extract_path: str) -> int:
    """Clean up a dataset's subfolder after download.

    - Deletes any file that is not a `.ts` file (e.g. `.arff`, `.txt`,
      `.md`, or anything else the archive shipped).
    - Deletes "old"/uncorrected `.ts` files when a corrected counterpart
      exists (suffixes `_nmv`, `_eq`, `_disc`), keeping only the
      corrected version.

    Returns the total number of files deleted.
    """
    ds_dir = Path(extract_path) / name
    if not ds_dir.is_dir():
        return 0

    deleted = _remove_non_ts_files(ds_dir)
    deleted += _remove_old_ts_duplicates(ds_dir)
    return deleted

def download_dataset(
    name: str, extract_path: str, max_retries: int = 3, force: bool = False
) -> tuple[str, bool, str]:
    """Download a single dataset (train + test splits), retrying on failure.

    Skips the download entirely if the dataset already exists under
    `extract_path`, unless `force` is True. After a successful download,
    cleans up the dataset's subfolder: removes any non-.ts file, and
    removes old/uncorrected .ts files that have a corrected counterpart.
    """
    if not force and is_already_downloaded(name, extract_path):
        # Still clean up in case leftover docs/arff files exist from a
        # previous run of an older version of this script.
        clean_dataset_folder(name, extract_path)
        return name, True, "skipped (already downloaded)"

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            # Loading the "TRAIN" split is enough to trigger the download
            # and extraction of the full dataset (both TRAIN and TEST
            # arrive together in the .ts archive).
            load_classification(name, split="TRAIN", extract_path=extract_path)
            load_classification(name, split="TEST", extract_path=extract_path)
            clean_dataset_folder(name, extract_path)
            return name, True, ""
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 * attempt)  # simple backoff
    return name, False, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="./redux_142",
        help="Directory to download/extract datasets into (default: ./redux_142)",
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

    datasets = get_redux_142()
    print(f"Preparing to download {len(datasets)} datasets to '{out_dir}'\n")

    failed = []
    skipped = 0
    for i, name in enumerate(datasets, start=1):
        print(f"[{i}/{len(datasets)}] '{name}' ...", end=" ", flush=True)
        _, ok, err = download_dataset(
            name, str(out_dir), max_retries=args.retries, force=args.force
        )
        if ok:
            if err:  # non-empty message on success means it was skipped
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
        print("All datasets downloaded successfully.")

if __name__ == "__main__":
    main()