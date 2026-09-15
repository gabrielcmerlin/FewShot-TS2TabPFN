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
import shutil
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
    "AsphaltObstaclesUni",
    "AsphaltPavementTypeUni",
    "AsphaltRegularityUni",
    "Colposcopy",
    "Covid3Month_disc",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "ElectricDeviceDetection",
    "FloodModeling1_disc",
    "FloodModeling2_disc",
    "FloodModeling3_disc",
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

OLD_VERSION_SUFFIXES = ("_nmv", "_eq", "_disc")

def get_redux() -> list[str]:
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
    """Clean up a dataset's subfolder after download."""
    ds_dir = Path(extract_path) / name
    if not ds_dir.is_dir():
        return 0

    deleted = _remove_non_ts_files(ds_dir)
    deleted += _remove_old_ts_duplicates(ds_dir)
    return deleted

def download_dataset(
    name: str, extract_path: str, max_retries: int = 10, force: bool = False
) -> tuple[str, bool, str]:
    """Download a single dataset (train + test splits), retrying on failure.

    If an error occurs at any attempt, the existing dataset directory is completely
    deleted to remove corrupted cache files before trying again.
    """
    ds_dir = Path(extract_path) / name

    if not force and is_already_downloaded(name, extract_path):
        clean_dataset_folder(name, extract_path)
        return name, True, "skipped (already downloaded)"

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            # Se for uma tentativa de retry e o diretório existir, apaga o cache incompleto/corrompido
            if ds_dir.exists():
                shutil.rmtree(ds_dir, ignore_errors=True)

            # Baixa e extrai os splits TRAIN e TEST
            load_classification(name, split="TRAIN", extract_path=extract_path)
            load_classification(name, split="TEST", extract_path=extract_path)
            
            clean_dataset_folder(name, extract_path)

            # Valida se ambos os arquivos foram realmente baixados com sucesso
            if is_already_downloaded(name, extract_path):
                return name, True, ""
            
            raise RuntimeError("Download completed but .ts files are missing or empty.")

        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            # Garante a remoção da pasta em caso de erro para a próxima tentativa ser 100% limpa
            if ds_dir.exists():
                shutil.rmtree(ds_dir, ignore_errors=True)

            if attempt < max_retries:
                time.sleep(2 * attempt)  # backoff exponencial simples

    return name, False, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="./redux",
        help="Directory to download/extract datasets into (default: ./redux)",
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

    datasets = get_redux()
    print(f"Preparing to download {len(datasets)} datasets to '{out_dir}'\n")

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
        print("All datasets downloaded successfully.")

if __name__ == "__main__":
    main()