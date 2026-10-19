import os
import glob
import pandas as pd

REDUX_DATASETS = [
    "ACSF1", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME",
    "Car", "CBF", "Chinatown", "ChlorineConcentration", "CinCECGTorso",
    "Coffee", "Computers", "CricketX", "CricketY", "CricketZ",
    "DiatomSizeReduction", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Earthquakes",
    "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
    "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "FaceAll",
    "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB",
    "FreezerRegularTrain", "FreezerSmallTrain", "GunPoint",
    "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung",
    "Ham", "HandOutlines", "Haptics", "Herring", "HouseTwenty",
    "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
    "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances",
    "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain",
    "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme",
    "PigAirwayPressure", "PigArtPressure", "PigCVP", "Plane", "PowerCons",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType",
    "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2",
    "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
    "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
    "AconityMINIPrinterLarge", "AconityMINIPrinterSmall",
    "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "AsphaltObstaclesUni", "AsphaltPavementTypeUni", "AsphaltRegularityUni",
    "Colposcopy", "Covid3Month_disc", "DodgerLoopDay", "DodgerLoopGame",
    "DodgerLoopWeekend", "ElectricDeviceDetection", "FloodModeling1_disc",
    "FloodModeling2_disc", "FloodModeling3_disc", "GestureMidAirD1",
    "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1",
    "GesturePebbleZ2", "KeplerLightCurves", "MelbournePedestrian",
    "PhoneHeartbeatSound", "PickupGestureWiimoteZ", "PLAID",
    "ShakeGestureWiimoteZ", "SharePriceIncrease", "Tools"
]

UEA_DATASETS = [
    "ArticularyWordRecognition", "AsphaltObstaclesCoordinates",
    "AsphaltRegularityCoordinates", "AtrialFibrillation",
    "AutomotiveRoadTrials", "BeijingPM10Quality_disc",
    "BeijingPM25Quality_disc", "BenzeneConcentration_disc",
    "AppliancesEnergy_disc", "Alzheimers", "BIDMC32HR_disc",
    "BIDMC32SpO2_disc", "Blink", "BoneIntensitiesAgeGroup",
    "BoneProbAgeGroup", "CharacterTrajectories", "CounterMovementJump",
    "Cricket", "DuckDuckGeese", "EigenWorms", "EmoPain", "Epilepsy",
    "ERing", "EthanolConcentration", "EyesOpenShut",
    "HandMovementDirection", "Handwriting",
    "HouseholdPowerConsumption1_disc", "HouseholdPowerConsumption2_disc",
    "IEEEPPG_disc", "IRDS-SFL", "JapaneseVowels", "KERAAL-RTK",
    "KIMORE-PR-C", "KINECAL-QSEO", "Libras", "Locust2022", "LowCost",
    "LSST", "MotionSenseHAR", "MotorImagery", "NATOPS", "PEMS-SF",
    "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1",
    "PhotoStimulation", "MindReading", "SpokenArabicDigits",
    "StandWalkJump", "TactileTextureRecognition", "UCDHE-Rowing-MC",
    "UCIActivity", "UIPRMD-DS-C", "UWaveGestureLibrary", "WISDM"
]

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results", "few")

SELECTION_STRATEGIES = ["kmedoids", "random", "stratified_kmedoids", "stratified_random"]
K_SHOTS = [1, 3, 5, 10]
BENCHMARKS = {
    "redux": REDUX_DATASETS,
    "uea": UEA_DATASETS,
}

PROJECT_REGEX = r"^(?P<dataset>.+)_run(?P<run>\d+)_predict$"


# ============================================================
# Loaders
# ============================================================

def load_standard(file, columns):
    df = pd.read_csv(file)
    cols_to_keep = ["dataset", "run"] + columns
    return df[[c for c in cols_to_keep if c in df.columns]]


def load_emissions(file, columns):
    df = pd.read_csv(file, usecols=["project_name"] + columns)
    parsed = df["project_name"].str.extract(PROJECT_REGEX)
    
    n_bad = parsed["dataset"].isna().sum()
    if n_bad > 0:
        print(f"    Warning: {n_bad} rows with unexpected project_name (ignored)")

    df["dataset"] = parsed["dataset"]
    df["run"] = pd.to_numeric(parsed["run"])
    df = df.dropna(subset=["dataset", "run"])
    df["run"] = df["run"].astype(int)

    return df[["dataset", "run"] + columns]


# ============================================================
# Sources Definition
# ============================================================

SOURCES = {
    "metrics": {
        "subdir": "",
        "loader": load_standard,
        "outputs": [
            (m, m) for m in [
                "accuracy", "auroc", "balacc", "f1",
                "logloss", "sensitivity", "specificity",
            ]
        ],
    },
    "time": {
        "subdir": "time",
        "pattern": "time_*.csv",
        "loader": load_standard,
        "outputs": [("predict_time", "inference_time")],
    },
    "gpu": {
        "subdir": "gpu",
        "pattern": "gpu_*.csv",
        "loader": load_standard,
        "outputs": [
            ("vram_model_mb", "vram_model_mb"),
            ("vram_data_mb", "vram_data_mb"),
            ("vram_total_mb", "vram_total_mb"),
        ],
    },
    "emissions": {
        "subdir": "emissions",
        "pattern": "emissions_predict_[0-9]*.csv",
        "loader": load_emissions,
        "outputs": [("energy_consumed", "inference_energy")],
    },
}

ACTIVE_SOURCES = ["metrics", "time", "gpu", "emissions"]


# ============================================================
# Conversion Processor
# ============================================================

def process_experiment(strategy, k_shot):
    k_dir = os.path.join(BASE_DIR, "few", strategy, f"K_{k_shot}")
    
    if not os.path.isdir(k_dir):
        return

    print(f"\n==========================================")
    print(f"Processing Strategy: {strategy} | K={k_shot}")
    print(f"==========================================")

    for source_name, source in SOURCES.items():

        if source_name not in ACTIVE_SOURCES:
            continue

        for benchmark, target_datasets in BENCHMARKS.items():
            # Metrics files are directly in K_dir as results_{benchmark}_*.csv
            if source_name == "metrics":
                src_dir = k_dir
                pattern = f"results_{benchmark}_*.csv"
            else:
                src_dir = os.path.join(k_dir, source["subdir"])
                pattern = source["pattern"]

            if not os.path.isdir(src_dir):
                continue

            files = sorted(glob.glob(os.path.join(src_dir, pattern)))
            if not files:
                continue

            print(f"  [{source_name.upper()} / {benchmark}] Found {len(files)} files")

            source_columns = [col for col, _ in source["outputs"]]
            dfs = []

            for file in files:
                dfs.append(source["loader"](file, source_columns))

            if not dfs:
                continue

            combined = pd.concat(dfs, ignore_index=True)

            for column, output_name in source["outputs"]:
                if column not in combined.columns:
                    continue

                pivot = combined.pivot(
                    index="dataset",
                    columns="run",
                    values=column,
                )

                pivot = pivot.sort_index(axis=1)
                pivot.columns = [str(run) for run in pivot.columns]
                pivot.index.name = "Resamples:"

                available_datasets = [d for d in target_datasets if d in pivot.index]
                if not available_datasets:
                    continue

                bench_df = pivot.loc[available_datasets]

                output_dir = os.path.join(
                    RESULTS_DIR,
                    strategy,
                    f"K_{k_shot}",
                    benchmark,
                    output_name,
                )
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(
                    output_dir,
                    f"TabPFN-few_{strategy}_K{k_shot}_{output_name}.csv",
                )

                bench_df.to_csv(output_file)
                print(f"    Saved: {output_file} (Shape: {bench_df.shape})")


if __name__ == "__main__":
    print("Starting conversion of few-shot experiment results...")
    for strategy in SELECTION_STRATEGIES:
        for k_shot in K_SHOTS:
            process_experiment(strategy, k_shot)