"""Constants shared by every notebook and module. Change them here only."""

from pathlib import Path

RANDOM_STATE = 42
TARGET = "STROKE"
TEST_SIZE = 0.20

CV_FOLDS = 5
MAIN_METRIC = "pr_auc"
TARGET_RECALL = 0.80
RESULTS_FILE = "results.csv"

CLASS_NAMES = ["İnme Yok", "İnme Var"]

# Repo layout: src/../data. Colab: the working directory the files were uploaded to.
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data" if _HERE.name == "src" else Path.cwd()
