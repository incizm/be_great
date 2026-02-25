from be_great import GReaT

import logging, os
from dotenv import load_dotenv
from pathlib import Path

from utils import set_logging_level
from sklearn import datasets

logger = set_logging_level(logging.INFO)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

LOADED_MODEL_NAME = os.getenv("LOADED_MODEL_NAME")
NUMBER_OF_SAMPLES = int(os.getenv("NUMBER_OF_SAMPLES"))
GEN_FILE_PATH = os.getenv("GEN_FILE_PATH")

great = GReaT.load_from_dir(LOADED_MODEL_NAME)

# Continuous column as start
# data, target = datasets.load_iris(return_X_y=True)
# sepal = list(data[:, 0])
# samples = great.sample(20, device="cpu", k=5, start_col="sepal length", start_col_dist=sepal)

# Random Start
# samples = great.sample(12, device="cpu", k=6)

# Categorical column as start
samples = great.sample(
    NUMBER_OF_SAMPLES, k=5, start_col="target", start_col_dist={"0.0": 0.33, "1.0": 0.33, "2.0": 0.33}
)

print(samples)
samples.to_csv(GEN_FILE_PATH)
