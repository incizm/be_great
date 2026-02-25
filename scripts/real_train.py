from be_great import GReaT
from sklearn import datasets
import pandas as pd
import logging, os
from dotenv import load_dotenv 
from pathlib import Path
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

MODEL_NAME = os.getenv("MODEL_NAME")
EPOCHS = int(os.getenv("EPOCHS"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EXPERIMENT_DIR = os.getenv("EXPERIMENT_DIR")
SAVE_STEPS = int(os.getenv("SAVE_STEPS"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS"))
USE_CPU = os.getenv("USE_CPU") == "True"
USE_FP16 = os.getenv("USE_FP16") == "True"
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")
COLUMN_NAMES = os.getenv("COLUMN_NAMES")
SAVE_PATH = os.getenv("SAVE_PATH") 

data = pd.read_csv(CSV_FILE_PATH)
print(data.head())

column_names = COLUMN_NAMES.split(",")
data.columns = column_names

import torch
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

great = GReaT(
    MODEL_NAME,
    epochs=EPOCHS,
    experiment_dir=EXPERIMENT_DIR, 
    use_cpu=USE_CPU, 
    save_steps=SAVE_STEPS, 
    logging_steps=LOGGING_STEPS, 
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="constant",
    batch_size=BATCH_SIZE,     
    fp16=USE_FP16          
)

if USE_CPU == "False" and torch.cuda.is_available():
    great.model.to("cuda")

print("great model device:", great.model.device)

trainer = great.fit(data, column_names=column_names)

great.save(SAVE_PATH)
