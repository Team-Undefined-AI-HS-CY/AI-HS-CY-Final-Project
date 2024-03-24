from ultralytics import YOLO
from settings import *

WEIGHT_PATH = f"{RESULTS_DIR}/Results_Output_Actual/lp_test2/weights/best.pt"

def load_model(version=1):
    return YOLO(WEIGHT_PATH)
