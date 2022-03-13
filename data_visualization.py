import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os


PATH_TO_DATA = "/home/saurabhg/ego4d-all/ego4d-data/v1/annotations"
LTA_FILE = os.path.join(PATH_TO_DATA, "fho_lta_val.json")
SCOD_FILE = os.path.join(PATH_TO_DATA, "fho_scod_val.json")
STA_FILE = os.path.join(PATH_TO_DATA, "fho_sta_val.json")

with open(LTA_FILE, 'r') as f:
  lta_data = json.load(f)
  
with open(SCOD_FILE, 'r') as f:
  scod_data = json.load(f)
  
with open(STA_FILE, 'r') as f:
  sta_data = json.load(f)

print(type(lta_data))
print(type(scod_data))
print(type(sta_data))