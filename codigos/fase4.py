import numpy as np
import matplotlib as plt
import pandas as pd
import glob
import csv
from pathlib import Path
import os

path = "./fase/csv/"

filenames = glob.glob(path + "*.csv")
print('File names:', filenames)

for file in filenames:
   # reading csv files
   print("\nReading file = ",file)
   print(pd.read_csv(file))