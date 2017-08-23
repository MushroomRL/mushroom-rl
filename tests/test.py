import glob
import os
from tqdm import tqdm

tests = glob.glob('*/*.py')
for t in tqdm(tests, dynamic_ncols=True,
              disable=False, leave=False):
    os.system('python ' + t)
