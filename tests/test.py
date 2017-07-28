import glob
import os

tests = glob.glob('*/*.py')
for t in tests:
    os.system('python ' + t)
