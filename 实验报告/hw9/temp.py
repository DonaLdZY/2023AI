import random
import numpy as np
from office_world import Game
from rl_test import run_test

a=np.zeros([4,3])
a[1][2]=1
print(np.argmax(a[1]))