import inspect
import math

import ultralytics
from ultralytics.engine.trainer import BaseTrainer

print("ultralytics", ultralytics.__version__)
src = inspect.getsource(BaseTrainer.build_optimizer)
start = src.find('if name == "auto"')
print("\n--- build_optimizer, auto branch ---")
print(src[start:start + 1100] if start >= 0 else src[:1100])

REAL = 13110
REAL_X50 = 12587
NBS = 64

ARMS = [
    ("realonly_yolo26",            REAL,                   100),
    ("hybrid v2b r2.00",           REAL + 2.00 * REAL,      25),
    ("hybrid v2  r2.00",           REAL + 2.00 * REAL,      25),
    ("hybrid v9  r2.00",           REAL + 2.00 * REAL,     100),
    ("hybrid v2  r0.75",           REAL + 0.75 * REAL,     100),
    ("hybrid v2b r0.75",           REAL + 0.75 * REAL,      25),
    ("hybrid v2  r1.00",           REAL + 1.00 * REAL,     100),
    ("hybrid v2b r1.00",           REAL + 1.00 * REAL,      25),
    ("hybrid v2  r1.25",           REAL + 1.25 * REAL,     100),
    ("hybrid v2b r1.25",           REAL + 1.25 * REAL,      25),
    ("hybrid v7/v8/perspecies r0.75", REAL + 0.75 * REAL,   25),
    ("hybrid v7/v8/perspecies r1.00", REAL + 1.00 * REAL,   25),
    ("hybrid v7/v8/perspecies r1.25", REAL + 1.25 * REAL,   25),
    ("hybrid v9 r1.25 _x50",       REAL_X50 + 1.25 * REAL_X50, 25),
    ("hybrid v9B r1.25 _x50",      REAL_X50 + 1.25 * REAL_X50, 25),
]

print(f"\n{'arm':34s} {'n_train':>8s} {'epochs':>7s} {'iterations':>11s}  branch")
for name, n, ep in ARMS:
    n = int(round(n))
    it = math.ceil(n / NBS) * ep
    print(f"{name:34s} {n:8d} {ep:7d} {it:11d}  "
          f"{'>10k' if it > 10000 else '<=10k'}")
