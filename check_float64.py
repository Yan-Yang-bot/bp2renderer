#!./bp2renderer/bin/python

import platform
import sys

# ---------- ANSI colors ----------
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    END = "\033[0m"

def green(x): return C.GREEN + x + C.END
def red(x):   return C.RED + x + C.END
def cyan(x):  return C.CYAN + x + C.END
def bold(x):  return C.BOLD + x + C.END

print()
print(bold("OS detected:"), cyan(platform.system()))
print()

# ---------- CPU float64 ----------
print(bold("CPU float64:"), end=" ")

try:
    import numpy as np
    a = np.ones((2,2), dtype=np.float64)
    _ = a @ a
    print(green("SUPPORTED ✅"))
except Exception as e:
    print(red("NOT supported ❌"))
    sys.exit()

print()

# ---------- GPU checks ----------
try:
    import torch
except:
    print(red("PyTorch not installed → GPU test skipped"))
    sys.exit()

cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# ---------- CUDA ----------
if cuda_available:
    name = torch.cuda.get_device_name(0)
    print(bold("CUDA GPU detected:"), cyan(name))

    try:
        x = torch.ones((3,3), device="cuda", dtype=torch.float64)
        _ = x @ x
        print(bold("CUDA float64:"), green("SUPPORTED ✅"))
    except Exception:
        print(bold("CUDA float64:"), red("NOT supported ❌"))

else:
    print(bold("CUDA"), red("not available"))

# ---------- MPS ----------
if mps_available:
    print(bold("Apple MPS GPU detected"))

    try:
        x = torch.ones((3,3), device="mps", dtype=torch.float64)
        _ = x @ x
        print(bold("MPS float64:"), green("SUPPORTED ✅"))
    except Exception:
        print(bold("MPS float64:"), red("NOT supported ❌ (expected on macOS)"))

else:
    print(bold("MPS"), red("not available"))

print()
