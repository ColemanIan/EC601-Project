import importlib, os, site, sys

print(f"python      {sys.version.split()[0]}  {sys.executable}")
print(f"prefix      {sys.prefix}")
if site.ENABLE_USER_SITE:
    print("WARNING     ~/.local packages are visible; set PYTHONNOUSERSITE=1")

for name in ["numpy", "scipy", "PIL", "cv2", "trimesh", "pycocotools", "shapely", "yaml", "torch", "torchvision", "cage"]:
    module = importlib.import_module(name)
    where = os.path.dirname(module.__file__)
    flag = "" if where.startswith(sys.prefix) or name == "cage" else "   <-- outside the environment"
    print(f"{name:12s}{getattr(module, '__version__', ''):12s}{where}{flag}")

import torch
print(f"torch CUDA  build {torch.version.cuda}, available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device      {torch.cuda.get_device_name(0)}")