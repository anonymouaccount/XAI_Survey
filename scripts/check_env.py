"""Print the versions of the key packages and whether a GPU is visible."""
import importlib

for name in ["numpy", "pandas", "sklearn", "xgboost", "shap", "lime", "anchor", "pyxai",
             "torch", "torchvision", "torch_geometric", "captum", "transformers"]:
    try:
        m = importlib.import_module(name)
        print(f"{name:16s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:16s} MISSING ({type(e).__name__}: {e})")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available(),
          "| GPUs:", torch.cuda.device_count(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:
    pass
