# CAGE version 2: setup and data phase

These files are the code of version 2 of the project plan. Copy them over the repository, keeping the paths.

Delete from a version 1 repository:
- scripts/check_dataset.py           (its checks moved into cage/data/report.py)
- tests/test_frames_registry.py      (replaced by tests/test_frames.py)
- any second copy of a converter kept from the version 1 Converter reference

Then, on the laptop or the SCC:
    python -m pip install -e .
    pytest                           # the tests that need no dataset
On the SCC, after the converters have run:
    python -m cage.data.report       # writes processed/<dataset>/REPORT.md and processed/RAW_INVENTORY.md
    pytest -m data                   # the tests that read the real data

Executed before delivery (NumPy only): tests/test_frames.py, test_convert_*.py, test_occlusion.py, test_io_depth.py,
test_report.py: 150 cases. Not executed: tests/test_dataset.py, cage/data/dataset.py, cage/data/loaders.py (PyTorch),
scripts/view_sample.py (matplotlib, Open3D), tests/test_datasets_on_disk.py against real data.
