import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import plotting


class SharedBaselineImportTests(unittest.TestCase):
    def test_workspace_root_dependencies_are_importable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_dir = root / "baselines"
            baseline_dir.mkdir()
            (baseline_dir / "baseline_common.py").write_text(
                "from clustering_common import TOKEN\n"
                "from baseline_config import SETTING\n"
                "VALUE = (TOKEN, SETTING)\n",
                encoding="utf-8",
            )
            (root / "clustering_common.py").write_text("TOKEN = 'root'\n", encoding="utf-8")
            (root / "baseline_config.py").write_text("SETTING = 'config'\n", encoding="utf-8")
            module_names = ("baseline_common", "clustering_common", "baseline_config")
            previous_modules = {name: sys.modules.pop(name, None) for name in module_names}
            previous_path = sys.path[:]
            try:
                sys.path[:] = [
                    entry for entry in sys.path
                    if not (Path(entry) / "baseline_common.py").is_file()
                ]
                with patch.object(plotting, "__file__", str(root / "new_model1" / "src" / "plotting.py")):
                    module = plotting._shared_baseline_plotter()
                self.assertEqual(module.VALUE, ("root", "config"))
                self.assertEqual(Path(module.__file__), baseline_dir / "baseline_common.py")
            finally:
                sys.path[:] = previous_path
                for name in module_names:
                    sys.modules.pop(name, None)
                    if previous_modules[name] is not None:
                        sys.modules[name] = previous_modules[name]


if __name__ == "__main__":
    unittest.main()
