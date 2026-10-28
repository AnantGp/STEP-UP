"""Check released fusion behavior without competition data or PyTorch."""

import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
from fuse_rank_scores import empirical_rank, load_scores


class FusionTests(unittest.TestCase):
    def test_stable_ordinal_ranks_for_ties(self):
        np.testing.assert_array_equal(
            empirical_rank(np.array([4.0, 1.0, 1.0, 9.0])),
            np.array([0.75, 0.25, 0.5, 1.0]),
        )

    def test_cli_weights_and_submission_contents(self):
        script = pathlib.Path(__file__).resolve().parents[1] / "code" / "fuse_rank_scores.py"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            primary = root / "primary.txt"
            secondary = root / "secondary.txt"
            output = root / "submission.zip"
            np.savetxt(primary, [1, 2, 3, 4])
            np.savetxt(secondary, [4, 3, 2, 1])
            subprocess.run(
                [sys.executable, str(script), "--primary", str(primary),
                 "--secondary", str(secondary), "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            np.testing.assert_allclose(load_scores(output), [0.33625, 0.52875, 0.72125, 0.91375])
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {"scores.txt", "threshold.txt"})
                self.assertEqual(archive.read("threshold.txt"), b"0.5")


if __name__ == "__main__":
    unittest.main()
