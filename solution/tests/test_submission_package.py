"""Delivery safeguards: exclude private assets, detect corruption and keep archives reproducible."""
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from package_submission import ROOT, safe_name, zip_bytes, report
from verify_submission_package import audit_archive


class SubmissionPackageTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'outputs/test_temporary'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_unsafe_archive_paths_rejected(self):
        for name in ('../data.csv', '/data', 'C:/data', 'folder\\data', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)

    def test_archive_is_byte_reproducible(self):
        first, second = self.folder/'a.zip', self.folder/'b.zip'
        zip_bytes(first, {'b.txt': b'b', 'a.txt': b'a'})
        zip_bytes(second, {'a.txt': b'a', 'b.txt': b'b'})
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_private_prediction_cannot_enter_public_archive(self):
        path = self.folder/'leak.zip'
        zip_bytes(path, {'private_candidates/event4/prediction.csv': b'frame,label\n'})
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            audit_archive(path)

    def test_payload_corruption_detected_before_extraction(self):
        original = ROOT/'outputs/submission/package_20261101/mtah2027-possession-runtime-20261101.zip'
        if not original.exists():
            self.skipTest('Local delivery artifact not built')
        with zipfile.ZipFile(original) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        files['solution/predict.py'] += b'\n# corrupted\n'
        modified = self.folder/'corrupt.zip'
        zip_bytes(modified, files)
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            audit_archive(modified)

    def test_reports_leave_administrative_choices_pending(self):
        for variant in ('baseline', 'event2', 'event4'):
            rendered = report(variant).decode()
            self.assertNotIn('{{', rendered)
            self.assertIn('Pending registered team name', rendered)
            self.assertIn('no primary variant selected', rendered)
            self.assertIn('--variant ' + variant, rendered)
