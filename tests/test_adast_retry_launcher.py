"""Check retry orchestration using tiny fake result files; no model training."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_scientific_execution_and_input_unchanged(self):
        old = (ROOT / 'runs/adast_fullsource_completion_20261006/colab_launch.py').read_text(encoding='utf8')
        new = (ROOT / 'runs/adast_fullsource_retry_20261007/colab_launch.py').read_text(encoding='utf8')
        start, end = new.index('# Orchestration only:'), new.index("with (work / 'launcher.log')")
        restored = new[:start] + new[end:]
        restored = restored.replace('        backup_completed_pairs()\n', '').replace('    backup_completed_pairs()\n', '')
        restored = restored.replace('sleeptcn_adast_fullsource_retry_20261007', 'sleeptcn_adast_fullsource_completion_20261006')
        restored = restored.replace('adast_fullsource_retry_20261007', 'adast_fullsource_completion_20261006')
        self.assertEqual(ast.dump(ast.parse(old)), ast.dump(ast.parse(restored)))

    def test_backup_only_complete_immutable_pairs_and_hashes(self):
        source = (ROOT / 'runs/adast_fullsource_retry_20261007/colab_launch.py').read_text(encoding='utf8')
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'backup_completed_pairs')
        downloaded = []
        class Files:
            download = staticmethod(downloaded.append)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            (output / 'fold_02').mkdir(parents=True)
            (output / 'fold_03').mkdir()
            (output / 'execution_specification.json').write_text('{}')
            (output / 'fold_02/pair_verification.json').write_text('{"status":"passed"}')
            (output / 'fold_02/final.pt').write_bytes(b'tiny fake checkpoint')
            (output / 'fold_03/latest.pt').write_bytes(b'incomplete')
            namespace = {'output': output, 'backed_up': set(), 'zipfile': zipfile, 'json': json,
                         'sha': lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), 'files': Files}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<backup>', 'exec'), namespace)
            namespace['backup_completed_pairs']()
            namespace['backup_completed_pairs']()
            self.assertEqual(len(downloaded), 1)
            self.assertEqual(namespace['backed_up'], {2})
            with zipfile.ZipFile(downloaded[0]) as archive:
                manifest = json.loads(archive.read('export_manifest.json'))
                self.assertEqual(set(archive.namelist()), set(manifest) | {'export_manifest.json'})
                for name, digest in manifest.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                self.assertNotIn('fold_03/latest.pt', manifest)


if __name__ == '__main__':
    unittest.main()
