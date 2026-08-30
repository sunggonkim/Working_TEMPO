from pathlib import Path
import unittest

class HighLoadLauncherTest(unittest.TestCase):
    def test_one_factor_highload(self):
        text=Path(__file__).with_name('run_tempo_pd_same_server_cache_catalog_v145_highload_in_allocation.sh').read_text()
        self.assertEqual(text.count('srun --exact'),1)
        self.assertIn(' 64 48 128 8 3000 250 16000',text)
        self.assertNotIn('salloc',text)

if __name__ == '__main__': unittest.main()
