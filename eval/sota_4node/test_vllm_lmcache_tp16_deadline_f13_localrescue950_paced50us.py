import json,threading,unittest
from pathlib import Path
from unittest import mock
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_f13_localrescue950_paced50us_entry as r
ROOT=Path(__file__).resolve().parents[2]
class T(unittest.TestCase):
 def setUp(self):
  with r.e11._RESCUE_RECORDS_LOCK:r.e11._RESCUE_RECORDS.clear()
 def test_contract(self):self.assertEqual(json.loads((ROOT/'eval/sota_4node/real_tp16_deadline_f13_localrescue950_paced50us.json').read_text()),r._expected_contract())
 def test_paced_worker(self):
  class A:
   def __init__(self):self.s=iter(('PROC','PROC','DONE'))
   def transfer(self,h):return 'PROC'
   def check_xfer_state(self,h):return next(self.s)
  class C:
   def __init__(self):self.nixl_agent=A()
   def tempo_prepare(self,o,s):return 'h'
  st={'error':None}; b=threading.Event(); e=threading.Event(); d=threading.Event()
  with mock.patch.object(r.time,'perf_counter_ns',side_effect=(0,950_000_000,951_000_000,960_000_000)),mock.patch.object(r.time,'sleep') as sl:r._paced_worker(channel=C(),obj=object(),receiver_id='rank-8',mode=r.CANDIDATE_MODE,boost=b,entered=e,done=d,state=st)
  rec=r.e11._take_rescue_record(threading.current_thread().name);self.assertEqual(sl.call_args_list,[mock.call(.00005),mock.call(.00005)]);self.assertEqual(rec['paced_boost_sleeps'],st['boost_polls']);self.assertEqual(st['yields'],0)
 def test_launcher(self):
  s=(ROOT/'eval/sota_4node/run_vllm_lmcache_tp16_deadline_f13_localrescue950_paced50us_in_allocation.sh').read_text();self.assertEqual(s.count('srun --exact'),1);self.assertNotIn('salloc',s)
if __name__=='__main__':unittest.main()
