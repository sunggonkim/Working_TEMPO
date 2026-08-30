#!/usr/bin/env python3
"""R25 validation: Q24 unchanged, with 256 generated tokens per request."""
from pathlib import Path
from typing import Any
import json
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
CONTRACT_ID="tp16-calibrated-cache-admission-r25-tokens256";RESULT_SCHEMA="tempo-vllm-tp16-calibrated-admission-result-25";TOKENS=256
def _expected_contract()->dict[str,Any]:
 c=q._expected_contract();c["schema_version"]="tempo-tp16-calibrated-admission-contract-25";c["contract_id"]=CONTRACT_ID;c["controller"]["measurement_horizon_tokens"]=256;c["controller"]["single_factor_from"]="Q24 tokens 64 -> 256";return c
def _load_contract(path:Path):
 p=json.loads(path.read_text());
 if p!=_expected_contract():raise ValueError("R25 contract changed")
 return p
def main():
 q.CONTRACT_ID=CONTRACT_ID;q.RESULT_SCHEMA=RESULT_SCHEMA;q.old.TOKENS=256;q._load_contract=_load_contract;q.main()
if __name__=="__main__":main()
