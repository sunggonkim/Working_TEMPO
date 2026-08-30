#!/usr/bin/env python3
"""J17: frozen G14 policy over 32 physical 512KiB descriptors per source."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as e11
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry as e12
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_f13_localrescue950_paced50us_entry as f13
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_g14_localrescue950_paced25us_entry as g14
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol

CANDIDATE_MODE="tempo_fragmented32_local_rescue_950ms_paced25us"
CONTRACT_ID="tp16-fragmented32-local-rescue-950ms-paced25us-j17"
RESULT_SCHEMA="tempo-vllm-tp16-fragmented32-local-rescue-result-17"
CHUNK_BYTES=512<<10
FRAGMENTS=32
BLOCKS=((0,old.FG),(0,old.LMCACHE),(0,CANDIDATE_MODE),(1,CANDIDATE_MODE),(1,old.FG),(1,old.LMCACHE),(2,old.LMCACHE),(2,CANDIDATE_MODE),(2,old.FG))
_ORIGINAL_CHANNEL_CLASS=old._hybrid_channel_class
_ORIGINAL_DESCRIPTOR_COUNT=old._descriptor_count

def _expected_contract():
    p=g14._expected_contract(); p["schema_version"]="tempo-tp16-fragmented32-local-rescue-contract-17"; p["contract_id"]=CONTRACT_ID
    p["algorithm"].update(mode=CANDIDATE_MODE,single_factor_from="G14 physical_descriptors_per_source 1 -> 32",basis_result="results/vllm_lmcache_tp16_deadline_G14v2_job_56975950/result.json")
    p["transfer"].update(physical_descriptors_global=256,physical_descriptors_per_source=32,descriptor_bytes=CHUNK_BYTES,full_buffer_verification=True)
    p["campaign"]["modes"]=[old.FG,old.LMCACHE,CANDIDATE_MODE]
    return p

def _load_contract(path:Path):
    p=json.loads(path.read_text(encoding="utf-8"))
    if p!=_expected_contract(): raise ValueError("J17 contract changed")
    return p

def _make_memory(torch:Any,TensorMemoryObj:Any,MemoryObjMetadata:Any,MemoryFormat:Any):
    backing=torch.empty(old.BYTES_PER_SOURCE+CHUNK_BYTES-1,dtype=torch.uint8,device="cuda")
    offset=(-backing.data_ptr())%CHUNK_BYTES; buffer=backing[offset:offset+old.BYTES_PER_SOURCE]
    if buffer.numel()!=old.BYTES_PER_SOURCE or buffer.data_ptr()%CHUNK_BYTES: raise RuntimeError("fragment buffer alignment changed")
    fragments=[]; index={}
    for i in range(FRAGMENTS):
        raw=buffer[i*CHUNK_BYTES:(i+1)*CHUNK_BYTES]; shape=torch.Size([CHUNK_BYTES])
        obj=TensorMemoryObj(raw_data=raw,metadata=MemoryObjMetadata(shape=shape,dtype=torch.uint8,address=raw.data_ptr(),phy_size=CHUNK_BYTES,ref_count=1,pin_count=0,fmt=MemoryFormat.BINARY,shapes=[shape],dtypes=[torch.uint8]),parent_allocator=None)
        fragments.append(obj); index[raw.data_ptr()]=i
    full_shape=torch.Size([old.BYTES_PER_SOURCE])
    primary=TensorMemoryObj(raw_data=buffer,metadata=MemoryObjMetadata(shape=full_shape,dtype=torch.uint8,address=buffer.data_ptr(),phy_size=old.BYTES_PER_SOURCE,ref_count=1,pin_count=0,fmt=MemoryFormat.BINARY,shapes=[full_shape],dtypes=[torch.uint8]),parent_allocator=None)
    primary._tempo_fragment_objects=fragments
    return backing,buffer,[primary],index

def _channel_class(base_channel:Any):
    parent=_ORIGINAL_CHANNEL_CLASS(base_channel)
    class FragmentedChannel(parent):
        def __init__(self,*args,**kwargs):
            kwargs["align_bytes"]=CHUNK_BYTES; super().__init__(*args,**kwargs)
            actual=_ORIGINAL_DESCRIPTOR_COUNT(self)
            if actual!=FRAGMENTS: raise RuntimeError(f"expected 32 physical descriptors, got {actual}")
            self.tempo_actual_descriptor_count=actual
        @staticmethod
        def _expanded(objects,transfer_spec):
            if len(objects)!=1 or not hasattr(objects[0],"_tempo_fragment_objects"): raise RuntimeError("fragment object marker missing")
            expanded=list(objects[0]._tempo_fragment_objects); spec=dict(transfer_spec); spec["remote_indexes"]=np.arange(FRAGMENTS,dtype=np.uint64)
            return expanded,spec
        def batched_write(self,objects,transfer_spec=None):
            if transfer_spec is None: raise ValueError("transfer_spec required")
            expanded,spec=self._expanded(objects,transfer_spec); completed=int(super().batched_write(expanded,spec))
            if completed!=FRAGMENTS: raise RuntimeError(f"fragmented write completed {completed}/32")
            return 1
        def tempo_prepare(self,objects,transfer_spec):
            expanded,spec=self._expanded(objects,transfer_spec); return super().tempo_prepare(expanded,spec)
    return FragmentedChannel

def _descriptor_count_compat(channel):
    actual=getattr(channel,"tempo_actual_descriptor_count",None)
    if actual!=FRAGMENTS: raise RuntimeError("fragment descriptor proof missing")
    return 1

def _run_block(*args,**kwargs):
    result=f13._run_block(*args,**kwargs); channel=kwargs["channel"]
    result["physical_descriptor_count"]=int(channel.tempo_actual_descriptor_count)
    result["physical_descriptor_bytes"]=CHUNK_BYTES
    result["full_16mib_buffer_verified"]=bool(result["correctness_met"])
    return result

def _aggregate(records,trace,args):
    result=g14._aggregate(records,trace,args); result["schema_version"]=RESULT_SCHEMA; result["contract_id"]=CONTRACT_ID
    exact=all(int(block.get("physical_descriptor_count",0))==FRAGMENTS and int(block.get("physical_descriptor_bytes",0))==CHUNK_BYTES and bool(block.get("full_16mib_buffer_verified")) for record in records for block in record["blocks"])
    for block in result["blocks"]: block.update(physical_descriptors_per_source=FRAGMENTS,physical_descriptor_bytes=CHUNK_BYTES)
    result["config"].update(candidate_mode=CANDIDATE_MODE,physical_descriptors_per_source=FRAGMENTS,physical_descriptors_global=FRAGMENTS*old.SOURCE_COUNT,descriptor_bytes=CHUNK_BYTES)
    result["candidate_gates"]["all_rank_blocks_fragment_geometry_and_full_buffer_exact"]=exact
    result["screen_outcome"]="fragmented32_candidate_pass" if result["overall_correctness_met"] and all(result["candidate_gates"].values()) else "fragmented32_candidate_revise"
    return result

def main():
    g14.CANDIDATE_MODE=f13.CANDIDATE_MODE=CANDIDATE_MODE; g14.CONTRACT_ID=f13.CONTRACT_ID=CONTRACT_ID; g14.RESULT_SCHEMA=f13.RESULT_SCHEMA=RESULT_SCHEMA; g14.BLOCKS=f13.BLOCKS=BLOCKS; f13.PACED_SLEEP_S=0.000025; f13._install_mode()
    for m in (c9,e11,e12): m.CANDIDATE_MODE=CANDIDATE_MODE; m.CONTRACT_ID=CONTRACT_ID; m.RESULT_SCHEMA=RESULT_SCHEMA; m.BLOCKS=BLOCKS
    e11.LOCAL_RESCUE_TRIGGER_MS=950.0; fixed._transfer_worker=f13._paced_worker
    old._make_memory=_make_memory; old._hybrid_channel_class=_channel_class; old._descriptor_count=_descriptor_count_compat
    protocol.install_async_release_protocol(); old.protocol.ReleaseFrame=protocol.ReleaseFrame; old.protocol.install_generic_release_protocol=protocol.install_async_release_protocol; old.bulk.protocol.ReleaseFrame=protocol.ReleaseFrame; old.bulk.protocol.install_generic_release_protocol=protocol.install_async_release_protocol
    v8.CONTRACT_ID=CONTRACT_ID; v8.RESULT_SCHEMA=RESULT_SCHEMA; v8._load_contract=_load_contract; v8._run_block=_run_block; v8._validate_trace=c9._validate_trace; v8._aggregate=_aggregate; v8.main()
if __name__=="__main__": main()
