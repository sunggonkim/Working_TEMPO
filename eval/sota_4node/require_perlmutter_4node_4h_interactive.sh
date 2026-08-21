#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?existing Slurm allocation required}"
: "${SLURM_JOB_NODELIST:?Slurm allocation nodelist required}"
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}" == 4 ]]

# Query once at experiment admission.  Never poll Slurm from a loop.
TEMPO_PD_ALLOCATION_RECORD=$(
  scontrol show job "${SLURM_JOB_ID}" --oneliner
)
[[ " ${TEMPO_PD_ALLOCATION_RECORD} " == *" JobState=RUNNING "* ]]
[[
  " ${TEMPO_PD_ALLOCATION_RECORD} " == *" QOS=interactive "*
  || " ${TEMPO_PD_ALLOCATION_RECORD} " == *" QOS=gpu_interactive "*
]]
[[
  " ${TEMPO_PD_ALLOCATION_RECORD} " == *" TimeLimit=04:00:00 "*
  || " ${TEMPO_PD_ALLOCATION_RECORD} " == *" TimeLimit=4:00:00 "*
]]
[[ " ${TEMPO_PD_ALLOCATION_RECORD} " == *" NumNodes=4 "* ]]
[[ "${TEMPO_PD_ALLOCATION_RECORD}" == *"gres/gpu=16"* ]]
unset TEMPO_PD_ALLOCATION_RECORD
