#!/bin/bash
###############################################################################
# reproduce_gpqa_1.5B.sh
#
# GPQA-Diamond 재현 스크립트 — DeepSeek-R1-Distill-Qwen-1.5B
# 논문 Table 1 (τ=0.0) 및 Table 2 (τ=0.6) 의 GPQA-Diamond 1.5B 행을 재현합니다.
#
# 실행 방법:
#   bash sh/reproduce_gpqa_1.5B.sh
#
# 사전 준비:
#   1) uv sync   (프로젝트 의존성 설치)
#   2) HuggingFace 에서 GPQA-Diamond 데이터 접근 권한 필요
#      (Idavidrein/gpqa 는 gated dataset — HF 토큰 로그인 필요)
#   3) GPU 1장 이상 (1.5B 모델이므로 single GPU 가능)
#
# 논문 기대값:
#   Table 1 (τ=0.0):
#     Original → Acc: 32.83%, Len: 23449
#     WSC      → Acc: 31.82%, Len: 10004
#   Table 2 (τ=0.6):
#     Original → Acc: 35.86%, Len: 7790
#     WSC      → Acc: 35.35%, Len: 5708
###############################################################################
set -e

export TOKENIZERS_PARALLELISM=false
# GPU 설정 (필요에 따라 변경)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="gpqa_diamond"
DTYPE="bfloat16"
OUT_DIR="./outputs"
RESCUE_PROMPT="I can find a clearer solution if I focus on the core problem."

START_TS=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

format_hms() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

print_elapsed() {
  local label="$1"
  local now_ts elapsed_seconds
  now_ts=$(date +%s)
  elapsed_seconds=$((now_ts - START_TS))
  echo "[경과] ${label}: $(format_hms "${elapsed_seconds}") (${elapsed_seconds}초)"
}

echo "시작 시각: ${START_HUMAN}"

# Prober 다운로드 (없으면)
PROBER_DIR="prober/DeepSeek-R1-Distill-Qwen-1.5B_s1"
PROBER_PATH="${PROBER_DIR}/probe.pkl"
if [ ! -f "$PROBER_PATH" ]; then
  echo "==> Prober 다운로드 중..."
  mkdir -p "$PROBER_DIR"
  uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='xiewenya/WordSaladChopper_Classifier',
    filename='DeepSeek-R1-Distill-Qwen-1.5B_s1/probe.pkl',
    repo_type='model',
    local_dir='prober',
)
"
  echo "==> Prober 다운로드 완료: ${PROBER_PATH}"
fi

echo "============================================================"
echo " [1/4] Table 1 재현: Original (Vanilla), τ=0.0"
echo "============================================================"
uv run python src/generate.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --method vanilla \
  --dtype "$DTYPE" \
  --temperature 0.0 \
  --top-p 1.0 \
  --token-budget 32768 \
  --seed 41 \
  --out-dir "${OUT_DIR}"
print_elapsed "[1/4] Original τ=0.0 완료"

echo ""
echo "============================================================"
echo " [2/4] Table 1 재현: WSC (Ours), τ=0.0"
echo "============================================================"
uv run python src/generate.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --method wsc \
  --dtype "$DTYPE" \
  --prober-path "$PROBER_PATH" \
  --prober-kind logistic \
  --thresh 0.5 \
  --streak-len 2 \
  --len-threshold 10 \
  --short-streak-len 5 \
  --temperature 0.0 \
  --top-p 1.0 \
  --token-budget 32768 \
  --rescue-budget 4096 \
  --max-rescues 1 \
  --rescue-prompt "$RESCUE_PROMPT" \
  --seed 41 \
  --out-dir "${OUT_DIR}"
print_elapsed "[2/4] WSC τ=0.0 완료"

echo ""
echo "============================================================"
echo " [3/4] Table 2 재현: Original (Vanilla), τ=0.6"
echo "============================================================"
uv run python src/generate.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --method vanilla \
  --dtype "$DTYPE" \
  --temperature 0.6 \
  --top-p 0.95 \
  --token-budget 32768 \
  --seed 41 \
  --out-dir "${OUT_DIR}"
print_elapsed "[3/4] Original τ=0.6 완료"

echo ""
echo "============================================================"
echo " [4/4] Table 2 재현: WSC (Ours), τ=0.6"
echo "============================================================"
uv run python src/generate.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --method wsc \
  --dtype "$DTYPE" \
  --prober-path "$PROBER_PATH" \
  --prober-kind logistic \
  --thresh 0.5 \
  --streak-len 2 \
  --len-threshold 10 \
  --short-streak-len 5 \
  --temperature 0.6 \
  --top-p 0.95 \
  --token-budget 32768 \
  --rescue-budget 4096 \
  --max-rescues 1 \
  --rescue-prompt "$RESCUE_PROMPT" \
  --seed 41 \
  --out-dir "${OUT_DIR}"
print_elapsed "[4/4] WSC τ=0.6 완료"

echo ""
echo "============================================================"
echo " 완료! 결과는 ${OUT_DIR} 에 저장됩니다."
echo " 각 실행의 summary.json 에서 accuracy 와 avg_used_tokens 를 확인하세요."
echo "============================================================"
echo ""
echo "결과 확인 명령어:"
echo "  find ${OUT_DIR} -name 'summary.json' | sort | xargs -I{} sh -c 'echo \"--- {} ---\" && cat {}'"

END_TS=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SECONDS=$((END_TS - START_TS))

echo ""
echo "시작 시각: ${START_HUMAN}"
echo "종료 시각: ${END_HUMAN}"
echo "총 소요 시간: $(format_hms "${ELAPSED_SECONDS}") (${ELAPSED_SECONDS}초)"
