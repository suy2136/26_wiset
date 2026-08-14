# NetLLM-VP: Patch Selection + Selector/Speculative Decoding + AdaLoRA

360도 영상 viewport prediction(VP)을 위한 [NetLLM](https://github.com/duowuyms/NetLLM) 기반
파이프라인에, 궤적 조건부 이미지 patch 선택(**Patch Selection**), 임베딩 시퀀스 축소 +
draft-verify 가속(**Selector + Speculative Decoding**), 그리고 rank-adaptive LoRA
(**AdaLoRA**)를 하나의 파이프라인으로 통합한 프로젝트입니다.

기반 코드는 [MANSY](https://github.com/duowuyms/MANSY_ImmersiveVideoStreaming) /
[NetLLM](https://github.com/duowuyms/NetLLM)의 viewport prediction 구현입니다.

> 통합 과정에서 발견한 버그, 설계 결정, 실험 결과의 전체 타임라인은
> [`report_assets/patch_selection_integration_summary.md`](report_assets/patch_selection_integration_summary.md)
> 를 참고하세요.

## 통합 파이프라인 구조

Patch selection(이미지 patch 결정, ViT 앞단)과 Selector(임베딩 시퀀스 축소, LLM 앞단)는
**서로 다른 단계에서 독립적으로 작동하는 병렬 컴포넌트**입니다. AdaLoRA는 **학습 시에만**
`plm` 내부에서 동작하고, Selector/speculative decoding은 **추론 시에만** 학습된 pipeline을
감싸는 wrapper로 동작하기 때문에 서로 다른 시점에 있어 코드상 충돌하지 않습니다.

```
history viewport(roll, pitch, yaw)
        │
        ▼
┌────────────────────────────────────────────┐
│ models/patch_selection.py                   │  궤적 → 4x4 grid 중 어떤 patch를 볼지 결정
│ PatchSelectionModule (사전학습, frozen)      │  (ViT 태우기 전, 연산 절감이 목적)
└──────────────────┬───────────────────────────┘
                   │ (patch-selection 모드일 때만)
                   ▼
┌────────────────────────────────────────────┐
│ models/pipeline.py :: Pipeline               │
│  get_multimodal_information()                │  baseline: 캐시된 ViT CLS 토큰 1개
│                                                │  all-patch: 전체 16 patch → ViT
│                                                │  patch-selection: 선택된 patch만 → ViT
│  [image_tokens] ++ conv1d/embed_vp(trajectory)│
│  → embed_ln() → plm (AdaLoRA/LoRA-wrapped)    │  AdaLoRA는 여기, 학습 시에만 동작
│  → KV-cached autoregressive loop              │
└──────────────────┬───────────────────────────┘
                   │ (추론 시에만, 학습된 pipeline을 감싸서 사용)
                   ▼
┌────────────────────────────────────────────┐
│ models/selectable_pipeline.py                │  embed_ln() 직후 시퀀스를 자름
│ LlamaSelectablePipeline + Selector            │  (RecentK 등, 이미지 토큰은 보호)
├────────────────────────────────────────────┤
│ models/speculative_pipeline.py                │  위 선택 로직 재사용 + KV 캐시로
│ LlamaSpeculativeBlockVerifyPipeline           │  draft-verify 블록 디코딩
└────────────────────────────────────────────┘
```

## 코드 구조

- `data/` — 데이터셋 / 사전학습·파인튜닝 체크포인트 저장 위치 (심볼릭 링크, git에는 미포함 — 아래 "데이터 & 가중치" 참고)
- `dataset/` — 데이터셋 전처리·로딩 (`preprocess.py`, `load_dataset.py`, `extract_saliency.py`, `extract_features.py`, `extract_features_cache.py`)
- `models/`
  - `gpt2.py, llama.py, opt.py, mistral.py` — 커스텀 LLM 래퍼
  - `low_rank.py` — LoRA/AdaLoRA용 low-rank 행렬 구현
  - `pipeline.py` — NetLLM의 기본 VP 파이프라인 (baseline / all-patch / patch-selection 3-way 멀티모달 지원)
  - `patch_selection.py` — 궤적 조건부 patch 선택 모듈
  - `selectors.py`, `selectable_pipeline.py` — 임베딩 시퀀스 축소(Selector, 예: RecentK) + 이미지 토큰 보호
  - `speculative.py`, `speculative_pipeline.py` — draft-verify 기반 speculative decoding
  - `regression.py, velocity.py, track.py` — 베이스라인 모델
  - `old/` — 논문 재현용 구버전 구현 (레거시)
- `utils/` — 로딩/지표 계산/정규화 등 유틸리티
- `analysis/` — 검증·벤치마크 스크립트와 결과물 (patch selection 동적 반응성 검증, KV-cache/speculative 벤치마크, 3-condition 평가 스크립트 등)
- `report_assets/` — 최종 보고용 그래프와 통합 결과 정리 문서
- `run_baseline.py` — 베이스라인 모델 실행
- `run_plm.py` — NetLLM(PLM 기반 VP) 학습/평가 메인 스크립트

## 3-Condition 비교 재현 방법

`run_plm.py`는 `--multimodal-mode` 플래그로 이미지 정보를 얼마나/어떻게 사용할지 선택합니다:

| 값 | 설명 |
|---|---|
| `baseline` | 캐시된 ViT CLS 토큰 1개만 사용 (프레임 전체 요약) |
| `all-patch` | 4x4 grid 전체 16개 patch를 모두 ViT에 통과 |
| `patch-selection` | `PatchSelectionModule`이 궤적 기반으로 선택한 patch만 ViT에 통과 (연산 절감) |

```bash
# 1) baseline
python run_plm.py --adapt --multimodal-mode baseline \
    --train-dataset Jin2022 --test-dataset Jin2022 \
    --plm-type llama --plm-size base --rank 32 --epochs 4

# 2) all-patch
python run_plm.py --adapt --multimodal-mode all-patch \
    --train-dataset Jin2022 --test-dataset Jin2022 \
    --plm-type llama --plm-size base --rank 32 --epochs 4

# 3) patch-selection (사전학습된 PatchSelectionModule 가중치 필요)
python run_plm.py --adapt --multimodal-mode patch-selection \
    --patch-selection-weights data/models/best_patch_selection.pth \
    --train-dataset Jin2022 --test-dataset Jin2022 \
    --plm-type llama --plm-size base --rank 32 --epochs 4
```

`multimodal_mode`는 체크포인트/결과 저장 경로에도 포함되므로 세 조건의 결과가 서로 덮어쓰지
않습니다. AdaLoRA를 함께 쓰려면 `--use-adalora` 플래그를 추가하세요 (`--rank`가 `target_r`이
되고, `init_r = rank * 2`에서 시작해 학습 중 pruning됩니다).

Selector / Speculative decoding은 학습된 pipeline을 감싸는 추론 전용 wrapper이며, 사용 예시는
`analysis/eval_3condition.py`, `analysis/verify_selectable_pipeline_equivalence.py`,
`analysis/verify_speculative_pipeline.py`를 참고하세요.

### 주요 결과 요약

- **Selector + Speculative 통합**: MAE -0.65%, latency -60.5% (2.53배 단축) — 전체 1,698 test
  샘플 기준. 상세는 `report_assets/selector_speculative_ablation.png`.
- **Patch Selection 모듈 검증**: F1 0.731, collapse 없음(평균 Jaccard 0.546), 이동 속도-선택
  patch 수 상관계수 r=0.55. 상세는 `report_assets/patch_selection_heatmap_reused.png`.
- **3-condition MAE**: LR 배수 조정에 따른 overshoot 진단 진행 중 — 자세한 내용과 현재 미해결
  항목은 `report_assets/patch_selection_integration_summary.md`의 "다음 단계" 참고.

## 데이터 & 가중치

아래 항목들은 용량이 크고(수GB~수십GB) 환경마다 경로가 달라서 이 저장소에는 포함하지
않았습니다 (`.gitignore` 참고). 직접 준비해서 아래 경로에 배치해야 합니다.

- `data/viewports/`, `data/images/` — 원본 viewport/영상 데이터셋
- `data/ft_plms/` — 파인튜닝된 LLM 체크포인트
- `data/models/` — 학습된 baseline / patch selection 모듈 체크포인트
- `../downloaded_plms/` — 사전학습 LLM 원본 가중치 (예: Llama2-7B)

## 환경

```
torch==2.2.0, numpy==1.24.4, munch==4.0.0, transformers==4.34.1, peft==0.6.2
```

추가로 필요한 패키지는 `requirements_extra.txt` 참고 (`torchvision`, `opencv-python`).
