# Patch Selection 통합 프로젝트 종합 정리

작성일: 2026-08-14 (세션 로그 기반 최신화)

---

## 1. 프로젝트 개요 및 통합 파이프라인 구조

목표: NetLLM 기반 viewport prediction(VP) 파이프라인에 **patch selection**(궤적 조건부 이미지 patch 선택), 소윤 팀의 **Selector + Speculative decoding**, 수현 팀의 **AdaLoRA**를 하나의 파이프라인으로 통합하고, 각 컴포넌트가 실제로 작동하는지 + 서로 충돌 없이 결합되는지 + 방향성 있는 성능 개선을 보이는지 검증.

### 컴포넌트별 역할과 연결 지점

```
history viewport(roll,pitch,yaw)
        │
        ▼
┌────────────────────────────────────────────┐
│ models/patch_selection.py                    │  ← 궤적 → 어떤 4x4 grid patch를 볼지 결정
│ PatchSelectionModule (사전학습, frozen)       │     (ViT 태우기 전, 연산 절감이 목적)
└──────────────────┬───────────────────────────┘
                   │ (patch-selection 모드일 때만)
                   ▼
┌────────────────────────────────────────────┐
│ models/pipeline.py :: Pipeline                │
│  get_multimodal_information()                │  ← baseline: 캐시된 ViT CLS 토큰 1개
│                                                │     all-patch: 전체 16 patch → ViT
│                                                │     patch-selection: 선택된 patch만 → ViT
│  [image_tokens] ++ conv1d/embed_vp(trajectory)│
│  → embed_ln() → plm (AdaLoRA/LoRA-wrapped)    │  ← AdaLoRA는 여기, 학습 시에만 동작
│  → KV-cached autoregressive loop              │
└──────────────────┬───────────────────────────┘
                   │ (추론 시에만, 학습된 pipeline을 감싸서 사용)
                   ▼
┌────────────────────────────────────────────┐
│ models/selectable_pipeline.py                 │  ← embed_ln() 직후 시퀀스를 자름
│ LlamaSelectablePipeline + Selector             │     (RecentK 등, 이미지 토큰은 보호)
├────────────────────────────────────────────┤
│ models/speculative_pipeline.py                │  ← 위 선택 로직 재사용 + KV 캐시로
│ LlamaSpeculativeBlockVerifyPipeline            │     draft-verify 블록 디코딩
└────────────────────────────────────────────┘
```

**핵심 설계 원칙**: patch selection(이미지 patch 결정, ViT 앞단)과 Selector(임베딩 시퀀스 축소, LLM 앞단)는 파이프라인의 **서로 다른 단계에서 독립적으로 작동하는 병렬 컴포넌트**다. patch selection이 Selector 계약을 구현할 필요가 없다는 것을 Phase 0 조사에서 확인했고, 이후 실제로 이 방식으로 구현했다. AdaLoRA는 **학습 시에만** `plm` 내부에서 동작(rank-adaptive), Selector/speculative decoding은 **추론 시에만** 학습된 pipeline을 감싸는 wrapper로 동작 — 서로 다른 시점에 있어 코드상 충돌하지 않는다.

---

## 2. 통합 과정에서 발견한 버그/설계 결정 타임라인

| # | 시점 | 발견/결정 | 성격 |
|---|---|---|---|
| 1 | Phase 0 | Selector는 임베딩 후 시퀀스를 자르고, patch_selection은 ViT 전 이미지 patch를 고른다 — 같은 문제의 두 구현이 아니라 파이프라인의 다른 단계 | 설계 이해 정정 |
| 2 | Phase 0 | "speculative decoding은 제어흐름만, 가속 미구현"이라는 가정은 초기 프로토타입(`continuous_draft_verify.py`)에만 해당 — 최신 구현(`block_verify.py`)은 실제 checkpoint 기준 측정된 가속 존재 | 오래된 가정 정정 |
| 3 | Phase 0 → 5단계 | `SelectablePipeline`의 `using_multimodal` guard는 기술적 불가능이 아니라 Phase 3A 스코프를 좁히기 위한 안전장치였음(문서: `VP_EXTENSION_POLICY.md`) — 제거해도 안전함을 코드 레벨로 확인 후 제거 | 설계 결정 확인 후 guard 제거 |
| 4 | AdaLoRA 통합 | peft==0.6.2에서 `init_r == target_r`이면 `mask_to_budget()`의 `k = init_bgt - budget = 0`이 되어 `torch.kthvalue(k=0)`가 무조건 크래시 — 라이브러리 자체의 제약. 최소 여유(`init_r=target_r+1`)로 대체 검증 | 라이브러리 버그 발견 |
| 5 | AdaLoRA 통합 | AdaLoRA의 `target_r`은 레이어별 고정 하한이 아니라 전체 모듈에 걸친 평균 예산 — 실측으로 개별 레이어가 불균등하게 배분됨을 확인 | 동작 방식 확인 |
| 6 | 5단계 (Selector 이식) | RecentK류 selector가 시퀀스 앞쪽(이미지 토큰)부터 잘라먹을 수 있는 문제 — `protect_multimodal_prefix`로 이미지 토큰을 selector 후보 풀 밖에 둬서 대응 | 설계 보완 |
| 7 | 5단계 | `selector=None` 경로가 `Pipeline.auto_regressive()`와 `torch.equal` 완전 일치 확인 (baseline/all-patch/patch-selection 전부) | 동등성 검증 |
| 8 | 6단계 (Speculative 이식) | 우리 baseline은 이미 KV-cache됨(소윤 팀의 old-pipeline은 안 됨) — speculative decoding의 forward-count 절감폭이 소윤 리포트의 4-5배가 아니라 우리 기준으로는 다르게 나옴(원인 규명, 버그 아님) | 재현 안 됨 원인 규명 |
| 9 | 6→7단계 사이 | run_plm.py의 checkpoint 디렉토리 이름이 표준 LoRA와 AdaLoRA에서 동일해서, 같은 rank로 두 방식을 돌리면 서로 덮어썼을 것 — `_adalora` 접미사 추가로 수정 | 실제 버그, 실행 전 발견 |
| 10 | 7.1단계 (실 checkpoint 검증) | `peft_model()`이 1차원 파라미터(RMSNorm weight 포함)를 fp32로 올리는데, fp16 base와 섞이면 `LlamaRMSNorm.forward()`의 `weight * hidden_states`가 fp32로 프로모션되어 다음 fp16 Linear에서 크래시 — 이 프로젝트가 fp16을 실제로 끝까지 돌려본 적이 없어서 처음 걸린 버그. RMSNorm weight만 되돌리는 최소 패치로 해결 | 실제 버그, 실행 중 발견 |
| 11 | 7.1단계 | 실 checkpoint(`try_llama2_7b`)가 리네임 전 이름(`task_head`)으로 저장돼 있어서 `strict=False`로 조용히 스킵되던 것 발견 — 키 리매핑으로 실제 학습된 헤드 가중치 로드하도록 수정 | 실제 버그, 실행 중 발견 |
| 12 | 7.3단계 | patch-selection 모드의 latency 절감폭이 가장 작은 이유 조사 — 반복 호출/per-forward 비용 문제 아님(라이브 계측으로 배제), 미학습 patch selector 때문에 draft-target accept율이 낮아서 forward count 자체가 덜 줄어든 것 | 원인 규명(버그 아님) |
| 13 | patch-selection MAE 조사 | dropout이 테스트 시점에 켜져 있었는지 확인 → 아님(48개 서브모듈 전부 eval mode 확인). 별도로 `run_plm.py::test()`가 `pipeline.eval()`을 아예 호출하지 않는 실제 버그 발견(이번 MAE와는 무관, `--adapt --test` 동시 사용 패턴에 영향) | 가설 배제 + 별도 버그 발견 |
| 14 | patch-selection MAE 조사 | `conv1d.parameters()`가 `adapt()`의 optimizer 그룹에서 아예 누락 — trajectory encoder가 학습 안 된 채로 지금까지의 모든 MAE 숫자가 나왔다는 뜻. 실제 버그, 3개 모드 전부에 공통 영향 | 실제 버그, 발견 후 즉시 수정 |
| 15 | 버그 수정 | conv1d 누락 + test() eval() 누락 둘 다 수정, 회귀 테스트로 conv1d가 실제로 학습되는 것 확인(diff 0.0 → 0.00046) | 수정 완료 |
| 16 | LR 실험 | embed_multimodal(+conv1d)에 5~10배 높은 LR을 주는 실험 설계 — 14분 축소판(1000샘플/1epoch)에서 patch-selection ↔ all-patch 격차가 128.7%→80.0%로 축소되는 방향성 확인 | 가설 지지 증거 |
| 17 | LR×5 본실험 | 전체 데이터/2epoch/LR×5로 재학습 결과, **all-patch가 오히려 크게 악화**(23.92°→40.55°, +70%) — validation loss는 계속 개선되고 있었음에도 불구하고 | 예상 밖 결과, 즉시 조사 |
| 18 | all-patch 악화 원인 진단 | (1) valid-set MAE(38.14°)가 test-set MAE(40.55°)와 거의 동일 → 분포 차이 아니라 **loss 지표 자체가 MAE를 대변 못함**. (2) 오차가 균일하게 분포(median≈mean), outlier 아님. (3) embed_multimodal이 원래(6%)보다 훨씬 많이(28.8~30.5%) 이동 — LR 배수와 거의 비례 | 근본 원인 확정: LR×5 overshoot |
| 19 | patch-selection도 같은 메커니즘인지 확인 | embed_multimodal 이동량 24.9%(all-patch의 28.8%와 큰 차이 없음), valid MAE(48.57°)≈test MAE(49.69°) — 같은 metric-mismatch 패턴 확인. patch-selection의 "개선 추세"가 진짜인지 결론 유보, LR×2 재확인 필요 | 검증 필요 항목으로 확정 (LR×2 결과 대기) |

---

## 3. 성능 결과

### 3.1 Selector + Speculative Decoding 최종 결과

전체 1,698샘플 test set, baseline 모드(4-epoch, LR 부스트 없음) 체크포인트 기준:

| 구성 | MAE | forward count | latency |
|---|---:|---:|---:|
| A. 직접 (wrapper 없음) | 20.7582° | 20.0 | 413.22 ms/sample |
| B. Selector (RecentK k=6) | 20.5011° | 20.0 | 403.76 ms/sample |
| C. Speculative (γ=4, th=0.3) | 20.8754° | 6.58 | 173.49 ms/sample |
| **D. Selector + Speculative (통합)** | **20.6207°** | **6.50** | **163.35 ms/sample** |

**D(최종 통합 구성)가 MAE는 소폭 개선(-0.65%)하면서 latency는 2.53배 단축(-60.5%)** — 소윤 팀이 리포트한 것과 같은 방향의 결과를 우리 checkpoint/전체 test set 기준으로 재현.

(그림: `selector_speculative_ablation.png`)

### 3.2 Patch Selection 모듈 자체 검증 (사전학습 시점, 7/28)

**분류 성능** (`best_patch_selection.pth`, 최종 epoch 26/30, valid_loss 기준 best):
- Precision **0.747**, Recall **0.715**, **F1 0.731**, 평균 선택 patch 수 6.91/16

**동적 반응성 검증** (전체 1,698 test 샘플, `PATCH_SELECTION_VERIFICATION.md` 참조 — 반박 가능한 형태로 설계된 3단계 검증):
- **A. Collapse 여부**: 전체 평균 Jaccard **0.546** (collapse라면 ~1.0 기대) — collapse 아님, 영상당 40~52종의 서로 다른 선택 조합 확인
- **B. 궤적 반응성**: 현재 위치 patch 포함률 **92.99%**(우연 수준 ~44% 대비 크게 상회), 이동 속도-선택 개수 상관계수 **r=0.55**(정지 시 평균 4.91개 → 최고속 시 9.10개), 방향 anticipation은 반증되었으나 관성 기반 반응성 자체는 확인
- **C. 정적 사전지식**: 상하 극지방 4개 patch가 궤적 종류와 무관하게 항상 <5% 선택 — 360도 영상의 통계적 규칙성을 학습이 흡수한 것으로 해석

**결론**: patch selection 모듈은 collapse 없이, 궤적의 운동학적 상태(위치/속도)에 설명 가능한 방식으로 반응하며, 여기에 학습된 정적 우선순위(극지방 배제)가 결합된 하이브리드 구조로 작동한다. (그림: `patch_selection_heatmap_reused.png`)

### 3.3 VP 3-Condition MAE — LR Overshoot 진단까지

| 실험 | baseline | all-patch | patch-selection | patch-selection vs all-patch 격차 |
|---|---:|---:|---:|---:|
| 원래 4epoch, LR 부스트 없음 | 20.76° | 23.92° | 54.72° | +128.7% |
| 14분 방향확인 (1000샘플/1ep/LR×5) | 41.53° | 29.29° | 52.72° | +80.0% |
| **LR×5 (2ep, 전체데이터)** | 21.87° | **40.55° (악화)** | 49.69° | +22.5%\* |
| **LR×2 (2ep, 전체데이터)** | **결과 대기 중** | **결과 대기 중** | **결과 대기 중** | **결과 대기 중** |

\* LR×5 시점의 격차 축소(128.7%→22.5%)는 **patch-selection의 실질적 개선이 아니라 all-patch 자체가 크게 악화된 것에 의한 confound**로 진단됨 — 아래 참조.

**LR×5 overshoot 진단 결과**:
1. all-patch의 valid-set MAE(38.14°)가 test-set MAE(40.55°)와 거의 동일 → validation loss가 계속 개선되는 것처럼 보였던 것은 **분포 차이가 아니라 정규화 Tanh-space MSE 지표 자체가 실제 각도 오차(MAE)를 대변하지 못했기 때문**
2. 오차는 전체 test set에 걸쳐 균일하게 나쁨(median≈mean) — 소수 outlier 샘플이 평균을 끌어올린 게 아님
3. embed_multimodal 가중치가 원래(6% 이동)보다 훨씬 크게(28.8~30.5% 이동) 움직였음 — LR 5배와 대략 비례(4.8배)
4. patch-selection도 유사한 이동량(24.9%)과 동일한 valid≈test MAE 패턴을 보여, **patch-selection의 "개선 추세"가 진짜인지는 아직 확정할 수 없음** — LR×2 결과로 재확인 필요

**현재 상태**: LR×2(더 완만한 배수) 실험은 이 문서 작성 시점 기준 **아직 실행 전** — 다음 세션에서 진행 예정. (그림: `mae_comparison_4experiments.png`, `gap_trend_patchselection_vs_allpatch.png`, `loss_vs_mae_divergence.png`)

---

## 4. 다음 단계

1. **LR×2 재학습+평가** (baseline/all-patch/patch-selection, 2epoch, 전체 데이터, `--steps-per-valid 4410`) — all-patch가 overshoot 없이 개선되는지, patch-selection의 개선 추세가 진짜인지 동시에 확인
2. LR×2 결과에 따라: all-patch가 여전히 나쁘면 LR 배수를 더 낮추거나(1.5x 등) 다른 정규화 방법 고려. patch-selection이 진짜 개선이면 이 방향으로 본 학습 규모 확대 논의
3. (별도 과제, 이번 스코프 밖) `auto_regressive()`가 학습 모드에서 gradient checkpointing으로 인한 캐시 비활성화를 인지하도록 수정 — scheduled sampling 지원을 위해 필요
4. loss 지표(정규화 MSE)와 실제 MAE 사이 괴리 문제 자체가 일반적으로 재발할 수 있는지 — 학습 중 주기적으로 rotation-aware MAE도 같이 로깅하는 방안 검토

---

## 첨부 파일 목록

- `mae_comparison_4experiments.png` — 3-condition MAE, 4개 실험 grouped bar
- `gap_trend_patchselection_vs_allpatch.png` — 격차 추이 line chart
- `selector_speculative_ablation.png` — Selector+Speculative A/B/C/D ablation
- `patch_selection_heatmap_reused.png` — 4×4 grid 선택 빈도 히트맵 (사전학습 검증 시점)
- `loss_vs_mae_divergence.png` — 학습 지표 vs 실제 MAE 괴리 (핵심 발견 시각화)
