# KV 캐시 적용 범위 (코드 레벨 명세)

이 문서는 "KV 캐시가 정확히 어디에 적용되는가"에 대한 질문에 코드 레벨로 답한다.
결론부터: **KV 캐시는 `Pipeline.auto_regressive()` 단 하나의 함수에만 적용되어
있고, `models/llama.py`는 이 최적화를 위해 변경된 적이 없다.** 그리고 이 최적화는
`multimodal_mode`(`baseline`/`all-patch`/`patch-selection`) 중 어느 것을 골라도
동일하게 적용되는, 모드에 무관한 인프라 레벨 최적화다.

## 1. 적용 위치: `auto_regressive()`만, `teaching_forcing()`은 아님

`models/pipeline.py`의 `Pipeline` 클래스에는 시퀀스를 만드는 두 개의 경로가 있다
(`forward()`, `models/pipeline.py:128-141`가 `teacher_forcing` 플래그로 분기):

| 경로 | 언제 쓰이나 | KV 캐시 | 이유 |
|---|---|---|---|
| `teaching_forcing()` (`pipeline.py:186-212`) | **학습(training)** — ground-truth future를 통째로 이어붙여 **1회의 forward pass**로 전체 시퀀스를 처리 | **적용 안 됨 / 적용할 필요 없음** | 애초에 반복 루프가 없다. `history + future`를 한 번에 `self.plm(inputs_embeds=x, ...)`에 넣고 끝난다 (`pipeline.py:211`). 캐시로 절약할 "다음 스텝"이 없으므로 최적화 대상 자체가 아니다. |
| `auto_regressive()` (`pipeline.py:143-184`) | **추론(inference)** 및 auto-regressive 방식 학습 — `fut_window`(=20) 스텝을 하나씩 생성 | **적용됨** | 매 스텝 전체 시퀀스를 처음부터 재계산하던 O(fut_window²) 낭비를 없애기 위해 도입 (§3). |

`inference()`(`pipeline.py:214-220`)는 항상 `auto_regressive()`를 호출하므로,
실제 서비스 추론 경로는 전부 캐시의 혜택을 받는다. 학습 경로 중 teacher forcing을
쓰는 쪽(`forward(..., teacher_forcing=True)`, 기본값)은 애초에 스텝 루프가 없어
KV 캐시가 관여할 지점이 없다.

## 2. 어떤 파일/클래스가 바뀌었나

| 파일 | 변경 여부 | 내용 |
|---|---|---|
| `models/pipeline.py` | **변경됨** | `Pipeline.auto_regressive()` 내부 루프만 수정. `past_key_values`/`use_cache=True`를 도입하고, 매 스텝 입력을 "누적 전체 시퀀스"에서 "새로 생성된 토큰 1개"로 축소 (`pipeline.py:170-181`). |
| `models/llama.py` (`LlamaNetworkingHeadModel`) | **변경 안 됨 (이 최적화 관련해서는)** | `forward()` 시그니처는 처음부터 `past_key_values: Optional[List[torch.FloatTensor]] = None`과 `use_cache: Optional[bool] = None`을 받아 그대로 내부 `self.model(...)`(HuggingFace `LlamaModel`)에 전달하고 있었다 (`models/llama.py:29-59`). 즉 KV 캐시를 "받아서 흘려보내는" 배관은 이 클래스가 `LlamaForCausalLM`을 상속한 시점부터 이미 존재했고, 이번 최적화는 그 배관에 실제로 `past_key_values`를 채워 넣기 시작한 **호출부(`pipeline.py`)만** 바꾼 것이다. `git diff models/old/llama.py models/llama.py`로 실제 변경 이력을 확인한 결과, 이 파일에서 있었던 변경은 (a) `task_head` → `networking_head` 이름 변경, (b) `hidden_states`를 fp16→fp32로 명시적으로 캐스팅한 것 두 가지뿐이며 KV 캐시 로직과는 무관하다. |
| `models/gpt2.py`, `models/opt.py`, `models/mistral.py` | 변경 안 됨 | 같은 이유로 이 세 파일도 각자의 HF 베이스 클래스로부터 `past_key_values`/`use_cache` 배관을 이미 상속받고 있었을 가능성이 높지만, 현재 `run_plm.py`/벤치마크는 `llama`로만 실행되었으므로 이 문서는 llama 경로만 검증했다. |

## 3. `past_key_values` 생성/전달 흐름

### 수정 전 (매 스텝 전체 재계산, `models/old/pipeline.py` 기준)

```python
x = <history 토큰 + (있다면) 이미지 토큰>          # 초기 시퀀스, 길이 L0
outputlist = []
for step in range(fut_window):                     # 20회 반복
    outputs = plm(inputs_embeds=x,                  # 매번 길이가 L0+step인 전체 시퀀스를 통째로 forward
                   attention_mask=ones(L0+step))
    outputlist.append(outputs.logits)
    next_token = embed_vp(conv1d(outputs.logits))
    x = concat([x, next_token])                     # 시퀀스가 계속 길어짐 -> 다음 스텝에 또 전체 재계산
```
스텝 `t`에서 시퀀스 길이가 `L0+t`이므로 총 연산량은 `sum_{t=0}^{19}(L0+t)` ~
`O(fut_window^2)`. 토큰 하나하나의 hidden state는 causal attention 하에서 그
이전 토큰들에만 의존하는데도, 이전 스텝에서 이미 계산했던 부분까지 매번 다시
계산하는 구조였다.

### 수정 후 (`models/pipeline.py:170-181`, 현재 코드)

```python
past_key_values = None
current_input = x            # 첫 스텝만 초기 시퀀스 전체 (길이 L0)
total_len = x.shape[1]
for step in range(fut_window):                       # 20회 반복
    attention_mask = ones(total_len)                 # "지금까지의 실제 길이" 기준 마스크
    outputs = plm(inputs_embeds=current_input,        # 스텝 0: 길이 L0 / 스텝 1+: 길이 1
                   attention_mask=attention_mask,
                   past_key_values=past_key_values,   # 이전까지의 K/V 캐시를 그대로 재사용
                   use_cache=True)                    # 새로 계산한 K/V도 캐시에 이어붙여 반환하라는 신호
    logits = outputs.logits
    outputlist.append(logits)
    past_key_values = outputs.past_key_values          # 다음 스텝을 위해 갱신된 캐시를 들고 감
    current_input = embed_vp(conv1d(logits)).unsqueeze(1)  # 다음 스텝엔 새 토큰 1개만
    total_len += 1
```

다이어그램으로 보면:

```
step 0: input=[t0 t1 ... t_{L0-1}]                    -> plm -> KV_cache_0 (L0개 토큰의 K/V)
                                                                  |
step 1: input=[t_L0]  (새 토큰 1개) + KV_cache_0 재사용 -> plm -> KV_cache_1 (L0+1개 토큰의 K/V)
                                                                  |
step 2: input=[t_{L0+1}] (새 토큰 1개) + KV_cache_1 재사용 -> plm -> KV_cache_2
                                                                  |
                                                                 ...
step 19: input=[t_{L0+18}] (새 토큰 1개) + KV_cache_18 재사용 -> plm -> KV_cache_19
```

각 스텝은 새 토큰 1개에 대해서만 self-attention의 query를 계산하고, key/value는
캐시에 있는 것 + 새로 계산한 것 1개만 사용한다. 그 결과 스텝당 비용이 `O(1)`
토큰-forward가 되어 전체 루프가 `O(fut_window)`로 줄어든다. 수학적으로는 "매
스텝 전체 재계산"과 동일한 결과를 내는데(causal masking이 있으므로 이전
토큰들의 hidden state는 이후 토큰이 추가되어도 바뀌지 않는다), 중복 계산만
제거한 것이다. 이 수치적 동치성은 `torch.allclose(..., atol=1e-2)`로
실측 검증됐다(§4의 신규 벤치마크에서도 재검증, `max_abs_diff` 리포트 참고).

## 4. 세 조건(baseline/all-patch/patch-selection) 모두에 공통 적용되는가?

**그렇다.** `auto_regressive()`는 `multimodal_mode`를 전혀 참조하지 않는다 — 이
함수가 하는 일은 (1) `self.using_multimodal`이 켜져 있으면
`get_multimodal_information()`을 호출해 이미지 토큰을 앞에 붙이고 (`pipeline.py:157-158`),
(2) 그 뒤로는 완전히 동일한 KV-캐시 루프를 돈다 (`pipeline.py:162-184`).
`get_multimodal_information()`이 내부적으로 `baseline`/`all-patch`/`patch-selection`
중 어디로 디스패치되는지(`pipeline.py:237-260`)는 **루프가 시작되기 전에 이미
끝나는 일**이라, 이미지 토큰이 몇 개 붙든(`baseline`=1개,
`all-patch`=16개, `patch-selection`=가변 k개) 그 뒤 20-스텝 auto-regressive
루프의 캐시 로직에는 영향을 주지 않는다.

즉 KV 캐시는 "patch-selection 전용 최적화"가 아니라 **auto-regressive 디코딩
루프를 쓰는 모든 실행 경로에 적용되는 인프라 레벨 최적화**이며, 세 조건 모두
같은 코드 경로(`Pipeline.auto_regressive()`)를 통과하므로 동일하게 그 효과를
받는다. 실측으로도 이는 확인된다 — `IMPLEMENTATION_NOTES.md` §3의 표에서
baseline(7.9%), all-patch(9.6%), patch-selection(8.4%) 세 모드 모두
비슷한 폭(8~10%)의 개선을 보였다는 것이, 캐시 효과가 이미지 토큰 처리 방식과
무관하게 일관적으로 나타난다는 것의 실증이다.

## 5. 요약

- KV 캐시 적용 함수: `Pipeline.auto_regressive()` (`models/pipeline.py:143-184`) **1곳뿐**.
- `teaching_forcing()`(학습 경로, `pipeline.py:186-212`)은 스텝 루프가 없어 애초에
  최적화 대상이 아니며, 실제로 캐시가 적용되어 있지 않다.
- `models/llama.py`의 `LlamaNetworkingHeadModel`은 `past_key_values`/`use_cache`
  파라미터를 이미 받고 있었지만(HF `LlamaForCausalLM` 상속), 이번 최적화를 위해
  이 파일 자체가 수정된 적은 없다 — 실제 변경은 전부 호출부인 `pipeline.py`에서
  일어났다.
- 이 최적화는 `multimodal_mode`(baseline/all-patch/patch-selection)에 무관하게
  **세 조건 모두에 동일하게 적용**되는 공통 최적화다. 특정 모드에만 배타적으로
  걸려 있는 것이 아니다.
