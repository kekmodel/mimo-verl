# XiaomiMiMo/verl 코드 분석 노트

대상: `verl/` (클론 a2ad9f6, 2026-09-26) + 서브모듈 `third_party/mimoagent-osr` (467f0a1), `third_party/uni_agent` (c63e0b0).
기준: upstream verl-project/verl `c16b7ee5` (2026-08-05)와 가장 가까움. 그 대비 변경은 `mimo-verl-changes.patch` (28파일, +2234/−80).

주의: 이 코드는 리포트 7장의 **9B 공개 재현판**입니다. Pro/Flash 30스텝 런의 설정이 아닙니다. `recipes/arvo/REFERENCE_PENALTIES.json`은 원본("reference RL framework")의 페널티 프로필이라고 스스로 밝히지만, 본 런과 같은 값인지는 확인되지 않습니다.

## 1. 트레이너 구조

- 기본 모드 `trainer.v1.trainer_mode=colocate_async` (`recipes/code/run_train.sh:281`). 모드는 `colocate_async` / `separate_async` / `sync` 셋 (upstream verl v1).
- `PPOTrainerColocateAsync` (`verl/trainer/ppo/v1/trainer_colocate_async.py`, upstream 파일):
  - `on_sample_end`: 미완료 요청 abort → 추론 복제본 sleep (가중치와 KV 폐기)
  - `on_step_end`: 가중치 갱신 → 생성 재개
  - docstring: "Trainer and rollout are colocated. Partial rollout is enabled."
  - → 앞서 추정한 "colocate + partial rollout"이 그대로 구현 이름.
- staleness 한도: `trainer.v1.sampler.max_off_policy_threshold` (code 2, arvo 8), 넘으면 `drop`. 리포트의 본 런 값은 4.
- prompt-mean이면 `ppo_mini_batch_size == 전체 배치` 강제 (`trainer_base.py` +788) → 배치당 optimizer 1스텝. 대시보드 `actor_optimizer_steps = 1`과 일치.

## 2. `_compute_advantage` 실행 순서 (trainer_base.py 패치)

1. tool_call_error / repetition 채널 준비. 전략 `mask`면 response_mask에서 제거.
   repetition `early_stop`: 첫 반복 이전 토큰은 mask(prefix), 반복 span은 signed로.
2. KL in reward (꺼짐)
3. **레퍼런스 길이 페널티** (`reference_penalties.shape_training_rewards`) — reward 단계, GRPO 전
4. webdev 그룹 보상 재작성 (`WEBDEV_GRADE_MODE`)
5. rollout correction (IS 가중치; bypass 아닐 때)
6. **infra 제외**: `is_infra` 롤아웃의 uid를 고유값으로 바꿔 그룹에서 분리 → advantage 0, 그룹 평균에 영향 없음
7. **GRPO advantage** (std 정규화 끔)
8. deep_failure_mask (실험용): 성공 궤적 턴 수 중앙값 × α 이후의 턴을 실패 행에서 loss 제외
9. adv_reduction / adv_set (채널별)
10. **signed rebalance** (κ = 채널별 최댓값, prompt-mean 가중치 반영)
11. 레퍼런스 tool 페널티 (`reference_penalties.apply_tool_penalty`)
12. prompt-mean 가중치 `1/(활성 프롬프트 수 × 그룹 토큰 수)` 계산해 저장

## 3. 확인된 구현

| 구성 | 위치 | 내용 |
|---|---|---|
| signed | `verl/trainer/ppo/signed_rebalance.py` | 양수 지목 → 0, 음수 → ×κ, α = 1 + removed/pos_clean (≤ max_scale), β = 1 − added/neg_clean (≥ min_scale). 분모는 지목 안 된 토큰. clamp되면 보존 안 됨. 행 부호는 유효 토큰 advantage의 중앙값 부호 |
| 지목 단위 | `arvo_penalties.tool_error_hits` | tool call 오류가 난 **모델 턴 전체**(segment)에 κ |
| 길이 페널티 | `verl/utils/length_penalty.py` | 통과(reward ≥ 0.5) 롤아웃만. 지표 = 턴 수 / 입력 토큰(prefill = 프롬프트+도구 출력) / 출력 토큰, 기준 = 통과 롤아웃의 분위수, excess = (v − anchor)/anchor, 지표 간 max. penalty = X · t^γ |
| prompt-mean | `core_algos.compute_prompt_loss_weights`, `agg_loss` | 행 가중치 1/(G · T_g), `masked_sum(loss·w) · dp_size` |
| infra sentinel | `core_algos.compute_grpo_outcome_advantage` | `invalid_reward_value`(원본 −999): 그룹의 유효 평균으로 대체 후 advantage 0 |
| DAPO 필터 | `replay_buffer._classify_group` | infra 제외한 유효 롤아웃이 0~1개이거나 전부 같으면 제외 |
| GRS | `mimoagent/environments/rubric_judge.py` | 판정 에이전트가 pod에 들어가 루브릭 판정. `solution_combine: product` → S·B, `combine: product`로 verifier와 곱 = R_test · S_sol · S_beh |
| 시각 그룹 채점 | `recipes/design/webdev/group_reward.py` | 그룹 전체 스크린샷의 상대 미감 pick − 절대 query-fit 감점. 무효 행은 그룹 평균으로 대체 |
| 반복 탐지 | `recipes/design/repetition.py` | 500단어 이상, 4-gram 고유 비율 < 0.15 |

## 4. REFERENCE_PENALTIES.json

- tool_call_error: `adv_signed`, segment, κ = 2, min_scale 0.5, max_scale 2
- length_penalty: X 0.2, δ 0, s 1, γ 1.5, 분위수 0.3, 통과율 문턱 0.5 (초과여야 적용), 지표 turns / input / output, combine max
- 공통: std 정규화 없음, prompt-mean, KL 없음
- **excluded_reference_rules: tool_name_invalid, toxic_reasoning, overlong, agent_context_filter** → 원본에 별도 overlong 규칙 존재. 대시보드의 −0.8 램프가 이것.

## 5. 공개판에 없는 것

- GAR (그룹 비교 grader, λ 재배분, hack → reward 0): 검색어 redistribut / groupwise / confirmed hack / stage_credit / select_* 모두 없음. 대시보드의 `stage_credit_group/*`, `select_*`, `tq_adv_*` 태그를 내보내는 코드도 없음
- Sample Mixer (소스별 할당량, 이월, deficit 스케줄링): 레시피가 단일 도메인이라 없음
- 엔트로피 기반 4방향 IS 경계: 없음. 공개 레시피는 `use_rollout_log_probs: true` + PPO clip 0.2 / dual-clip c 3.0
- overlong, toxic_reasoning 등 원본 규칙

## 6. 발견한 결함

`arvo_penalties.apply_tool_penalty`가 `rebalance_dense(advantages, hit, response_mask, invalid, ...)`로 4번째 인자를 위치 인자로 넘기지만, 함수는 위치 인자 3개만 받음 → `TypeError`. 실행해서 재현함. 즉 `algorithm.arvo_penalties.enable=true`면 첫 advantage 계산에서 죽습니다. 고쳐도 `row_weights`를 넘기지 않아 이 경로의 질량 보존은 prompt-mean 가중치가 아닌 토큰 기준입니다.

## 7. 공개 레시피 기본값 (code)

`adv_estimator: grpo`, `norm_adv_by_std_in_grpo: false`, `loss_agg_mode: prompt-mean`, `entropy_coeff: 0`, KL 없음, n 16, lr 1e-6, 배치 32, max_model_len 262,144, tool_call_error / repetition = monitor, deep_failure_mask 꺼짐, signed min/max 0.5 / 2.0.

## 8. 추가 확인

- `ppo_epochs: 1` (`verl/trainer/config/actor/actor.yaml:119`) + prompt-mean의 단일 minibatch 강제 → 배치당 optimizer 1스텝 확정.
- IS: 추가된 설정 필드에 엔트로피 기반 경계는 없음. upstream의 `rollout_correction`(`rollout_is_threshold` 기본 2.0 등)만 있고, 공개 레시피는 PPO clip 0.2 / dual-clip 3.0을 롤아웃 정책 대비로 씀. staleness가 있으면 clipfrac이 0이 아니게 되므로, 대시보드(pg_clipfrac = 0)의 본 런과 다른 메커니즘.
- 반복 탐지의 원본 이름은 `dirty_repetition` (n-gram이 window 안에서 min_repeat번). `early_stop`은 첫 반복 이전 토큰 mask + 반복 span에 signed κ, 세션 reward 0과 짝.
- deep_failure_mask: "biased estimator (음수 gradient 질량 제거), 깊은 턴의 엔트로피 증가 원인을 찾는 A/B용"이라고 명시. 대시보드의 r1(실패 롤아웃 약 6% 통째 마스킹)과는 다른 장치. r1은 여전히 미확인.
- 테스트: `tests/trainer/ppo/test_signed_rebalance_on_cpu.py`(8개)는 signed 함수만 검사. `arvo_penalties` 경로의 테스트는 없음. 어떤 레시피도 `algorithm.arvo_penalties.enable`을 켜지 않음(opt-in).
- GRS: 구현은 mimoagent에 있으나 RL 레시피 설정에서는 쓰지 않음. `example_configs/rubric-judge.yaml`에만 등장.
- `colocate_async`는 upstream verl 파일. MiMo 고유 구현이 아니라 그 위에서 기본값으로 택한 모드.

## 9. 환경 쪽 hack 방지 (mimoagent `environments/datasets/base.py`)

리포트 4.2.6절의 "환경 정리"가 코드로 있음. 주석의 `env_hacker audit`이 리포트의 hack agent.

- **git 유출 방지** (`_prevent_git_hack`, 모드 `strip` 기본 / `hide` / `none`)
  - strip: base로 detach → 모든 브랜치 삭제 → remote 제거 → replace/notes/stash/pull ref 삭제 → base의 조상이 아닌 태그 삭제(조상 태그는 버전 인식용으로 유지) → reflog 만료 → `gc.pruneExpire=now`, cruft pack 끄고 gc. 검증: base 이후 커밋이 0개가 아니면 실패 → hide로 후퇴
  - 주석에 과거 버그 두 개: 날짜 문자열 비교로 태그를 걸러 정답이 든 릴리스 태그가 살아남음, 기본 gc가 최근 unreachable 객체를 cruft pack에 남겨 `cat-file`로 복구 가능했음
  - hide: `.git`을 `/usr/lib/.<uuid>`로 옮겼다가 채점 때 복원. "root 에이전트가 `find / -name HEAD`로 찾을 수 있는 건 감수"
  - opensource_code 데이터셋은 이미지 빌드 때 이력을 잘라 두고, 셋업에서 잘렸는지만 검사
- **빌드 잔여물 정리** (`anti_hack_cleanup`, 기본 꺼짐, 데이터셋별 opt-in)
  - /tmp의 테스트 로그·패치·숨은 테스트 목록(`test_files.json`)·jest 캐시, `/tests`, `/logs`, `/var/log`, DB 로그
  - `git clean -fdx`로 빌드 산출물 삭제(언어별 의존성 캐시는 유지), Rust는 프로젝트 자체 crate의 rlib 삭제
  - Python: `pip install .`로 설치된 고쳐진 사본을 base 소스 editable 설치로 덮어씀
  - 레포 밖 캐시: `~/.m2` SNAPSHOT jar, Julia 컴파일 캐시, Gradle 빌드 캐시·데몬 로그, Bazel. 주석: "env_hacker audit이 찾은 가장 큰 유출 경로"
- **네트워크 차단**: `answer_leak_blocklist` 도메인을 `/etc/hosts`에서 0.0.0.0으로. 의존성 설치 뒤, 롤아웃 직전에 적용. 쓰기 실패 시 중단(fail closed)
- **채점 전 테스트 파일 리셋**: 테스트 패치가 건드리는 파일을 base 상태로 되돌리고, base에 없던 파일(에이전트가 심었을 수 있음)은 삭제한 뒤 패치 적용

## 10. 궤적 기록 (uni-agent `framework/framework.py`)

- tool call 오류: 러너(에이전트)가 턴마다 오류 여부를 분류하고, 게이트웨이가 기록한 `generation_spans`(모델 턴 토큰 구간)에 투영. 길이가 안 맞으면 추측하지 않고 마스크 0 (fail closed). 오류 종류: unknown_tool / invalid_arguments / incompatible_payload / other
- 반복 탐지 원본 기본값(`_check_repetition_tokens` 이식): 300토큰 n-gram이 8,300토큰 창 안에서 15회 이상, 턴 길이 16,384 이상일 때만 검사. 롤링 해시로 구현
- `turn_index`: 모델 턴 i의 토큰은 i, 관측·도구 출력은 −1
- 궤적마다 `min_global_steps` / `max_global_steps`(게이트웨이가 보고한 가중치 버전 범위) → staleness 판정
- 세션에 궤적이 여럿이면(서브에이전트, 압축) code 레시피는 `trajectory_selection: longest`로 가장 긴 것 하나만 학습
- 러너가 에이전트를 끝까지 못 돌려도(`termination_kind: truncated`) 최종 상태로 채점함 → 잘린 롤아웃도 성공 가능

## 11. 시각(webdev) 그룹 채점 — GAR과 가장 가까운 공개 코드

`recipes/design/grader_service/src/design_grader/service/group_pick.py`, `recipes/design/webdev/group_reward.py`

- 롤아웃마다 먼저 개별 채점: query 적합도(query_score)와 런타임 게이트(인라인 스크립트 문법 오류, 로딩 중 예외, 렌더러 무응답 → 0). 이 단계의 reward는 0 자리표시자
- 그룹이 다 모이면 `_compute_advantage` 안에서 그룹 채점으로 reward를 덮어씀 (GRPO 직전)
- 그룹 채점 = 스크린샷 전부를 한 번에 보여주고 "명백히 좋은 것 / 명백히 나쁜 것"만 고르게 함. 전순위를 매기지 않고, 차이가 없으면 "no clear difference"
  - 8라운드, Williams 라틴 방진 순서(모든 샷이 모든 위치에 한 번씩, 모든 인접 쌍이 한 번씩) → 첫 자리 편향 ±0.28 → ±0.03표
  - 라운드마다 good +1, bad −1, 그 외 0. pick_norm = 합 / 성공 라운드 수 ∈ [−1, 1]
  - 노이즈 제거: 성공 라운드 < 5면 그룹 pick 0, "차이 없음" 라운드 ≥ 2면 그룹 pick 0, 한 샷의 순표 |합| ≤ 1이면 그 샷 0
  - raw = pick_norm − query 감점 (query_score ≥0.9 → 0, ≥0.6 → 0.2, ≥0.4 → 0.4, ≥0.2 → 0.8, <0.2 → reward 0), 런타임 게이트 실패 → reward 0
  - reward = (raw + 2)/3 ∈ [0, 1] (GRPO는 그룹 평균만 빼므로 이동은 상쇄, 스케일만 1/3)
- 그룹은 8개씩 잘라 채점(`WEBDEV_GROUP_SLICE=8`), 공개 레시피는 n = 8. 무효 행은 그룹 유효 평균으로 대체(= advantage 0)
- 판정 모델 하나(`GRADER_MODEL`)를 query와 pick에 같이 씀. 프롬프트는 중국어(`prompts/group_pick.md`): "보통 사람이 한눈에 가리킬 수 있는 차이만", "숫자 맞추려 중간을 고르지 말 것"
- 리포트 4.2.3절의 "개별 루브릭으로 기준을 다진 뒤 그룹 비교로 강약을 가림"이 이 구조

## 12. 공개 데이터셋 (HF `XiaomiMiMo/MiMo-V2.6-RL-oss`, `hf-data/`에 parquet 저장)

| 도메인 | 행 수 | 내용 |
|---|---|---|
| code | 2,698 | 전부 `opensource-code`, 도커 이미지 `format-code-task-*`, `test_patch` + `test_command`(bash), verifier 제한 1,800초. 문제 설명 중앙값 1,603자, 테스트 패치 중앙값 161줄 |
| cyber | 1,000 | ARVO(OSS-Fuzz). 프롬프트 = "AddressSanitizer: heap-buffer-overflow in function X in file Y" 형태 |
| general | 989 | 지식 노동 925(32개 직무 영역 × 영/중, 회계·세무, 금융·보험, 의료 운영, 컨설팅, HR, 공공, IT 등) + terminal_bench 64 |
| webdev | 2,093 | 웹사이트 제작 요청 |
| music | 1,000 | ABC 기보 작곡, 조성·BPM·박자·편성·마디 수 조건 |

리포트 7.2절 표 5의 규모(코드 3k, 보안 1k, 범용 1k, 시각 2k, 음악 1k)와 일치.

범용 환경 한 개(`general/envs/<task>/`) 구성: `instruction.md`, mock 소프트웨어 도구(`tools/*.py`, MCP 브리지), 시스템별 SQLite 상태 DB(`system/*/state.db`), 작업 파일(pptx, xlsx, docx, pdf, html), `verifier_meta.json`, `verify.py`.
- 채점 항목: 항목마다 tier(critical / important), method(llm / rule), weight, pass_anchor(정답 근거). 예시 과제는 LLM 항목 5개, 가중치 0.2 / 0.2 / 0.25 / 0.2 / 0.15
- 점수 = Σ w·s / Σ w (연속값). 규칙 gate가 하나라도 실패하면 0점이고 LLM 판정은 건너뜀
- **원본 보존 gate**(`src_protect`): 풀이가 허용된 범위 밖의 파일이나 DB 테이블을 건드리면 과제 전체 0점. 리포트의 "관계없는 파일·DB 변경을 잡는 부정 검사"
- 판정 모델에 전혀 닿지 못하면 0점이 아니라 mask. 여러 모델을 순서대로 시도하는 fallback, 판정 단계 전체에 600초 예산

## 13. mask 처리의 수학적 문제 (공개 코드 기준)

목적 함수(prompt-mean): 유효 프롬프트 집합 Q, 프롬프트 q의 유효 행 V_q, 유효 토큰 수 T_q에 대해
L = (1/|Q|) Σ_q (1/T_q) Σ_{i∈V_q} Σ_t ℓ_{i,t}

1. `DROP_INFRA_FROM_GROUP` (uid 재할당): 실패 행이 가짜 프롬프트로 |Q|에 들어감 → gradient = |Q|/(|Q|+k_t) · ∇L, k_t는 스텝마다 다른 실패 수 → 스텝별 무작위 스케일. 수정: 해당 행 loss_mask = 0 후 가중치 계산(프롬프트 수에서 제외)
2. `invalid_reward_value` / webdev 무효 행: advantage만 0이고 토큰은 T_q에 포함 → 실패가 있는 프롬프트만 가중치가 작아지는 상대 편향. 수정: 가중치 계산 전에 loss_mask = 0
3. GRPO 평균 baseline이 자기 자신을 포함: E[(r_i − r̄)∇log π_i] = (1 − 1/n)∇J_q. mask로 그룹별 유효 n이 달라지면 계수가 그룹마다 달라져 상대 편향. 수정: leave-one-out baseline(RLOO) 또는 n/(n−1) 보정
4. `rollout/seq_timeout`(궤적 벽시계 초과)을 infra로 분류: 모델 행동이 원인일 수 있는 실패를 학습에서 제외 → 조건부 추정 E[∇ | 시간 초과 아님] ≠ ∇J. 시간을 오래 끄는 행동이 음수 신호를 피함. 수정: 모델 기인 시간 초과는 실패로 채점. pod 연결 끊김 등도 궤적 길이와 상관되면 같은 종류의 편향
(참고: `invalid_reward_value`를 유효 평균으로 채우는 것 자체는 std 정규화를 끈 상태에서는 다른 행의 advantage를 바꾸지 않음. std 정규화를 켜면 std가 줄어 틀려짐)
