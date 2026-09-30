# MiMo verl 수정 사항 (브랜치 `mimo-fixes`)

대상: `verl/` 클론(a2ad9f6) + 서브모듈 `third_party/uni_agent`(c63e0b0). 근거는 [`code-analysis.md`](code-analysis.md) 13장. 리포트 역산 기록은 [`mimo-v2.6-rl-algorithm.md`](mimo-v2.6-rl-algorithm.md).

## 적용

이 브랜치(`mimo-fixes`)에 verl 수정이 커밋돼 있습니다. 서브모듈 uni-agent 수정은 패치로 들어 있고, 기본값이 꺼진 `timeout_as_failure` 옵션과 주석뿐이라 학습에는 필요 없습니다(선택).

```bash
git submodule update --init third_party/mimoagent-osr third_party/uni_agent
(cd third_party/uni_agent && git apply ../../patches/uni_agent-mimo-fixes.patch)
```

`git submodule update`는 서브모듈 수정 내용을 되돌리므로, 그 뒤에 uni-agent 패치를 다시 적용해야 합니다.

## 수학적 오류 수정

| # | 문제 | 수정 | 위치 |
|---|---|---|---|
| 1 | infra 행을 uid 재할당으로 분리 → 가짜 프롬프트가 prompt-mean 분모에 들어감 | 무효 행의 loss mask를 0으로, 프롬프트 수·토큰 수에서 제외 | `verl/trainer/ppo/advantage_fixes.py`, `trainer_base._compute_advantage` |
| 2 | sentinel/무효 행은 advantage만 0, 토큰은 T_q에 남음 → 해당 프롬프트만 가중치 감소 | 1과 같은 경로로 통합 | 같음 |
| 3 | arvo·webdev 기본값은 infra 실패를 reward 0으로 학습 (모델이 infra 탓에 벌받음) | `is_infra` 또는 sentinel 행은 항상 무효 처리 | 같음 |
| 4 | GRPO 평균 baseline이 자기 자신 포함 → 유효 행 수가 그룹마다 다르면 (1−1/n) 계수가 달라짐 | 그룹별 (1−1/n_ref)/(1−1/n_valid) 보정, 세션 단위로 계산 | `group_size_correction` |
| 5 | 무효 행 채움값을 행 평균으로 → 세션당 행 수가 다르면 GRPO 세션 평균과 어긋남 | 세션의 마지막 행(GRPO가 쓰는 행) 기준 평균 | `fill_invalid_scores` |
| 6 | 벽시계 시간 초과를 infra로 제외 → 모델 행동 기인 실패의 검열 편향 | 벽시계를 토큰 예산과 같은 "예산"으로 처리 (아래 절) | `recipes/general/{agent_loop,env_actor}.py`, `recipes/code/mimoagent_runner.py` |
| 7 | `apply_tool_penalty`가 `rebalance_dense`에 인자 하나를 더 넘김 → TypeError | 인자 수정, prompt-mean 가중치 전달 | `verl/trainer/ppo/arvo_penalties.py` |
| 8 | 예산에 잘린 궤적의 턴 구간이 응답 길이를 넘으면 예외 | 구간을 잘라 맞춤 | `recipes/*/trajectory_metadata.py`, `arvo_penalties.tool_error_hits` |
| 9 | General 레시피의 `algorithm.length_penalty`를 읽는 코드가 없음 (설정만 있고 무효) | 트레이너에 연결, 옛 키 이름(enable/deadzone/saturate)도 허용 | `advantage_fixes.length_penalty_config` |
| 10 | `group_advantage_by_harness`를 켜면 GRPO는 `uid::harness`로 묶는데 길이 페널티 기준은 `uid`로 잡음 | 모든 그룹 단계(GAR, 길이 페널티, 무효 행 채움, 크기 보정)가 GRPO와 같은 그룹 ID를 씀 | `trainer_base._compute_advantage` |
| 11 | bypass 손실은 유효 토큰 없는 마이크로배치에서 예외 | 손실 0, 같은 메트릭 키 | `core_algos.compute_policy_loss_bypass_mode` |

## 선택 가능하게 만든 것

원칙: 하이퍼파라미터와 verl 기본 기능 밖의 추가 기능은 설정으로 고를 수 있어야 한다. 기본값은 수학적으로 맞는 쪽, 원본 동작은 설정 한 줄로 되돌린다. 전체 표는 저장소 README.

- `algorithm.exclude_invalid_rows` (기본 true): 무효 행 제외. false면 원본처럼 sentinel 행만 baseline에서 빠지고 `is_infra` 행은 일반 행으로 학습
- 손실 방식: `algorithm.rollout_correction.*`와 `actor.policy_loss.loss_mode`. `bypass_mode`와 `loss_mode`가 어긋나면(한쪽만 바꾸면) 트레이너 시작과 Code preflight(`validate_resolved_config.py`)에서 막음. preflight는 더 이상 특정 손실 방식을 강제하지 않음. 원본 PPO로 되돌리려면 `bypass_mode=false`, `rollout_is=null`, `loss_mode=vanilla` 세 개를 모두 바꿈 (`rollout_is`가 남으면 decoupled 경로가 IcePop 가중치를 PPO 손실에 곱함)
- 도구 실행 예산: `exec_budget_seconds`, Code `exec_budget_agent_types`(null = 모델 기반 하니스), `exec_budget_probe_timeout`(30)
- `algorithm.gar.*` (아래 절)
- 제거: `DROP_INFRA_FROM_GROUP`. 트레이너가 더 읽지 않는데 실행 스크립트와 `scripts/*`가 기본으로 켜고 있었음. 이제 설정하면 실행 스크립트가 멈추고 `exclude_invalid_rows`를 쓰라고 알려 줌

## 시간 예산 처리 (6번의 설계)

원칙: 목표는 "주어진 예산 안의 성공"이고, 예산 끝의 상태를 채점하는 것이 그 정의 그대로다. 예산은 **정책이 책임지는 양**으로만 잰다. 벽시계는 학습 정지(colocate에서 학습 중 롤아웃 정지), 추론 대기열, 샌드박스 속도까지 세므로 정책 예산이 될 수 없다.

| 예산 (하이퍼파라미터) | 무엇을 세나 | 도달 시 |
|---|---|---|
| 토큰 `rollout.response_length` | 생성 토큰 | 멈추고 최종 상태 채점 (기존) |
| 스텝 `step_limit` | 모델 턴 | 같음 (기존) |
| **도구 실행 시간** Code `runner_kwargs.exec_budget_seconds`=3000, General `exec_budget_seconds`=300 | 에이전트가 실행한 pod 명령들의 실행 시간 합. 모델 호출 시간은 안 셈 | 동결 → 예산을 넘긴 명령은 끝까지 실행 → 생존 확인 → 최종 상태 채점. pod가 죽었으면 infra로 제외 |
| 벽시계 안전장치 Code 세션 7200초, General `trajectory_timeout` 1200초 (둘 다 원본 값) | 전체 경과 시간 | 멈춘 러너·샌드박스 감지용. infra로 제외. `wait`에서는 이 값이 학습 최대 정지 시간이 되므로 크게 잡지 않음 |
| Code 유휴 900초 | 모델 요청 없이 도구 하나가 멈춰 있는 시간 (학습 정지 중에는 안 셈) | 도구 먹통 감지. infra로 제외 |

- 동결: 다음 모델 호출과 다음 pod 명령이 `LimitsExceeded` → 에이전트 종료. 한 턴에 남은 도구 호출도 실행 안 됨. 예산을 넘긴 명령 하나만큼(최대 bash 600초) 초과 가능
- 채점과 경쟁 없음: 예산을 넘긴 명령이 반환된 뒤에야 동결이 적용되고 에이전트가 끝남
- 도구 실행 예산 값은 조정 가능한 하이퍼파라미터. `rollout/exec_budget_hit_rate`가 1% 안팎이 되도록 조정
- Code 블랙박스 하니스(claude code, codex): 모델 호출이 하니스 명령 안에 섞여 있어 시간을 나눌 수 없음 → 자체 `run_timeout` 유지, 평가 전용
- 도구 예산 크기: 예산 + 명령 하나 초과분(bash 최대 Code 600초, General 300초) = 벽시계 안전장치의 절반. 나머지 절반은 모델 생성·대기·학습 정지 몫. Code 3000+600=3600 (벽시계 7200), General 300+300=600 (벽시계 1200). `run_general.sh`가 `EXEC_BUDGET_SECONDS`·`TRAJECTORY_TIMEOUT`를 기본값과 함께 워커로 전달 (원본은 기본값이 없어 미설정 시 빈 문자열이 넘어가고 `float('')` 오류가 날 수 있음)
- staleness 한도 초과 처리 `max_off_policy_strategy`: `drop` → `wait`. drop은 정책 버전을 여러 번 걸칠 만큼 긴 롤아웃만 골라 버리는 길이 기반 검열. wait는 그 롤아웃이 끝날 때까지 다음 업데이트를 기다림 (처리량 손해, 편향 없음). 리포트 본 런은 expiry(버림)였던 것으로 보임
- `framework.timeout_as_failure`(기본 false): 자체 예산이 없는 러너용. 켜면 부분 궤적을 0점 처리 (uni-agent가 "채점하지 말라"고 적어 둔 abort 스냅샷을 씀)
- 첫 실행에서 확인할 지표: `rollout/exec_budget_hit_rate`, General `wall_backstop_hit`, uni-agent `num_failed_sessions`

## 리포트 충실도 (기본값 변경)

- tool call 오류 세그먼트 페널티: `adv_signed`, κ = 2, clamp 0.5~2.0 (Code는 uni-agent 마스크, General은 새로 기록한 턴 구간)
- 그룹 상대 길이 페널티: 레퍼런스 프로필 값 (X 0.2, p30, 통과율 > 0.5, γ 1.5, 턴·입력·출력 중 max)
- Code: n = 16, staleness 상한 4, `trajectory_selection: all` (기본 하니스는 세션당 궤적 1개라 현재는 차이 없음)
- 정책 손실을 리포트 식 (1)로: L = −Σ sg[π/μ]·M·A·log π. μ는 롤아웃 엔진 확률(재계산 안 함, partial rollout도 그대로), M은 비율이 [0.2, 5.0] 밖이면 0 (IcePop 방식, 마스크된 토큰도 prompt-mean 분모에는 남음). 원본 설정은 PPO clip 0.2 + dual-clip 3.0이었음. verl 기존 옵션 사용: `algorithm.rollout_correction` = bypass_mode true, loss_type reinforce, rollout_is token, rollout_is_threshold "0.2_5.0" / `actor.policy_loss.loss_mode: bypass_mode`. Code·General 둘 다
  - 리포트는 경계를 advantage 부호별로 나누고 엔트로피로 조절하지만 두 부호 모두 [0.2, 5.0]에서 시작하고 조절 규칙은 미공개 → 고정 범위 하나로 둠
  - 이를 위해 고친 코드: `ActorConfig`의 "prompt-mean은 vanilla만" 제약을 bypass_mode까지 허용 (둘 다 prompt 가중치를 agg_loss에 넘김). bypass 손실은 유효 토큰이 없는 마이크로배치에서 예외를 냈음 → 무효 행만 든 마이크로배치(긴 Code 궤적에서 흔함)가 학습을 멈출 수 있어, 손실 0과 같은 메트릭 키(값 0)를 내도록 수정 (`core_algos.compute_policy_loss_bypass_mode`)
  - `recipes/code/validate_resolved_config.py`가 bypass_mode false를 강제하던 것을, 두 손실 키가 서로 맞는지만 검사하도록 교체

## GAR (선택, 기본 꺼짐)

`verl/trainer/ppo/gar.py`, 설정 `algorithm.gar` (Code 레시피에 꺼진 상태로 있음). 리포트 4.3.2절과 GAGAR 논문(arXiv 2609.32577)의 식 6~8, 부록 A.1·A.2 그대로.

- 대상: 유효 세션 2개 이상, 통과(`score >= pass_threshold`, 기본 1.0)와 실패가 섞인 그룹. 세션은 마지막 행(GRPO가 쓰는 행)으로 판정
- GRPO 전: 채점기 호출 → 확정 hack은 reward 0 (`zero_confirmed_hacks`) → 그다음 길이 페널티와 GRPO
- GRPO 후: a = r − 평균, 통과는 a⁺ = max(a, 0), λ = min(Σa⁺ / Σf·a⁺, `lambda_max`), B = λ·f·a⁺(통과) 또는 a(실패), A = B − 평균(B). 세션의 모든 행에 적용. 그다음 그룹 크기 보정, 토큰 단위 페널티
- factor: T1 1등 1, T1 나머지 `f_runner` 0.9, T2 동률 그룹 순서대로 `f_max` 0.85 → `f_min` 0.4 선형, T3 `f_low` 0.2 (논문 Flash 설정)
- 채점기: `gar.grader.{path, name, kwargs}`로 불러오는 callable. `grade(groups: list[Group]) -> {group_id: GroupResult | None}`. `GroupResult(grades={session_key: Grade(tier, rank)}, hacks=[...])`. 통과 후보가 순위에서 빠졌거나 형식이 틀리면 그 그룹은 원래 advantage 유지 (`gar/groups_fallback`). 채점기가 예외를 내면 그 스텝의 모든 대상 그룹이 원래 advantage로 돌아감 (`gar/grader_error`)
- 조건: `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=false` (아니면 시작 시 오류)
- 지표: `gar/groups_eligible`, `gar/groups_graded`, `gar/groups_fallback`, `gar/confirmed_hacks`, `gar/tier_share_T{1,2,3}`, `gar/lambda_mean`, `gar/lambda_capped_rate`
- LLM API 채점기 `verl/trainer/ppo/gar_api_grader.py` (`APIGrader`, Code 설정의 기본 채점기): URL만 넣으면 OpenAI Chat Completions, OpenAI Responses, Anthropic Messages 중 하나로 호출. `run_train.sh`의 `GAR_ENABLE`, `GAR_GRADER_URL`, `GAR_GRADER_MODEL`, `GAR_GRADER_API`
  - 그룹마다 요청 1개 (병렬 `max_workers`, 429·5xx·타임아웃 재시도). 후보 순서는 그룹별 고정 시드로 섞고 C1, C2… 로 익명화. 실패 후보도 비교 맥락으로 넣음
  - 모델은 통과 후보마다 5개 기준 점수(1~5)와 플래그(요청 안 한 재작성, 테스트 맞춤 우회, 심각한 프로세스 문제, 미해결 회귀, hack)를 JSON으로 답함. 등급과 순위는 코드가 논문 A.1 규칙과 가중합으로 계산 (같은 점수는 동률). hack은 근거 문자열이 있어야 인정
  - 응답을 해석할 수 없으면 그 그룹만 fallback
  - 키: 트레이너 프로세스의 `GAR_GRADER_API_KEY` 또는 `api_key_file`. 설정·resolve된 설정 파일·실행 기록에 남지 않게 설정으로는 받지 않음
  - 채점 재료: 러너 결과의 `model_patch`, `test_output`, `result`, 그리고 `runner_kwargs.include_task_in_reward_info=true`일 때 과제 설명 (`GAR_ENABLE`이 같이 켬)
- 논문과 다른 점: 논문 채점기는 공개되지 않은 SFT 에이전트로 레포에 들어가 코드를 읽고 표적 테스트를 돌림. 이 채점기는 패치와 테스트 출력만 봄. 채점은 `_compute_advantage` 안에서 동기로 돌며, 논문처럼 롤아웃과 겹쳐 돌리려면 샘플러 쪽 작업이 필요

## 하지 않은 것

- GRS(오프라인 과제별 루브릭 + 채점 에이전트, Code 데이터에 루브릭 없음), 레포에 들어가는 에이전트형 GAR 채점기(위), 엔트로피 기반 IS 경계 조절(규칙 미공개), overlong 규칙(상수 미공개)
- Code와 General을 한 run에서 섞기와 Sample Mixer: 지금 Code는 uni-agent 어댑터가 롤아웃 매니저를 통째로 바꾸고 General은 verl AgentLoop라 한 run에 둘을 태울 수 없음. 공용 롤아웃 경로가 먼저 필요하고 실제 pod로 검증해야 하는 별도 작업
- webdev 무효 행 표시 (도메인 범위 밖)
- arvo·design 에이전트 루프에는 도구 실행 예산을 넣지 않음 (도메인 범위 밖, General과 같은 방식으로 옮기면 됨)

## 검증

- 새로 추가한 CPU 테스트: `test_advantage_fixes_on_cpu.py`(16), `test_exec_budget_on_cpu.py`(1), `test_bypass_prompt_mean_on_cpu.py`(3), `test_gar_on_cpu.py`(7: factor 표, 합·비율 보존, 상한과 재중심화, 보상 공간 등가식, 비이진 보상, 결과 검증, 설정 검증), `v1/test_compute_advantage_tq_on_cpu.py`(6), `test_gar_api_grader_on_cpu.py`(8: 로컬 가짜 서버로 세 API 형식 왕복·키 헤더·재시도, 등급 규칙, 응답 검증, fallback, 근거 없는 hack 무시)
- `test_compute_advantage_tq_on_cpu.py`는 실제 TransferQueue 파티션에 배치를 넣고 `_compute_advantage`를 끝까지 돌림: infra 행 마스킹, 여러 행 세션, 길이 페널티, 그룹 크기 보정, prompt-mean 가중치를 수식 값과 비교 / `exclude_invalid_rows=false`가 원본 동작 / 도구 오류 구간 페널티 / GAR hack 교정과 재분배 / 채점 결과 불량·채점기 예외 시 GRPO 그대로
- 저장소 CPU 테스트 `tests/recipes tests/trainer/ppo tests/workers/config`를 원본(`mimo-oss`)과 비교: 원본 535 통과·3 실패, 우리 580 통과·3 실패. 실패 3건은 양쪽 같은 테스트 (replay buffer DAPO 2, 로컬 모델 경로 1). 우리 쪽에만 있는 실패 0건
- 원본 기본값을 고정해 둔 테스트(trajectory_selection longest, bypass_mode false, DROP_INFRA_FROM_GROUP 관련 3곳)는 바뀐 기본값과 제거에 맞춰 수정
- 실행: `PYTHONPATH=.:third_party/mimoagent-osr/src:third_party/uni_agent uv run --no-project --python 3.12 --with openai --with anthropic --with tenacity --with requests --with typer --with kubernetes --with xxhash --with TransferQueue==0.1.8 --with torch --with numpy --with pytest --with pytest-asyncio --with pydantic --with omegaconf --with tensordict --with packaging --with hydra-core --with codetiming --with ray --with transformers --with pillow --with pandas --with pyarrow --with datasets --with httpx --with cachetools --with uvicorn --with fastapi --with torchdata --with peft --with pyyaml --with jinja2 --with python-dotenv --with platformdirs --with rich python -m pytest -q tests/recipes tests/trainer/ppo tests/workers/config`
- 두 레시피 설정을 Hydra로 합성해 손실 키 일치 검사 통과, 한쪽만 바꾸면 막힘, PPO 방식으로 되돌리기 가능 확인
- General env actor의 도구 시간 누적·동결(Ray 액터)과 두 레시피의 예산 경로는 실제 pod에서 돌려보지 않았음
- **GPU 실행(액터 손실까지의 전체 스텝)과 실제 pod는 검증하지 않았음**
