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
  - `recipes/code/validate_resolved_config.py`가 bypass_mode false를 강제하던 것을 새 설정으로 교체

## 하지 않은 것

- GRS(오프라인 과제별 루브릭 + 채점 에이전트, Code 데이터에 루브릭 없음), GAR(규칙·상수는 GAGAR 논문 arXiv 2609.32577에 공개, SFT 채점 모델만 미공개. 공개 모델 채점기로 구현 가능), Sample Mixer(도메인 혼합, 도메인별로 따로 학습하면 불필요), 엔트로피 기반 IS 경계 조절(규칙 미공개), overlong 규칙(상수 미공개)
- webdev 무효 행 표시 (도메인 범위 밖)
- arvo·design 에이전트 루프에는 도구 실행 예산을 넣지 않음 (도메인 범위 밖, General과 같은 방식으로 옮기면 됨)

## 검증

- 새로 추가한 CPU 테스트: `test_advantage_fixes_on_cpu.py`(16), `test_exec_budget_on_cpu.py`(1), `test_bypass_prompt_mean_on_cpu.py`(3: 식 (1) 기울기·범위 밖 마스크·분모, 전부 마스크된 마이크로배치, 다른 손실의 prompt-mean 거부)
- 저장소 CPU 테스트 `tests/recipes tests/trainer/ppo tests/workers/config`를 원본(HEAD)과 비교: 원본 대비 새 실패 0건. 우리 쪽 실패 3건은 원본에서도 같은 실패 (로컬 모델 경로 없음 1, replay buffer DAPO 테스트 2). 원본 기본값을 고정해 둔 테스트 3개(trajectory_selection longest, bypass_mode false, DROP_INFRA_FROM_GROUP 게이트)는 바뀐 기본값에 맞춰 수정
- 실행: `PYTHONPATH=.:third_party/mimoagent-osr/src:third_party/uni_agent uv run --no-project --python 3.12 --with openai --with anthropic --with tenacity --with requests --with typer --with kubernetes --with xxhash --with TransferQueue==0.1.8 --with torch --with numpy --with pytest --with pytest-asyncio --with pydantic --with omegaconf --with tensordict --with packaging --with hydra-core --with codetiming --with ray --with transformers --with pillow --with pandas --with pyarrow --with datasets --with httpx --with cachetools --with uvicorn --with fastapi --with torchdata --with peft --with pyyaml --with jinja2 --with python-dotenv --with platformdirs --with rich python -m pytest -q tests/recipes tests/trainer/ppo tests/workers/config`
- 두 레시피 설정을 Hydra로 합성하고 액터 설정을 dataclass로 변환해 bypass 손실까지 값이 전달되는 것 확인
- General env actor의 도구 시간 누적·동결(Ray 액터)과 두 레시피의 예산 경로는 실제 pod에서 돌려보지 않았음
- **GPU 실행과 실제 TransferQueue 배치를 거친 `_compute_advantage`·bypass 손실 통합 경로는 검증하지 않았음**
