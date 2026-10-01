# 다음 작업 설계: Code + General 혼합 학습과 Sample Mixer

상태: 설계 (구현 전). 기준 브랜치 `mimo-fixes`. 리포트 6.2 (multi-tenant rollout, heterogeneous harnesses), 6.3 (Sample Mixer).

## 1. 지금 왜 못 섞는가

| | Code | General |
|---|---|---|
| 롤아웃 매니저 | uni-agent `AgentFrameworkRolloutAdapter`가 `agent_loop_manager_class`를 통째로 교체 | verl 기본 `AgentLoopManager` + `GeneralAgentLoop` |
| 모델 호출 | 하니스 → uni-agent 게이트웨이(OpenAI 호환 HTTP) → 추론 엔진. 게이트웨이가 토큰을 기록 | `_VerlRolloutModel`이 추론 엔진을 직접 호출하고 토큰을 직접 기록 |
| 환경 | mimoagent `make_dataset_env(instance)` | 같은 함수, Ray 액터(`DatasetEnvActor`) 안에서 |
| 하니스 | 여러 개 (bashonly, mimocode, cc, codex …, `mix-four-whitebox.yaml`) | mimoagent `DefaultAgent` 하나 |

한 트레이너에는 롤아웃 매니저가 하나뿐이라 두 경로를 동시에 쓸 수 없다.

## 2. 선택지와 결정

- **A. General을 uni-agent의 두 번째 러너로 옮긴다 (권장).** 두 데이터 모두 같은 행 형식(`instance_json`, `dataset_type`)이고 같은 `make_dataset_env`로 환경을 만든다. Code의 멀티 하니스와 블랙박스 하니스를 그대로 유지한다. 리포트 6.2의 "하나의 실행 프레임워크 안의 여러 하니스"와도 같은 구조.
- B. Code를 verl AgentLoop로 옮긴다. 블랙박스 하니스(claude code, codex)는 HTTP 게이트웨이가 필요해서 잃는다. 기각.

## 3. General 러너 이식 (A)

uni-agent는 `agent_runners`를 여러 개 받고 행의 `agent_name`으로 고른다. 두 데이터 모두 `agent_name: mimo_swe_agent`라서 구분이 안 된다.
→ **로드 시점에 `data_source`로 러너를 정하는 매핑**을 어댑터 설정에 추가 (`runner_by_data_source: {"mimoagent/general_agent": general, "mimoagent/terminal_bench": general, "opensource-code": code}`). 데이터 파일을 고치지 않아도 되고, 매핑에 없는 소스는 시작 시 오류.

게이트웨이 확인 결과 (코드 확인):
- 요청의 `tools`를 그대로 chat template에 넘김 → 인스턴스마다 바뀌는 MCP 도구도 됨
- 도구 오류 마스크는 러너의 `tool_call_error_flags`와 게이트웨이의 생성 구간을 맞춰 만들고, 개수가 다르면 그 행은 0 (Code와 같은 동작). General도 `mask_source: field`로 통일
- 턴 수는 게이트웨이 값 대신 트레이너가 마스크의 연속 구간으로 셈 (이미 구현)

옮길 것 (`recipes/general/` → 새 러너 `recipes/general/uni_runner.py`, Code 러너와 공통 부분은 함수로 공유):

| 항목 | 지금 위치 | 옮긴 뒤 |
|---|---|---|
| mimoagent 설정 (`config/agent/general/s3k.yaml`) | `GeneralAgentLoop.__init__` | 러너 `config_path` |
| MCP 도구 탐색 (`mcp_proxy.discover_mcp_tools`) | `DatasetEnvActor._create` | 러너에서 환경 생성 직후 (예산 게이트 설치 전이라 예산에 안 들어감) |
| General 전용 도구 6개 (`recipes/general/tools`, `register_cc_tools`) | 모듈 import | 러너에서 등록 |
| 보상 이진화 `REWARD_BINARIZE` | `env_actor` 채점 뒤 | 러너 채점 뒤 (소스별 보상 변환 설정) |
| 채점 판정기 `GA_JUDGE_*` | 환경 변수 → `calculate_reward` | 같음 (Ray runtime env로 전달) |
| 도구 실행 예산 | `env_actor` | Code 러너의 `_ExecBudget` 재사용, 소스별 `exec_budget_seconds` |
| infra 실패 | `_failure_output` (is_infra 행) | uni-agent가 세션을 버림 → 배치에 없음 (그룹 크기 보정이 처리) |
| `length_signals`, 턴 구간 | 에이전트 루프가 기록 | 트레이너 대체 경로 (원래 마스크 기준) |

검증이 필요한 것 (pod 필요): General 과제를 두 경로(GeneralAgentLoop, uni-agent 러너)로 같은 시드에서 돌려 보상 분포와 토큰 수가 같은지. 게이트웨이의 토큰 기록이 General의 `chat_delta` 처리와 같은 토큰열을 만드는지가 핵심.

## 4. Sample Mixer (리포트 6.3)

목표: 소스별 실행 시간이 수십 배 달라도 매 학습 배치를 목표 비율로 채운다.

| 메커니즘 | 이번 범위 | 내용 |
|---|---|---|
| 적응형 동시성 (식 6) | 넣음 | 소스 i: 목표 `B_i`(배치당 채택 그룹 수), 채택률 `r_i`, 활성 시간 `t_i`. 수요 `m_i = B_i / r_i`, 예산 `(1 + p_i) m_i`, `p_i = clip(c·t_i − 1, p_min, p_max)`, `Σ m_i p_i / Σ m_i = p̄`가 되게 `c`를 이분 탐색 (단조) |
| 적응형 스케줄링 (식 7) | 넣음 | 예산 안에서 `w_i = α B_i / r_i + (1 − α)(B_i − A_i)⁺ / r_i` (α = 0.5)로 smooth weighted round-robin. `A_i`는 이번 배치에 이미 채택된 수 |
| 예측 배치 (KV 수요 기반 rank 배정) | 뺌 | 추론 엔진 내부 작업. verl 롤아웃 매니저의 기본 부하 분산 사용 |
| 샘플 재사용 (시작·복구 직후) | 뺌, 나중에 | 같은 시작 체크포인트의 저장된 롤아웃이 있어야 함 |

구현 위치:
- **프롬프트 공급**: 소스별 데이터로더(`StatefulDataLoader` 소스당 하나). 트레이너의 보충 경로(`_add_prompts_to_generate`, 시작 시 워밍업)가 소스를 식 7로 골라 꺼냄. 진행 중 + 채택 후 대기 그룹 수가 예산을 넘는 소스는 꺼내지 않음
- **배치 조립**: `ReplayBufferAsync` 하위 클래스(`trainer.v1.sampler.custom_sampler`). 소스마다 `B_i`개를 오래된 것부터 채우고, 넘치는 것은 다음 배치로 이월. 모든 소스가 몫을 채워야 배치를 냄
- **통계**: `r_i`, `t_i`의 지수 이동 평균. 체크포인트에 저장
- **`t_i` 측정**: 지금은 어디에도 없음. uni-agent 세션 시간은 colocate 학습 정지 시간을 포함한다. 트레이너가 정지 구간(`on_sample_end` → `on_step_end`)을 기록하고, 세션 시작·끝 시각에서 겹치는 정지 구간을 빼서 계산. 이게 없으면 식 6의 예산이 정지 비율만큼 틀림
- **설정**: `trainer.v1.sampler.mixer.{enable, sources: {이름: {data_sources: [...], weight}}, p_mean, p_min, p_max, alpha, ema}`. 기본 꺼짐

## 5. 수학적 주의점

1. **채택 기준 목표는 소스를 `1/r_i`로 재가중한다.** 동적 샘플링은 성공·실패가 섞인 그룹만 남긴다. 버려진 그룹은 기울기 0이므로, 소스 i의 생성 프롬프트 전체에 대한 기울기는 `r_i × (채택 그룹 평균)`. 채택 그룹 수 `B_i`를 고정하고 prompt-mean으로 평균하면 실제로 최적화하는 것은 `Σ_i (B_i / (r_i·B)) ∇J_i`. 채택률이 낮은 소스가 목표보다 크게 반영되고, `r_i`가 학습 중에 바뀌면 비율도 같이 흔들린다. 리포트는 `B_i`를 "채택 그룹 수"로 정의하므로 리포트에는 충실하지만, 명시한 도메인 비율과는 다르다 → 결정 1
2. **이월은 버리지 않는다.** 소스별 몫이 생기면 "넘치는 그룹을 버리는" 새 경로가 생길 수 있다. 버리면 늦게 끝나는(긴) 그룹이 빠지는 길이 검열이 된다. 이월은 오래된 것부터 쓰고 절대 버리지 않으며, 넘침은 예산에서 막는다 (예산 초과 소스는 새 프롬프트를 안 꺼냄)
3. **`rollout.n` 통일.** 그룹 크기 보정은 `n_ref = rollout.n` 하나를 쓴다. Code 16, General 8을 섞으면 General 그룹이 매번 보정된다 → 결정 2
4. **staleness `wait`과 소스 결합.** 느린 소스의 오래된 롤아웃이 한도에 닿으면 전체 학습이 기다린다. 편향은 없고 처리량 문제. 예산이 이를 줄인다

## 6. 검증 단계

1. CPU: 가짜 환경·가짜 시계로 식 6·7, 예산, 이월(버림 없음), 소스별 몫, `t_i`의 정지 구간 빼기를 테스트. 리포트 그림 16처럼 소스 6개 추적 시뮬레이션으로 α = 0 / 0.5 / 1 비교
2. 단일 노드: General 소량을 uni-agent 러너로 실행, GeneralAgentLoop 결과와 보상·토큰 비교
3. 단일 노드: Code + General 혼합 소량, 소스별 채택·이월·예산 지표 확인
4. 다중 노드 본 실행

## 7. 결정할 것

1. **목표 비율의 기준**: `accepted` (리포트 그대로, 채택 그룹 수 기준 — 소스가 `1/r_i`로 재가중됨) / `generated` (명시한 도메인 비율 그대로 — 소스별 채택 그룹 가중치에 현재 `r̂_i`를 곱함, 추정 잡음이 조금 늘어남). 둘 다 설정으로 두고 기본값을 정해야 함
2. **`rollout.n`**: 둘 다 16 (리포트, General 롤아웃 비용 2배) / 소스별 n (그룹 크기 보정에 소스별 `n_ref` 필요)
3. **Code : General 비율**: 리포트의 코딩 68% : 도구 사용 12%를 둘만 남겨 정규화하면 약 85 : 15
