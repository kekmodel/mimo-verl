# 다음 작업 설계: Code + General 혼합 학습과 Sample Mixer

상태: 1단계(General 러너 이식)와 2단계(Sample Mixer) 구현 완료, CPU 검증만. pod 검증 남음. 기준 브랜치 `mimo-fixes`. 리포트 6.2 (multi-tenant rollout, heterogeneous harnesses), 6.3 (Sample Mixer).

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

### 구현 (1단계)

- `recipes/mixed/runner.py` `mixed_runner`: uni-agent에는 러너 하나. 인스턴스의 `dataset_type`으로 경로(`route_by_dataset_type`)를 고르고, 공통 `runner_kwargs` 위에 경로별 값을 덮어 MimoAgent 러너를 실행. 경로 이름은 `reward_info.source`. 매핑에 없는 `dataset_type`은 세션 실패
- `recipes/code/mimoagent_runner.py`에 선택 인자: `config_path`(하니스 혼합 대신 고정 설정, 상대 경로는 저장소 기준), `environment_hooks`, `reward_binarize_threshold`(`raw_reward`도 기록), `reward_timeout`, `source`
- `recipes/general/uni_runner.py` `GeneralHooks`: General 환경 등록, 필요 시 cc 도구 등록, `env_task_dir`를 `GA_TASK_ROOT` 기준으로, KUBECONFIG·DOCKER_REGISTRY·labels, MCP 도구 탐색 후 에이전트 도구 목록에 추가(캐시된 정의 갱신), 브리지를 sidecar로 복사. 모두 예산 게이트 설치 전
- `config/agent/general/s3k-uni.yaml`: `s3k.yaml` + 게이트웨이용 `model` 블록
- `recipes/mixed/config/mixed.yaml`: `train.yaml`을 상속. 러너·경로, General 예산 300초·벽시계 1200초(`trajectory_timeout_by_dataset`), 이진화 1.0, `gar.sources: [code]`
- `recipes/mixed/run_mixed.sh`: `CODE_TRAIN_DATA`, `GENERAL_TRAIN_DATA`, `GA_*`를 받아 Code 실행 스크립트를 혼합 설정으로 실행. 판정기 키는 `GA_JUDGE_KEY_FILE` 권장
- `train.yaml`의 `hydra.searchpath` 블록 제거 (실행 스크립트가 명령줄로 넘김; 주 설정에서만 허용되어 상속을 막았음)
- `algorithm.gar.sources`: 지정한 소스의 그룹만 채점 (`gar/groups_other_source`)
- 테스트: `tests/recipes/mixed/test_mixed_runner_on_cpu.py` (경로 선택·거부, 인자 덮기, General 훅으로 MCP 도구 등록·이진화·경로·쿠버네티스 설정, 설정 경로, 혼합 설정 합성과 시작 검사), GAR 소스 필터

Sample Mixer 전까지는 데이터로더가 두 parquet을 이어 붙여 뽑으므로, 배치의 소스 비율은 데이터 크기를 따른다.

참고:
- General 채점 시간 제한(`reward_timeout`)은 설정하지 않음. 원본 `REWARD_TIMEOUT` 기본값도 0(없음)
- `include_task_in_reward_info`는 경로 공통이라 GAR을 켜면 General 롤아웃도 과제 설명을 싣는다. GAR은 `sources`로 General을 건너뛰므로 저장 공간만 조금 더 씀
- `code` 경로의 하니스 선택은 그대로 `MIXED_HARNESS_SPEC`(Code 하니스 혼합). 경로별 하니스 목록은 아직 없음
- MCP 도구 등록은 실제 `cc-agent`(s3k-uni.yaml)로 테스트: 모델 요청의 도구 정의에 들어가고 이름으로 실행됨

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

### 구현 (2단계)

- `verl/trainer/ppo/v1/sample_mixer.py` `SampleMixer`: 소스별 몫(`apportion`, 기준 `accepted`/`generated`), 채택률 `r_i`와 활성 시간 `t_i`의 지수 이동 평균, 식 6 예산(`oversampling`, 이분 탐색), 식 7 가중치와 smooth weighted round-robin, 그룹 장부(진행 중/채택 대기), colocate 학습 정지 구간 기록과 활성 시간에서 빼기, 지표와 체크포인트 상태
- `MixerReplayBuffer` (`replay_buffer.py`): 소스별 몫이 다 차야 배치를 냄, 소스마다 오래된 것부터, 남는 것은 이월. DAPO·실패로 버려진 그룹은 거절로 기록. 배치를 기다리는 동안 모자란 소스에 `ceil(부족분 / r̂_i)`개까지 그 소스 프롬프트를 지정해서 보충 (다른 소스의 이월분은 거절이 아니라 보충을 일으키지 않으므로, 이게 없으면 영원히 기다릴 수 있음). 재시작 뒤 모르는 그룹은 TransferQueue의 프롬프트 `data_source`로 소스를 복원
- 트레이너: 소스별 데이터로더(`data_source`로 학습 세트를 나눔, 소스마다 시드), 프롬프트 하나마다 믹서가 소스 선택, 소스 지정 보충, colocate에서 `on_sample_end` 뒤 정지·`on_step_end` 뒤 재개 기록, 스텝마다 `mixer/*` 지표, `mixer.pt` 체크포인트
- 설정 `trainer.v1.sampler.mixer` (`mixed.yaml`에서 켜짐): 85 : 15, `accepted`, α 0.5. `p_mean` 1.0·`p_min` 0·`p_max` 4는 리포트에 값이 없어 우리가 정한 기본값
- 예산은 "모든 소스가 예산에 닿으면 무시"하는 부드러운 상한 (`mixer/over_budget_picks`로 셈). 트레이너는 소비되거나 거절된 그룹 하나마다 프롬프트 하나를 넣으므로 들어오는 양과 나가는 양이 같아 쌓이지 않음
- 테스트 `tests/trainer/ppo/v1/test_sample_mixer_on_cpu.py`: 식 6 평균·단조성, 몫(두 기준), 정지 시간 제외, 식 7 비율, 부족분 우선, 시작 배분 ∝ t·m, 거부, 추적 시뮬레이션(지속 시간 10배·채택률이 다른 두 소스, 동시 실행 한도: 매 배치 정확히 몫대로, 채택된 그룹은 하나도 안 버림, 이월이 쌓이지 않음), 실제 TransferQueue에서 몫·오래된 순·이월, 부족 소스 지정 보충(한 번만)과 거절 기록, 재시작 그룹 소스 복원. 트레이너 연결 테스트(`tests/recipes/mixed`)

지표: `mixer/<소스>/{quota, accept_rate, active_duration_s, budget, inflight, accepted_waiting, generated_share}`. `accepted` 기준에서 `generated_share`가 실제로 생성 프롬프트 기준 몇 %로 반영되는지 보여 준다 (`B_i / r_i` 정규화)

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

## 7. 결정 (2026-10-01)

1. **목표 비율의 기준: 기본 `accepted`** (리포트 그대로, 배치의 채택 그룹 수로 비율을 맞춤). `generated`(명시한 도메인 비율 그대로, 소스별 채택 그룹 가중치에 현재 `r̂_i`를 곱함)도 설정으로 둠. `accepted`에서는 소스가 `1/r_i`로 재가중되고 학습 중 비율이 흔들리므로, 소스별 `r_i`와 실효 비율(`B_i / r_i` 정규화)을 지표로 남김
2. **`rollout.n`: 둘 다 16** (리포트). General 롤아웃 비용 2배. 그룹 크기 보정은 지금처럼 `n_ref` 하나
3. **Code : General = 85 : 15** (리포트 코딩 68% : 도구 사용 12%를 정규화)
