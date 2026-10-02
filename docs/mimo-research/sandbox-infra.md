# 샌드박스 인프라: Kubernetes 없는 망분리 GPU 클러스터에서 돌리기

상태: Code용 Docker 샌드박스 백엔드 구현 완료(`recipes/sandbox/`). 가짜 Engine API를 상대로 한 단위 테스트와 로컬 Docker(29.x)를 상대로 한 통합 테스트로 검증했습니다. 대상 노드(Docker 20.10)에서는 아직 학습을 돌려 보지 않았습니다. General(메인 + MCP 사이드카)용 Docker 백엔드는 아직 없습니다.

## 1. 대상 환경 (실측)

| 항목 | 값 |
|---|---|
| 노드 | 멀티노드 GPU 서버와 공유 NAS. 별도 CPU 서버는 없음 |
| 실행 규칙 | 모든 작업을 Docker 컨테이너 안에서 실행. 인터넷 차단. pip·npm은 Nexus 경유 |
| 이미지 | 사내 GitLab 레지스트리(`kcr…/<user>/<project>:tag`)에 본인이 올린 것만 pull 가능 |
| Docker | Engine 20.10.17 (API 1.41), containerd 1.6.7, 커널 4.18 el8, nvidia runtime |
| 권한 플러그인 | HBM. `/var/run/docker.sock` 마운트는 허용되고, 그 밖의 바인드 마운트(예: `/usr/bin/docker`)는 거부됨 |
| 컨테이너 안에서 소켓 API 호출 | create(`NetworkMode=none`, `NanoCpus`, `Memory`, `PidsLimit`, `Labels`) → start → archive PUT → exec → DELETE 모두 통과 |
| user namespace | `unshare -Ur` 거부 → 컨테이너 안에서 Apptainer, bwrap, enroot 사용 불가 |

## 2. 업계는 어떻게 하나 (2026-10 조사)

| 곳 | 방식 | 규모 |
|---|---|---|
| Kimi K2 → K3 (AgentENV, 오픈소스) | Kubernetes에서 Firecracker microVM으로 이전. OverlayBD 지연 로딩, 스냅샷·포크 | 클러스터당 3만 |
| DeepSeek V4 (DSec) | 함수 풀, Docker(EROFS 지연 레이어), Firecracker, QEMU를 한 SDK로 묶음 | 수십만 |
| Qwen3-Coder / MegaFlow | ACK Kubernetes + Argo. SWE 이미지만 25 TB 이상, 레지스트리를 내부망에서 서빙 | 2만 |
| GLM-4.5 / 5 (slime) | 고동시성 Docker 런타임. 실패 원인을 기록하고 환경 붕괴 샘플은 제외 | 동시 1천 이상 |
| MiMo-V2 / V2.6 | Kubernetes pod, Ray 액터 풀(롤아웃마다 액터를 띄우면 GCS 파일 디스크립터가 고갈됨) | 1만 이상 |
| Meituan LongCat (DORA) | 남는 CPU 서버에 환경을 배치 | 서버 400대로 3.2만 |
| NVIDIA Nemotron | root가 없는 SLURM이라 Apptainer `.sif` + tmpfs 오버레이 | 512 |
| DeepSWE (rLLM) | 데몬 하나에 컨테이너 512개를 몰자 Docker 데몬이 다운 → Kubernetes | 512 |

공통 패턴:
1. 샌드박스는 별도 CPU 서버에 두고, 학습은 비동기로 돌린다.
2. 격리 방식은 클러스터 사정을 따른다.
3. 병목은 이미지다(지연 로딩, 노드 NVMe에 미리 받기, 가까운 레지스트리).
4. 시작 시간을 학습 경로에서 뺀다(웜 풀, 재사용, 스냅샷 포크).
5. 제어부에는 동시 요청 수를 제한한다.
6. 인프라 실패는 0점이 아니라 학습에서 제외한다. 우리는 이미 `exclude_invalid_rows`와 세션 단위 무효 처리로 하고 있다.

우리 환경에서 쓸 수 있는 것:

| 기술 | 가능 여부 |
|---|---|
| Docker 형제 컨테이너 (DooD, runc) | ✅ 검증됨 |
| Apptainer, enroot, bwrap | ❌ user namespace가 막혀 있음 |
| gVisor, Kata, Firecracker | ❌ 데몬 설정, KVM, 커널 5.10 이상이 필요(관리자 권한) |
| 지연 로딩, P2P 배포 | ❌ containerd 스냅샷터 전용이라 Docker 20.10에서는 전체 pull |

## 3. 과제 이미지 용량 (Docker Hub `xiaomimimo/mimo-v2.6-rl-oss` 실측)

| | 이미지 수 | 압축 상태 | 디스크에 풀었을 때 |
|---|---|---|---|
| Code (`format-code-task-*`) | 2,698 | 약 6.7 TB (중앙값 3.1 GB, 최대 13.7 GB) | 약 16–20 TB (압축률 2.4–3배, 샘플 2개) |
| General (`general-agent-env-*`) | 65 | 약 20 GB | 약 50 GB |

샘플 60개를 확인해 보니 이미지마다 레이어가 1장으로 합쳐져 있어 **레이어 공유가 없다**. 따라서 노드마다 전부 받아 두는 방식은 불가능하다. 일부 과제(500–800개, 1.5–2 TB)로 시작하고 학습 parquet도 그 이미지 목록으로 거른다. 노드에서는 미리 받거나(`pull_policy: never`), 처음 쓸 때 받는다(`missing`, 노드당 동시 pull 상한 적용).

## 4. 구현 (`recipes/sandbox/`)

- `docker_api.py`: 표준 라이브러리만 쓰는 Engine API 클라이언트(유닉스 소켓 또는 tcp). 비TTY attach 스트림 분리, pull 진행 줄의 `error` 감지, `config.json` 형식에서 `X-Registry-Auth` 생성.
- `docker_env.py` `DockerSandboxEnvironment`: mimoagent Kubernetes 백엔드의 계약을 그대로 따른다.
  - `timeout N /bin/bash -lc "cd <cwd> && cmd"`로 실행하고 종료 코드 124는 `pod_timeout`으로 본다.
  - 클라이언트 마감(`+5s`)을 넘기면 `client_timeout`, 샌드박스를 잃으면 `transport_error`(`raise_on_transport_error`이면 `TransportError`)를 낸다.
  - 출력은 마지막 50 MB만 남긴다.
  - `copy_to`/`copy_out`은 tar로 옮기며 항목을 `dest_path`에 놓는다. 부모 디렉터리는 데몬이 만든다.
  - 컨테이너 생성 규칙:
    - 이미지의 ENTRYPOINT를 `sh -c "sleep <max_lifetime>"`으로 교체한다. 수명이 끝나면 `AutoRemove`로 스스로 사라진다.
    - `NetworkMode=none`, `NanoCpus`/`Memory`/`CpuShares`/`MemoryReservation`/`PidsLimit`를 건다. 값은 하네스 yaml의 Kubernetes 이름(`cpu_limit` 등)을 그대로 받는다.
    - 라벨: `mimo.sandbox`, `mimo.exp`, `mimo.instance`, `mimo.owner`.
  - 노드당 동시 생성·pull 수를 `flock` 슬롯으로 제한한다(같은 노드의 러너 프로세스가 공유하고, 죽은 프로세스의 슬롯은 자동으로 풀린다).
  - 정리는 동기식이며 404가 날 때까지 확인한다.
- `reap.py`: 라벨 `mimo.sandbox=1`이 붙은 것만 지운다(공유 데몬이므로).
- 연결:
  - `SANDBOX=docker` → `+sandbox=docker`(`recipes/code/config/sandbox/docker.yaml`, 값은 `SANDBOX_*` 환경 변수).
  - 러너의 `environment_overrides`에서 값이 null이면 하네스 yaml의 키를 지운다(`node_selector`).
  - run_train.sh는 시작할 때 데몬에 닿는지 확인하고, 망분리용 `MIMOAGENT_RG_PATH`/`MIMOAGENT_CODE_MODE_HOST_PATH`를 Ray 워커로 넘긴다.
- 배치: 러너는 Ray 태스크(`dispatch_mode: ray_task`)라서 태스크가 도는 노드의 데몬에 샌드박스가 생긴다. 노드당 동시 롤아웃 수는 Ray CPU 수 ÷ `UNI_AGENT_RUNNER_TASK_NUM_CPUS`로 묶인다.

## 5. 남은 일

1. **대상 노드에서 확인**
   - 학습 이미지에 이 저장소를 넣고 `PREFLIGHT_ONLY=1 SANDBOX=docker`를 실행한다.
   - 이어서 과제 몇 개로 짧게 실제 실행해 본다.
   - 확인할 지표: `num_failed_sessions`, `transport_error` 비율, 샌드박스 시작 시간.
2. **이미지 이전 도구**
   - 공개 이미지를 kcr로 옮기는 스크립트(재개 가능, 완료 목록 기록).
   - parquet을 레지스트리에 있는 이미지로 거르는 도구.
   - 노드 사전 pull 도구.
3. **노드 디스크 캐시**: 디스크 상한 안에서 오래 안 쓴 과제 이미지부터 지운다. 대상은 우리 접두사가 붙고 사용 중이 아닌 것만이다.
4. **General Docker 사이드카**
   - 메인 컨테이너와 사이드카가 네트워크를 공유해야 한다(`NetworkMode: container:<main>`).
   - 공유 볼륨이 필요하다(이름 있는 볼륨).
   - 둘 다 HBM이 허용하는지 먼저 테스트해야 한다.
5. **(선택) 그룹 친화 배치**: 한 과제의 롤아웃 16개를 같은 노드로 보내서 이미지 pull을 과제당 한 번으로 줄인다.
