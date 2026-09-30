# MiMo-V2.6 RL 알고리즘 역산 정리

공개 대시보드 https://mimo.xiaomi.com/rl/ 의 메트릭 API(`api/runs`, `api/tags`, `api/series`, `api/live`)에서 데이터를 받아 역산한 결과입니다. 두 런 모두 30스텝에서 종료됐고(pro 127시간, flash 82시간), 본문 1~10장은 14~16스텝 시점의 분석, 11장은 30스텝 전체로 다시 검증한 결과입니다. 13장은 공개된 기술 리포트와의 대조이며, 1~12장과 충돌하면 13장이 우선합니다. 최종 갱신 2026-09-23.

각 항목에 **[확정]** 또는 **[추정]** 을 붙였습니다. 확정은 로그 수치끼리의 관계로 검증된 것, 추정은 태그 이름과 값의 패턴에서 유추한 것입니다.

---

## 0. 표기

| 기호 | 의미 |
|---|---|
| $\mathcal{D}$ | 데이터셋 25개 (code 11, visual 6, general 4, chat 3, cyber 1) |
| $q_D$ | 데이터셋 $D$의 스텝당 프롬프트 쿼터, $\sum_D q_D = 1568$ |
| $x$ | 프롬프트, 프롬프트당 롤아웃 수 $n = 16$ |
| $y_i$ | 롤아웃 $i$의 토큰열 (멀티턴 궤적 전체), $t$는 토큰 인덱스 |
| $\pi_{\text{roll}}$ | 롤아웃을 생성한 추론 엔진의 정책. 트레이너보다 $s$ 스텝 오래됨, $s \in [0, 1.9]$ |
| $\pi_\theta$ | 트레이너의 현재 정책 |

---

## 1. 동적 샘플링 (dynsam)

프롬프트 $x$의 통과율과 수락 조건:

$$
p(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\text{롤아웃 } i \text{ 성공}], \qquad
\text{accept}(x) \iff 0 < p(x) < 1
$$

measurable은 infra 오류로 실패한 시도를 제외하고 통과율을 잴 수 있는 프롬프트입니다 (`dynsam/avg@n_no_infra`, `infra_error/seq_rate` 0.3~0.9%).

**[확정]** pro 1스텝에서 measurable 프롬프트 4040개, passrate zero 14.6%, one 17.8%이고 $4040 \times (1 - 0.324) = 2731 \approx$ 데이터셋별 accepted 합 2736. 전부 성공하거나 전부 실패한 그룹은 샘플링 단계에서 버립니다 (DAPO식 dynamic sampling).

쿼터 채우기와 이월:

$$
\text{held}_D(k) = \text{new}_D(k) + \text{carry}_D(k-1), \qquad
\text{train}_D(k) = \min(q_D,\ \text{held}_D(k)), \qquad
\text{carry}_D(k) = \text{held}_D(k) - \text{train}_D(k)
$$

**[확정]** pro 4~10스텝에서 데이터셋별 step, carryover, held 값이 이 관계를 만족. 트레이너 재시작 시 carry가 소실됨 (pro 1~3, 11스텝).

| 카테고리 | 쿼터 | 비중 |
|---|---|---|
| code | 1061 | 68% |
| visual | 206 | 13% |
| general | 190 | 12% |
| cyber | 64 | 4% |
| chat | 47 | 3% |

---

## 2. 보상

$$
r_i = \text{score}_i \in [-0.8,\ 1]
$$

**[확정]** `critic/score`와 `critic/rewards`가 325개 비교점 전부에서 동일. reward에 KL 항이나 다른 shaping 항이 없습니다. `critic/returns`도 `critic/advantages`와 동일하여 value 모델과 할인이 없습니다.

**[정정]** 이전 판의 "길이 페널티 없음"은 틀렸습니다. 1M 한도 데이터셋에는 처음부터 soft overlong 감점이 있고(11.3), 전반부에 안 보였던 건 궤적이 램프 구간(약 50만 토큰 이상)에 거의 안 들어갔기 때문입니다. step 3의 "도달했는데 0점" 사례는 256k 한도인 chat 데이터셋이었습니다. advantage와 loss 층에는 길이 항이 없습니다. 잘린 시퀀스는 그룹에 남아 실패(0)로 학습됨: `train/verdicts/trained`가 전 스텝 정확히 25,088이고 `clip_ratio × 25088`이 정수(스텝당 pro 0~5개, flash 1~18개)로 떨어지므로 배치에서 제거되지 않음 **[확정]**. 256k 한도 데이터셋(chat 3개, code 4onq)은 잘려도 0점이고, 1M 한도 데이터셋은 램프 감점을 받아 최대 −0.8 (11.3). advantage를 0으로 두는 loss 마스킹 가능성은 집계로 배제 불가 **[추정]**.

**[확정]** chat과 general 일부는 $r \in \{0, 1\}$ 이진. code, cyber, visual 일부는 $-0.8$까지 내려가는 음수 감점이 존재. visual 두 데이터셋은 최댓값 0.995의 연속 점수.

---

## 3. 그룹 advantage

$$
A_i = r_i - \frac{1}{n}\sum_{j=1}^{n} r_j
$$

**[확정]** 이진 데이터셋에서 advantage 최댓값이 거의 매 스텝 $0.9375 = 1 - \tfrac{1}{16}$, 다음 값이 $0.875 = 1 - \tfrac{2}{16}$. 표준편차로 나누지 않습니다. 시퀀스 내 모든 토큰에 동일한 $A_i$가 붙습니다.

**[확정]** 로그된 `advantages/mean`은 응답 토큰 가중 평균이라 0이 아니며 (pro $-0.003 \to -0.008$), 음수는 실패 롤아웃이 성공 롤아웃보다 길다는 뜻입니다.

---

## 4. 그룹 내 credit assignment (stage_credit_group)

한 문장으로: **테스트를 통과한 롤아웃이 정말 풀어서 통과했는지 LLM judge로 판정하고, 그 결과로 그룹 안의 credit 배분을 바꾼다.** reward는 건드리지 않고 advantage만 바꿉니다. `stage`는 트레이너 파이프라인의 한 단계 이름이고, 현재 켜진 건 롤아웃 단위 factor뿐입니다 (턴/토큰 단위 재작성 `tq_adv_*`는 전부 0).

### 4.1 그룹 funnel (pro step 1 실측)

| 단계 | 그룹 수 | 설명 |
|---|---|---|
| accepted | 2736 | dynsam이 넘긴 전체. `groups_total` |
| routed off | 1333 | judge 대상이 아닌 데이터셋. 비코드 17개와 code의 m1dt, yfch |
| routed on | 1403 | `select_v4` 648 (gold 패치 있음) + `select_v4_nogold` 755 |
| uniform 생략 | 124 | judge **이전**에 제외. 판정할 게 없는 그룹 (`select_groups_skipped_uniform`) |
| pending | 56 | 스텝 마감까지 judge 차례가 안 옴 (`judge_pending`) |
| attempted | 1223 | = 1403 − 124 − 56 |
| pod 셋업 실패 | 16 | 샌드박스 오류 |
| pass1 실행 | 1207 | = 1223 − 16. 실패 0. **pass2는 시도 0회** (`judge_pass2_attempts` = 0) |
| select 실패 | 27 | judge 출력이 일관성 검사에 걸림 |
| judged | 1180 | = 1207 − 27. 이 그룹 전부가 factor를 받음 |

**[확정]** 위 산술이 step 1, 2에서 정확히 맞음 (step 2: 1103 − 85 − 142 = 876). step 13, 14는 pending을 빼지 않아야 맞는데, pending 집계 시점 차이로 보임.

**[추정]** "uniform"은 factor가 아니라 judge 전에 알 수 있는 성질입니다. 통과 롤아웃이 하나뿐이거나 통과들 사이에 순위를 매길 차이가 없는 그룹일 가능성이 높습니다. 어느 쪽이든 재중심화 뒤에 효과가 없을 그룹에 judge 비용을 쓰지 않는 장치입니다.

**[추정]** routed 그룹의 84%만 판정됩니다. pending과 pod 실패, select 실패로 판정을 못 받은 그룹은 factor 없이 ($f = 1$) 학습되는 것으로 보입니다. 별도 처리 태그가 없습니다.

**[확정]** judge 소요 시간: pod 셋업 14초, pass1 평균 520~640초, 그룹당 총 평균 655초, 최대 2시간. 스텝 시간(2~2.6시간) 안에 못 끝나는 그룹이 `groups_expired_unjudged`로 빠짐 (스텝당 0~1개).

**[확정]** pro step 14에서 pass1 성공률이 96%에서 77%로 떨어지고 select 실패가 194로 7배 튐. judge 쪽 장애가 있었던 스텝.

### 4.2 judge가 보는 것

**[추정, 근거 강함]** 롤아웃 단위 지표(`select_hack_attempt` 4563 / rate 0.3833, `select_r2_flagged` 379 / rate 0.03184)의 분모가 모두 11,903입니다. judged 그룹의 롤아웃 18,880개의 63%이고, 이 데이터셋들의 통과율(약 0.6)과 일치합니다. 즉 **hack 판정과 티어는 통과한 롤아웃에 대해서만 매깁니다.** 실패한 롤아웃은 별도로 r1(4.3)의 대상이 됩니다.

통과 롤아웃 하나마다 산출하는 것 (태그 이름 기준, 정의는 **[추정]**):

| 출력 | pro 값 | 의미 |
|---|---|---|
| 점수 5축 $(A, B, E, P, S)$ | 평균 4.1~4.6 | 5점 척도 루브릭 |
| 티어 $\{T_1, T_2, T_3, H\}$ | 65% / 33% / 1% / 1% | 통과의 품질 등급. $H$는 hack |
| hack 시도 | 통과의 38% | 테스트 수정, 특수 케이스 처리 등 |
| hack 시도가 최소 심각도 이상 | 4.5% | |
| 시도했지만 결과에 의존 안 함 | 스텝당 30~90건 | 감점 완화 대상 |
| 프로세스 심각 위반 | 스텝당 10~30건 | r1 마스킹 근거 |
| 회귀 유발 | 스텝당 10~70건 | 기존 테스트를 깨뜨림 |
| 새 테스트 통과 | 70% | judge가 만든 추가 테스트 |
| gold보다 우수 | 30% | gold 패치가 있는 그룹만 |
| 구현 크기 / gold | 평균 2~140 (heavy tail) | 과도한 구현 감지 |
| probe 불일치 | 57~66% | 보조 검사와 judge의 의견 차 |

judge 출력의 일관성 검사 (`rank_invalid`, `rank_score_conflict`, `tier_mismatch`)에 걸리면 그 그룹은 `groups_failed_select`로 버립니다.

### 4.3 factor와 세 가지 룰

$$
f_i \in [0, 1], \qquad \bar{f} \approx 0.83
$$

**[사실상 확정]** $f$는 티어별 상수입니다. 하니스 4개, select 변형 2개, 전체 집계, 두 런의 전 스텝을 합친 217개 관측점에서 `select_factor_mean`을 티어 비중으로 회귀(상수항 없음)한 결과:

| 티어 | 비중 | 계수 | 값 |
|---|---|---|---|
| $T_1$ | 64.8% | 0.976 | $f(T_1) \approx 1$ |
| $T_2$ | 33.3% | 0.587 | $f(T_2) \approx 0.6$ |
| $T_3$ | 0.8% | 0.24 | $f(T_3) \approx 0.25$, 비중이 작아 $\pm 0.15$ |
| $H$ | 1.1% | 0.011 | $f(H) \approx 0$, $\pm 0.1$ |

$R^2 = 0.953$, 잔차 표준편차 0.0026 (factor_mean의 표준편차 0.0119). 마스킹 비율이나 hack 심각도 비율을 회귀에 추가해도 계수가 변하지 않습니다. 30스텝 전체로 다시 돌리면 pro $(0.955, 0.634)$, flash $(0.968, 0.613)$, $R^2$ 0.98~0.99로 $f(T_1) \approx 0.96{\sim}0.98$, $f(T_2) \approx 0.6$이 유지됩니다. 실패 롤아웃은 티어가 없으므로 $f = 1$ (r1 마스킹 제외).

| 룰 | 동작 | 분모 | 상태 |
|---|---|---|---|
| r1 | 실패 롤아웃 중 일부의 advantage를 마스킹 | 실패 롤아웃 (비율 1.05~1.14로 가장 안정) | 발동, 실패의 약 6%, 스텝당 180~450 롤아웃 |
| r2 | 근거 약한 통과 $\Rightarrow$ 양수 advantage 상한 | 통과 롤아웃 | flag 100~380건, cap 발동 0 |
| r3 | gold 패치가 샌드박스에서 실패 $\Rightarrow$ 그룹 제외 | gold 있는 그룹 | 대상 10~23그룹, 실패 0 |

**[확정]** r2와 r3는 현재 학습에 영향이 없고 집계만 됩니다. 실제로 작동하는 건 factor와 r1뿐입니다.

**[추정]** r1의 분모를 `select_r1_masked / select_r1_rate`로 복원하면 실패 롤아웃 수의 1.05~1.14배로, 통과 기준(0.56~0.73)이나 그룹 기준(6.0~6.9)보다 훨씬 안정적입니다. 즉 r1은 **실패 롤아웃**을 대상으로 합니다. 실패를 마스킹하는 자연스러운 이유는 처벌이 아니라 면책입니다. 환경 결함이나 하니스 오류처럼 정책 탓이 아닌 실패의 음수 advantage를 지우는 것으로 읽힙니다. 10% 초과분의 정체는 알 수 없습니다.

### 4.4 advantage 재계산

$$
\tilde{A}_i = f_i \, A_i
$$

$$
\hat{A}_i = \tilde{A}_i - \frac{1}{n}\sum_{j} \tilde{A}_j
$$

$$
k = \min\!\left(K_{\text{cap}},\ \frac{\sum_i |A_i|}{\sum_i |\hat{A}_i|}\right), \qquad
A'_i = k \, \hat{A}_i
$$

- **[확정]** 재중심화: `select_adv_group_sum_abs_mean` $\approx 10^{-16}$, 즉 $\sum_i \hat{A}_i = 0$.
- **[확정]** $k$ 평균 1.19, 그룹의 3~5%가 cap에 걸림.
- **[추정]** $k$가 $|A|$ 질량을 보존하는 형태라는 것. 태그 명명(`mass`)과 5장에서 검증된 같은 형태의 공식에 근거. $K_{\text{cap}}$은 1.5 전후로 추정.
- **[확정]** 재작성된 행 수 `select_tq_adv_rows_rewritten` = 18,840 ≈ judged 그룹 × 16. factor는 롤아웃 단위이고 시퀀스의 모든 토큰에 같은 값이 곱해집니다.

### 4.5 예시: 왜 재중심화해도 페널티가 살아남는가

$n = 4$, 앞의 둘이 통과, 뒤의 둘이 실패. 원본 $A = [+1, +1, -1, -1]$.

| 상황 | $f$ | $f \odot A$ | 재중심화 후 | 결과 |
|---|---|---|---|---|
| 통과 둘 다 hack | $[0.5, 0.5, 1, 1]$ | $[0.5, 0.5, -1, -1]$ | $[0.75, 0.75, -0.75, -0.75]$ | 균일 factor는 소멸. 이래서 `skipped_uniform` |
| 하나만 hack | $[0.5, 1, 1, 1]$ | $[0.5, 1, -1, -1]$ | $[0.625, 1.125, -0.875, -0.875]$ | hack 통과 < 정직한 통과. 상대 순위 유지 |
| 실패 하나가 r1 면책 | $[1, 1, 1, 0]$ | $[1, 1, -1, 0]$ | $[0.75, 0.75, -1.25, -0.25]$ | 재중심화 전 $f=0$이면 면책된 실패도 약한 음수를 받음 |

마지막 행은 주의가 필요합니다. r1 마스킹이 "재중심화 전에 $f = 0$"인지 "재중심화 후 loss에서 제외"인지는 데이터로 구분되지 않습니다. 면책이 목적이라면 후자가 자연스럽고, 그 경우 마스킹된 롤아웃은 정확히 0을 받고 나머지 셋만 남습니다. **[추정]**

핵심은 이 단계가 **그룹 안의 상대 재순위**라는 점입니다. "이 프롬프트의 16개 시도 중 누가 credit을 받을 자격이 있나"를 다시 정하는 것이지, 그룹 전체를 벌주는 것이 아닙니다. 그룹 전체에 걸리는 절대적 압력은 5장의 signed 페널티가 담당합니다.

---

## 5. 토큰 단위 signed 페널티 (배치 전체)

지목 토큰 집합 $H_+$ (양수 advantage 시퀀스 안), $H_-$ (음수 advantage 시퀀스 안)에 대해:

$$
A''_{i,t} =
\begin{cases}
A'_i - \delta_{i,t}, & t \in H_+ \quad (\delta > 0,\ \text{양수 질량 } M_+ \text{ 제거}) \\
A'_i - \varepsilon_{i,t}, & t \in H_- \quad (\varepsilon > 0,\ \text{음수 질량 } M_- \text{ 추가}) \\
A'_i, & \text{otherwise}
\end{cases}
$$

**$\delta$, $\varepsilon$의 정의.** 토큰별 조정량이며 토큰 단위 규칙은 비공개입니다. 로그에 있는 건 합계와 개수뿐입니다.

$$
M_+ = \sum_{t \in H_+} \delta_{i,t}, \qquad
M_- = \sum_{t \in H_-} \varepsilon_{i,t}, \qquad
|H_+| = \texttt{pos\_hit\_tokens}, \quad |H_-| = \texttt{neg\_hit\_tokens}
$$

**[확정]** pro step 1 기준 토큰당 평균값과 비교 기준:

| 양 | 값 | 계산 |
|---|---|---|
| $\bar{\delta} = M_+ / \lvert H_+ \rvert$ | 0.149 | $1.229\times10^5 / 8.257\times10^5$ |
| $\bar{\varepsilon} = M_- / \lvert H_- \rvert$ | 0.320 | $3.267\times10^5 / 1.020\times10^6$ |
| 전체 토큰 평균 $\lvert A' \rvert$ | 약 0.12 | $(1.036 + 1.063)\times10^8 / 1.7\times10^9$ |
| 양수 토큰 평균 $A'$ (추정) | 약 0.11 | 통과 롤아웃이 토큰의 약 55%라고 가정 |
| 음수 토큰 평균 $\lvert A' \rvert$ (추정) | 약 0.14 | 실패 롤아웃이 더 길다는 3장 관측 반영 |

**[추정]** 후보 규칙과 데이터 정합성:

| 후보 | $\delta$ | $\varepsilon$ | 정합성 |
|---|---|---|---|
| 0으로 만들기 | $\delta = A'_i$ | $\varepsilon = \lvert A'_i \rvert$ (크기 2배) | 지목 토큰이 평균 $A'$인 위치라면 $\bar\delta$는 0.11, $\bar\varepsilon$은 0.14여야 함. 관측은 0.15, 0.32. 지목 토큰이 $\lvert A' \rvert$가 큰 롤아웃(통과가 적은 그룹)에 몰려 있으면 설명 가능 |
| 비율 감쇠 | $\delta = (1-\lambda) A'_i$ | $\varepsilon = \mu \lvert A'_i \rvert$ | 지목 토큰이 평균 위치라면 $\lambda \approx 0$, $\mu \approx 2.3$. 같은 가정에 의존 |
| 상수 | $\delta = c_+$ | $\varepsilon = c_-$ | $c_+ \approx 0.15$, $c_- \approx 0.32$. 가정 불필요 |

지목 토큰의 $\lvert A' \rvert$ 분포를 모르므로 세 후보 중 어느 것도 배제할 수 없습니다. 확정할 수 있는 건 토큰당 조정량이 배치 평균 $\lvert A' \rvert$보다 크다는 것(양수 1.3배, 음수 2.6배)뿐입니다. $\delta \le A'_i$ (양수를 음수로 넘기지 않음)인지도 알 수 없습니다.

부호별 총합을 보존하는 전역 스케일:

$$
s_+ = 1 + \frac{M_+}{\sum_{A' > 0,\ t \notin H} A'}, \qquad
s_- = 1 - \frac{M_-}{\left|\sum_{A' < 0,\ t \notin H} A'\right|}
$$

분모는 지목 토큰을 뺀 나머지의 합입니다. 지목 토큰이 전체 질량의 1% 미만이던 전반부에는 $\sum$ 전체로 계산해도 4자리까지 일치했지만, 후반부에 지목 토큰이 음수 질량의 3~10%를 차지하자 전체 합 공식은 최대 0.01 어긋나고, 지목 토큰을 뺀 분모로 두면 관측된 scale과 방향이 맞습니다 **[추정]**.

$$
s_\pm \leftarrow \operatorname{clip}(s_\pm,\ 1 - c,\ 1 + c), \qquad
A''_{i,t} \leftarrow s_\pm \, A''_{i,t} \quad (t \notin H)
$$

**[확정]** pro 1스텝: $M_- / |\sum A'_{<0}| = 3.267\times10^5 / 1.063\times10^8 = 0.00307$, $1 - 0.00307 = 0.9969$ = 로그된 `neg_scale`. 양수 쪽도 동일하게 일치. `train/adv_pos_sum_pre_penalty` = `post_penalty`, 음수도 동일.

**[확정]** 지목 토큰은 스텝당 $5\times10^5 \sim 1.7\times10^6$개로 학습 토큰의 0.1% 미만. 토큰당 이동 질량은 양수 0.15, 음수 0.32로 배치 평균 $\lvert A' \rvert \approx 0.12$의 1.3배, 2.6배.

**[확정]** clamp 미발동 (30스텝 전부). 관측 범위 $s_+ \le 1.003$, $s_- \ge 0.896$ (flash step 30)이므로 $c > 0.10$. 값은 비공개.

**[추정]** 무엇을 지목하는지는 비공개. 시퀀스 단위 판정과 별개로 특정 span(테스트 수정, 하드코딩, skip 마킹 등)을 노리는 것으로 보임.

이 단계는 그룹이 아니라 배치 전체에서 정규화하므로, 지목 토큰이 몰린 그룹은 순음수가 되어 절대적인 압력이 남습니다.

---

## 6. 손실

롤아웃 정책 대비 중요도 비율과 TIS(truncated importance sampling) 가중치:

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid y_{i,<t})}{\pi_{\text{roll}}(y_{i,t} \mid y_{i,<t})}, \qquad
w_{i,t} = \rho_{i,t} \cdot \mathbb{1}\!\left[\tfrac{1}{C} \le \rho_{i,t} \le C\right]
$$

$$
\mathcal{L}(\theta) = -\,\frac{1}{\sum_i |y_i|}\ \sum_i \sum_t\ w_{i,t}\ A''_{i,t}\ \log \pi_\theta(y_{i,t} \mid y_{i,<t})
$$

- $w$는 stop-gradient로 취급합니다 (표준 IS 구현).
- **[정정]** 이전 판에서 `pg_loss` $= -\text{mean}(A'')$ 등식이 성립한다고 썼으나, 이는 pro 13스텝 한 점의 우연이었습니다. 30스텝 전체에서 `pg_loss` / $(-$`advantages/mean`$)$ 비는 0.4~2.4 사이를 오가고 flash 초반에는 부호도 다릅니다. 두 값은 부호와 추세(후반부로 갈수록 양수 증가)는 같지만 크기가 일치하지 않으므로, **loss 집계 방식(토큰 평균인지, 마이크로배치 평균의 평균인지 등)은 확정하지 못했습니다.** 위 식의 $\frac{1}{\sum_i |y_i|}$ 정규화는 가정입니다.
- **[확정]** PPO ratio $\pi_\theta / \pi_{\text{old}} \equiv 1$. 배치당 optimizer step이 1회라 (`actor_optimizer_steps` = 1) old = current. `ppo_kl` = 0, `pg_clipfrac` = 0. clip 설정 (`clip_low` 0.2, `clip_high` 0.27)은 존재하지만 무력.
- **[확정, 전반부]** $C = 10$, 즉 $\rho \in [0.1, 10]$ 밖을 자름. 정상 상태(staleness 1.7~1.9)에서 `pg_tis_clipfrac`과 $F(\tau{=}10)$의 비가 0.85~1.1이고, $F(5)$와 $F(10)$ 사이 거듭제곱 꼬리로 보간한 $C$가 pro 9.1~10.9, flash 9.5~11.1. 재시작 직후 스텝은 $F(10)$이 너무 작아 추정이 불안정(약 5). 후반부에 signed 지목 토큰이 급증한 스텝(11.2)에서는 같은 보간이 6.5~8을 주는데, 이는 $C$가 바뀐 게 아니라 clipfrac에 다른 항이 섞인 것으로 보임 **[추정]**. 잘리는 방향은 low 쪽 (트레이너가 샘플러보다 낮은 확률을 매김)이 high 쪽의 약 7배.
- **[추정]** 잘린 토큰이 마스킹인지 truncation인지는 구분 불가.
- **[확정]** KL 정규화 항 없음. 엔트로피는 로그만 있고 (`entropy_loss` 0.38~0.45) 보너스 계수는 알 수 없음.

---

## 7. 업데이트

$$
\theta \leftarrow \theta - \eta \cdot \text{Adam}(\nabla_\theta \mathcal{L}), \qquad \eta = 3\times10^{-6} \ (\text{고정})
$$

- **[확정]** 배치당 1스텝. grad norm 로깅 후 클리핑, 이상 시 스킵 가능 (`update_skipped`, 아직 0회). grad norm 0.005~0.009.
- **[확정]** pro 모델: 비전 인코더 약 295M 파라미터가 frozen (`num_zeros_in_grad_encoders` 상수), MTP 헤드 학습, MoE 죽은 expert 거의 없음.

---

## 8. 비동기 루프

```
스텝 k:
  샘플러:  π_roll = π_{θ_{k−s}} 로 롤아웃 생성. 진행 중인 궤적은 가중치 갱신을 넘어 계속 (partial rollouts)
  dynsam:  accept 된 프롬프트를 데이터셋 쿼터대로 채워 1568 × 16 배치 구성 (~2B 토큰)
  judge:   code 8개 데이터셋의 그룹을 judge 풀이 비동기로 판정
  트레이너: 4 → 5 → 6 → 7 을 한 번 수행
```

**[확정]** `partial/avg_staleness`가 0에서 1.9 근처로 올라 평형, 재시작 시 0으로 리셋. train-infer KL (0.002 → 0.0098)과 TIS clipfrac이 staleness와 같은 궤적을 그리므로, 이 불일치는 커널 오차가 아니라 정책 지연이 지배. staleness 0일 때의 바닥값 0.002~0.003이 순수 엔진 불일치.

**[확정]** 동시 활성 샌드박스 pro 약 2.3만, flash 약 3.8만. infra 오류율 0.3~0.9%.

---

## 9. 한 줄 요약

$$
A'' \;=\; \underbrace{s_\pm \circ \text{SignedHit}}_{\text{5. 토큰, 배치 정규화}}
\Big(\ \underbrace{k \cdot \text{Recenter}\big(f \odot (r - \bar{r})\big)}_{\text{4. 롤아웃, 그룹 정규화}}\ \Big)
$$

$$
\theta \;\leftarrow\; \theta - \eta\, \nabla_\theta
\Big[\, -\operatorname{tokenmean}\big(\, \text{TIS}(\rho) \cdot A'' \cdot \log \pi_\theta \,\big) \Big]
$$

reward는 검증기 출력 그대로 두고, advantage 위에 네 층을 얹습니다.

| 층 | 단위 | 정규화 범위 | 역할 |
|---|---|---|---|
| judge factor $f$ + 재중심화 + $k$ | 롤아웃 | 그룹 | hack 통과와 정직한 통과의 상대 순위 |
| signed hit + $s_\pm$ | 토큰 | 배치 | 특정 span에 절대적 압력, 부호별 총량 보존 |
| TIS 마스킹 | 토큰 | 없음 | 비동기 지연으로 인한 off-policy 보정 |
| 단일 스텝 + 작은 $\eta$ | 배치 | 없음 | KL 없이 안정성 확보 |

각 층은 "총량은 지키고 분배만 바꾼다"는 같은 원칙으로 설계되어 있어, 규칙 변경이나 재시작에도 유효 학습률이 흔들리지 않습니다.

---

## 10. 비공개로 남은 것

1. ~~다섯 축 점수와 티어의 정의. 티어별 $f$ 값~~ → GAGAR 논문으로 공개 (14.1). r1 분모의 10% 초과분은 여전히 미상
2. signed 페널티의 토큰 지목 기준 ($H_\pm$)과 토큰별 조정량 규칙 ($\delta$, $\varepsilon$)
3. ~~$K_{\text{cap}}$~~ → 1.5로 논문 확인 (14.1). $c$의 정확한 값 ($> 0.10$으로 추정). $C$는 전반부 기준 10으로 사실상 확정
8. loss 집계 방식 (토큰 평균 가정이 검증되지 않음, 6장 정정 참조)
9. 후반부 signed 지목 토큰 급증의 규칙 (11.2)
4. 엔트로피 보너스 계수 유무
5. ~~일부 환경의 $-0.8$ 음수 reward의 출처~~ → soft overlong 램프로 해소 (11.3). 남은 건 $L_{\text{soft}}$의 정확한 값과 데이터셋별 차이 여부
6. `skipped_uniform`의 정확한 기준, r1 마스킹이 재중심화 전인지 후인지, pass2의 용도
7. judge를 못 받은 routed 그룹(약 16%)의 처리 방식

---

## 11. 30스텝 종료 후 검증

두 런 모두 정확히 30스텝에서 `mode: ended`. pro 127시간, 재시작 14회, 누적 $2.62M, 학습 토큰 75B. flash 82시간, 재시작 5회, $854k, 81B.

### 11.1 왜 30스텝에서 끝났나

**[확정]** 계획된 예산입니다. 두 런이 서로 다른 날짜에, 마지막 스텝을 정상 완료한 직후(pro는 15분 뒤) 같은 스텝 수에서 끝났고, `trained_cum` = 30 × 25,088로 정확히 맞습니다. 장애 종료라면 스텝 수가 어긋나거나 마지막 스텝이 `redo`로 남았을 것입니다.

**[추정]** 30이라는 예산과 별개로, 계속 돌릴 유인이 줄어든 신호가 세 가지 겹칩니다.

| 신호 | pro | flash |
|---|---|---|
| headline `avg@n` | step 16부터 0.62~0.64 정체 | 0.51 → 0.64, 후반 완만 |
| in-house coding bench 기울기 (step ≥ 20) | +0.08/step | +0.18/step |
| 스텝당 wall-clock | 2.2h → 6.4h (마지막 5스텝 평균 4.8h) | 1.9h → 3.5h |
| 스텝당 비용 (과금 속도 × 시간) | 약 $48k → 약 $110k | 약 $20k → 약 $36k |

pro의 스텝 시간 증가는 두 요인입니다. 응답 길이가 1.8배(71k → 123k 토큰)로 늘어 토큰이 1.7배가 됐고, step 17의 GPU OOM(expert load imbalance) 뒤 병렬화 전략을 바꾸면서 trainer 시간이 3.8k초에서 6.4k초로 한 번에 65% 뛰었습니다. 이후 9번의 재시작이 마지막 16스텝에 몰려 있어 안정성도 한계였습니다.

### 11.2 가설 점검표

| 가설 (1~10장) | 30스텝 결과 | 판정 |
|---|---|---|
| 그룹 16, 평균만 빼는 advantage | 최댓값 0.9375 패턴 유지 | 유지 |
| reward에 KL 없음, value 모델 없음 | score ≡ rewards, returns ≡ advantages 유지 | 유지 |
| 배치당 1 optimizer step, PPO clip 무력 | `ppo_kl` = 0, `pg_clipfrac` = 0, 30스텝 전부 | 유지 |
| 티어별 $f$ = 1 / 0.6 / 0.25 / 0 | 30스텝 회귀 $(0.96, 0.62)$, $R^2$ 0.98 | 유지 |
| r2, r3, action 페널티, pass2, tq 재작성 전부 비활성 | 30스텝 전부 0 | 유지 |
| signed clamp 미발동 | 미발동, $s_-$ 최소 0.896 | 유지, $c > 0.10$ |
| signed 페널티는 극소수 토큰(0.1% 미만) | **전반부만.** pro step 20, flash step 18부터 음수 지목 토큰이 10~30배 급증 (최대 1.7e7, 2.6e7 = 토큰의 0.5~0.7%), 토큰당 추가 질량도 0.2~0.4에서 1.1~1.4로 | **수정** |
| `pg_loss` = −token-mean(adv) | 한 점의 우연. 비율 0.4~2.4 | **철회** |
| TIS $C$ = 10 | 전반부 유지. 후반 지목 급증 스텝에서 추정치 6.5~8 | 조건부 유지 |
| 길이 페널티 없음 | **틀림.** 1M 한도 데이터셋에 처음부터 soft overlong 램프가 있었고, 길이가 램프에 들어간 후반부에 드러남 (11.3) | **철회** |
| 엔트로피 보너스 판별 불가 | 엔트로피가 30스텝 내내 단조 상승 (pro 0.395 → 0.462, flash 0.413 → 0.471). 데이터셋별로는 code가 +0.14~0.16, chat은 하락. 균일한 보너스보다 과제 특성에 따른 상승 | 보너스 없음 쪽으로 기움 |
| 잘린 시퀀스는 그룹에 남아 학습 | `trained` = 25,088 고정 유지 | 유지 |
| dynsam 쿼터 고정 | pro step 15에 cyber 제거(공지대로, 이후 태그 NaN), step 24에 쉬운 과제 필터링(공지대로, `passrate/one` 0.28 → 0.20 급락) | 쿼터는 고정이나 풀은 중간에 편집됨 |

### 11.3 새로 드러난 것

**길이 페널티는 처음부터 있었습니다: soft overlong 램프.** 비채팅 데이터셋의 (데이터셋, 스텝) 셀마다 그 스텝의 최대 총 길이와 reward 최솟값을 맞춰보면 단조 관계가 나옵니다.

| 최대 총 길이 구간 | 셀 수 (pro / flash) | 최솟값이 음수인 비율 | 최솟값 평균 |
|---|---|---|---|
| < 400k | 397 / 325 | 0.00 / 0.00 | +0.00 |
| 400k~600k | 133 / 162 | 0.19 / 0.15 | −0.01 |
| 600k~700k | 16 / 46 | 0.62 / 0.72 | −0.12 / −0.15 |
| 700k~800k | 8 / 23 | 0.88 / 0.78 | −0.26 / −0.25 |
| 800k~900k | 10 / 19 | 0.90 / 0.89 | −0.35 / −0.38 |
| 한도 도달 (1,048,576) | 75 / 75 | 0.91 / 0.92 | −0.60 / −0.66, 정확히 −0.8인 비율 0.47 / 0.59 |

형태는 DAPO의 soft overlong penalty와 같습니다.

$$
r_i = \text{score}_i - 0.8 \cdot \operatorname{clip}\!\left(\frac{L_i - L_{\text{soft}}}{L_{\text{hard}} - L_{\text{soft}}},\ 0,\ 1\right), \qquad L_{\text{hard}} = 2^{20}
$$

- **[확정]** 최대 감점 0.8, 하드 한도 $2^{20}$. 셀의 최솟값은 "가장 긴 시퀀스가 실패했을 때"만 감점을 그대로 보여주므로 관측값은 실제 감점보다 덜 음수 쪽으로 치우칩니다.
- **[추정]** $L_{\text{soft}}$는 45만~52만. 제약 회귀(최댓값 0.8 고정)의 최적이 525k이고 $2^{19} = 524{,}288$이 자연스러운 후보입니다. 다만 x7wh는 437k~450k에서 이미 음수가 나와 데이터셋별 값이 다르거나 시작점이 조금 낮을 수 있습니다.
- **[확정]** 256k 한도 데이터셋(chat 3개, code 4onq)은 한도에 닿아도 0점입니다. 감점은 1M 한도 데이터셋에만 있습니다.
- **[확정]** 언제부터 물렸는지는 "최솟값이 음수인 1M 데이터셋 비율"과 "최대 길이가 525k를 넘는 1M 데이터셋 비율"이 스텝별로 거의 같이 움직이는 것으로 확인됩니다. pro는 step 1~19에 0.05~0.29, step 20~26에 0.10~0.35, step 27~30에 0.50~0.75입니다. 램프가 생긴 게 아니라 길이 분포의 꼬리가 램프로 들어간 것입니다.
- **[확정]** 그래도 평균 길이는 1.6~2.1배 늘었습니다. 램프가 50만 토큰 위에서만 작동하는데 평균은 7만~13만이라 꼬리만 누르고 본체는 못 누릅니다. 마지막 5스텝에서야 1M 데이터셋의 절반 이상이 매 스텝 램프에 걸렸습니다.

**hack 시도율은 줄지 않았습니다.** judge가 통과 롤아웃 중 hack 시도로 분류한 비율이 pro 0.37 → 0.39(최대 0.44), flash 0.40 → 0.50이고, 심각도 기준 이상은 flash에서 0.045 → 0.09로 두 배가 됐습니다. 4장의 credit 재배분이 그룹 내 상대 순위만 바꾸고 절대 압력은 주지 않는다는 분석과 일치하는 결과입니다. 같은 기간 $T_1$ 비중이 0.65 → 0.58로 내려가고 $f$ 평균이 0.83 → 0.80, renorm cap 발동률이 3.5% → 7~9%로 올랐습니다. judge가 깎는 양은 늘었지만 시도 자체는 억제되지 않았습니다.

**signed 페널티가 후반부에 절대 압력 쪽으로 강화됐습니다.** 음수 지목 토큰이 pro step 20, flash step 18부터 급증했고, 급증한 스텝은 staleness가 높은 스텝과 겹칩니다(flash 후반 상관 0.66). 토큰당 추가 질량 1.1~1.4는 배치 평균 $|A'|$ 0.14의 약 10배입니다. 규칙이 바뀐 것인지, 지목 기준이 stale 토큰과 관련된 것인지는 구분되지 않습니다. hack 시도율 상승에 대한 대응이었을 가능성이 있지만 공지에는 언급이 없습니다.

**길이는 모든 카테고리에서 1.6~2.1배 늘었습니다.** code 78k → 147k, general 53k → 90k, visual 69k → 123k, chat 3.4k → 7.0k(pro). advantage의 토큰 가중 평균은 −0.002에서 −0.018(pro), −0.025(flash)로 내려가 실패 궤적의 길이 우위가 커졌습니다.

**벤치마크는 셋 다 올랐고 후반에 기울기가 줄었습니다.**

| 벤치마크 | pro | flash | 후반 기울기 (step ≥ 20, pro / flash) |
|---|---|---|---|
| DeepSWE v1.1 (mini-swe-agent, avg@3) | 58.4 → 72.6 | 48.7 → 65.7 | +0.25 / +0.51 |
| In-house Coding (avg@3) | 57.5 → 65.4 | 53.8 → 62.9 | +0.08 / +0.18 |
| AutomationBench v1.0.6 (avg@3) | 45.2 → 53.1 | 44.8 → 52.7 | +0.23 / +0.28 |

flash가 pro보다 상승폭이 크고 후반에도 덜 꺾였습니다. headline `avg@n`은 flash가 0.644로 pro 0.633을 넘었지만, pro는 step 24 이후 쉬운 과제를 뺀 풀이라 직접 비교는 안 됩니다.

**공지로 확인된 운영 이력.** 09-16 pro VRAM 재시작. 09-17 flash step 15 되감기(infra 오류 미검출), pro 그레이더 네트워크 장애 재시작과 cyber 제거("롤아웃 로그에서 나쁜 패턴"). 09-18 pro step 17 OOM(expert load imbalance)과 병렬화 변경. 09-19 pro 쉬운 과제 필터링.

---

## 12. 턴 단위 credit 예상 (전부 [추정])

이번 런에서 꺼져 있던 경로의 태그 이름과 값 0으로만 추론한 것입니다. 실제 설계는 공개 자료로 확인해야 합니다.

### 12.1 태그가 말해주는 것

| 태그 | 값 | 읽히는 것 |
|---|---|---|
| `tq_adv_set_tokens`, `tq_adv_mul_tokens` | 0 | advantage를 토큰 span 단위로 **덮어쓰기(set)** 또는 **곱하기(mul)** 하는 두 연산. 단위가 토큰이므로 턴 span 대상 |
| `tq_adv_pos_mass`, `tq_adv_neg_mass` | 0 | 부호별 질량 장부. signed 페널티와 같은 "총량 보존" 설계 |
| `tq_adv_rows_rewritten` | 0 | 재작성된 롤아웃 수. 같은 접두사의 `select_tq_adv_rows_rewritten`은 롤아웃 단위 factor 경로에서 이미 18,840으로 찍힘. 즉 `tq_adv`는 advantage 재작성 모듈의 이름이고, 롤아웃 단위 모드만 켜져 있었음 |
| `keep_mass_capped` | 0 | 재작성 뒤 롤아웃이 **유지하는** 질량에 상한. 한 롤아웃이 credit을 통째로 잃거나 독식하지 않게 하는 장치 |
| `dev_neg_turns` | 0 | 음수로 판정된 턴 수. 턴 단위 판정 결과를 세는 자리 |
| `select_hack_attempt_turns_per_pass` | 0.65~0.87 | judge가 이미 hack 시도를 **턴 위치**로 찍고 있음. 통과 롤아웃당 약 0.8턴 |
| `select_pass_turns_mean` | 45~60 | 턴 수 집계. 턴 경계를 파이프라인이 알고 있음 |
| `judge_pass2_attempts` | 0 | pass1은 롤아웃 판정. pass2는 한 번도 안 돌았고 `tq` 경로와 함께 꺼져 있음. **pass2 = 턴 단위 주석 패스**일 가능성 |
| `penalty/action/adv_mul_min`, `adv_mul_tokens`, `adv_reduction_*` | 1, 0, 0 | judge 없이 규칙으로 특정 **action**(도구 호출)의 advantage를 하한 `adv_mul_min`까지 곱셈 감쇠하는 경로 |
| `stage_credit_group` | 이름 | "stage" = 궤적의 단계. 롤아웃 단위 factor는 이 이름의 축소판 |

### 12.2 예상 메커니즘

롤아웃 $i$의 턴 $u$가 차지하는 토큰 span을 $S_{i,u}$, judge pass2가 턴마다 매기는 라벨을 $\ell_{i,u} \in \{\text{good}, \text{neutral}, \text{bad}\}$라 하면:

$$
A''_{i,t} =
\begin{cases}
c_{\text{set}}, & t \in S_{i,u},\ \ell_{i,u} = \text{bad},\ A'_i > 0 \quad (\text{통과 롤아웃의 hack 턴: } c_{\text{set}} \le 0) \\
m_{\text{mul}} \cdot A'_i, & t \in S_{i,u},\ \ell_{i,u} = \text{neutral} \quad (\text{낭비 턴: } 0 < m_{\text{mul}} < 1) \\
A'_i, & \text{otherwise}
\end{cases}
$$

이어서 롤아웃 단위로 질량을 정리합니다.

$$
\text{keep}_i = \frac{\sum_t |A''_{i,t}|}{\sum_t |A'_{i,t}|}, \qquad \text{keep}_i \leftarrow \min(\text{keep}_i,\ \kappa)
$$

그리고 4.4와 같은 그룹 재중심화와 5장과 같은 부호별 전역 스케일이 뒤따를 것입니다. `tq_adv_pos_mass`, `tq_adv_neg_mass`가 그 장부입니다.

실패 롤아웃 쪽은 r1의 턴 버전입니다. `dev_neg_turns`로 세는 "일탈 턴"에만 음수를 남기고, 실패 궤적 안의 정상 턴은 `mul`로 음수를 덜어내는 방식이 자연스럽습니다. 지금 r1이 실패 롤아웃 6%를 통째로 면책하는데, 턴 단위면 "이 턴까지는 잘했고 여기서 망쳤다"로 나눌 수 있습니다.

`penalty/action`은 judge와 별개의 규칙 경로로 보입니다. 같은 명령 반복, 파괴적 명령, 테스트 파일 수정 같은 action 유형에 곱셈 계수를 걸고 `adv_mul_min`(예: 0.1~0.3)을 하한으로 두는 형태입니다. 이번 런은 하한이 1이라 아무것도 안 깎았습니다.

### 12.3 이게 왜 필요한가: 롤아웃 단위 factor의 구조적 한계

4.5에서 본 대로 롤아웃 단위 factor는 그룹 재중심화를 통과하면 **그룹 안의 상대 순위**만 남습니다. 통과 롤아웃이 전부 hack이면 factor가 균일해져 소멸하고, 실제로 그런 그룹을 `skipped_uniform`으로 매 스텝 60~120개씩 건너뛰었습니다. 11.3의 결과가 그 귀결입니다. hack 시도율이 30스텝 동안 줄지 않았고 flash에서는 0.40 → 0.50으로 올랐습니다.

턴 단위 재작성은 **롤아웃 안에서** credit을 옮깁니다. 같은 롤아웃의 hack 턴에서 credit을 빼서 정직한 턴에 남기므로, 그룹의 모든 통과가 hack이어도 "hack 턴은 강화되지 않고 그 앞뒤의 정상 작업은 강화된다"는 신호가 살아남습니다. 재중심화가 지울 수 없는 종류의 신호입니다. 이것이 hack 시도율을 실제로 내릴 수 있는 유일한 경로이고, 후반부에 signed 페널티를 10배로 키운 것은 이 경로 없이 절대 압력을 만들려던 임시 대응으로 읽힙니다.

부수 효과도 예상됩니다. 낭비 턴에 `mul` < 1을 걸면 길이가 눌리고(11.3의 램프가 못 한 일), 10만 토큰 궤적에 스칼라 하나를 붙이던 분산이 줄어 스텝당 롤아웃을 줄일 여지가 생깁니다.

### 12.4 왜 이번엔 안 켰을까

- **judge 비용.** pass1이 그룹당 평균 655초, 최대 2시간이고 그것도 routed 그룹의 84%만 소화했습니다. 턴 단위 pass2는 롤아웃 16개 × 50턴을 읽어야 하므로 지금 judge 풀로는 스텝 안에 끝나지 않습니다.
- **judge 노이즈.** 롤아웃 단위에서도 probe 불일치가 57~66%, 티어 불일치가 스텝당 200~400건입니다. 턴 단위 라벨은 더 시끄럽고, 시끄러운 라벨로 `set`을 하면 정직한 턴의 credit을 지웁니다. `keep_mass_capped`는 그 피해를 제한하는 안전장치로 보입니다.
- **검증 순서.** 이번 런은 롤아웃 단위 factor(v4)의 검증 런이었고, 그 위에 턴 단위를 얹는 게 다음 단계라는 순서가 자연스럽습니다.

### 12.5 켜졌을 때 확인할 지표

| 예상 | 지표 |
|---|---|
| `judge_pass2_attempts` > 0, `tq_adv_set_tokens` > 0 | 경로 활성화 |
| `select_hack_attempt_rate` 하락 | 절대 압력이 생겼다는 증거. 이번 런에서는 안 내려감 |
| `select_groups_skipped_uniform` 감소 | 균일 그룹도 턴 단위로는 신호가 있으므로 건너뛸 이유가 줄어듦 |
| `ctx_response_length/mean` 상승 둔화 | 낭비 턴 감쇠 효과 |
| `keep_mass_capped` > 0 | 재작성이 롤아웃 질량을 크게 깎는 경우가 실제로 발생 |
| `penalty/action/adv_mul_min` < 1 | 규칙 기반 action 페널티 활성화 |

---

## 13. 기술 리포트 대조 (2026-09-23)

`MiMo_V2_6_technical_report.pdf`(44쪽)의 4.1, 4.3, 5.1, 5.5, 6.1~6.4절과 1~12장을 대조했습니다. 리포트에는 하이퍼파라미터 값(A, B, X, κ, α_max, β_min, K)이 거의 없어서, 값은 대시보드 데이터로 다시 맞췄습니다(13.3).

### 13.1 채점표

| # | 우리 주장 (장) | 리포트 | 판정 |
|---|---|---|---|
| 1 | 1568 × 16, 도메인 비중 68 / 13 / 12 / 4 / 3% (1) | 똑같음 (coding 68, aesthetic design 13, tool use 12, cyber 4, context following 3) | 적중 |
| 2 | 평균만 빼는 advantage, std 나눗셈 없음 (3) | Eq.3 $A_i = R_i - \bar R$ | 적중 |
| 3 | 전부 통과, 전부 실패 그룹 제외 (1) | DAPO dynamic sampler | 적중 |
| 4 | 데이터셋 쿼터와 이월, 재시작 시 이월분 소실 (1) | Sample Mixer: 목표 분포, "surplus carried forward", 재시작 첫 스텝은 Sample Replay로 보충 | 적중 |
| 5 | 비동기 partial rollout, staleness 평형 1.9 (8) | partial rollout, staleness 상한 4 | 적중 |
| 6 | reward에 KL 없음, value 모델 없음 (2) | 목적식에 KL 항 없음, GRPO | 적중 |
| 7 | 배치당 1 step, PPO ratio 무력 (6) | ratio는 롤아웃 정책 $\mu$ 대비 $r = \text{sg}[\pi_\theta / \mu]$ 하나뿐 | 적중 |
| 8 | lr $3\times10^{-6}$ 고정 (7) | lr $3\times10^{-6}$, warmup 없음 | 적중 |
| 9 | judge: 코드 과제, 그룹 단위, 통과 롤아웃만 등급, 5축 (4.2) | SFT 에이전트 grader가 그룹 전체를 보고 통과 패치를 5개 차원으로 순위. 동률 허용 | 적중. 5축은 approach / precision / minimality / unintended effects / craftsmanship |
| 10 | judge를 못 받은 그룹은 $f=1$ (4.1) | "unusable grader outputs falling back to the original advantages" | 적중 |
| 11 | 질량 보존 rescale, cap 약 1.5 (4.4) | $\lambda = \sum_P A / \sum_P fA$, cap 있음, 값 비공개 | 적중. 13.3에서 $K = 1.5$ 확정 |
| 12 | 재중심화 $\sum \hat A = 0$ (4.4) | cap 뒤 그룹 평균을 뺌 | 적중 |
| 13 | signed 스케일, 지목 토큰을 뺀 분모 (5) | Eq.5: $\alpha = 1 + \sum_{H_+}A / \sum_{C_+}A$, $C$ = 지목 안 된 토큰 | 적중 |
| 14 | 부호별 총량 보존, 음수 압력 과다가 엔트로피를 올림 (5, 9) | 문장 그대로 있음 | 적중 |
| 15 | 엔트로피 보너스 없음 쪽 (11.2) | 언급 없음. 엔트로피는 IS 마스크 경계 조정으로 제어 | 적중 |
| 16 | pro step 14 judge 장애, step 17 OOM, flash step 15 (11.3) | grader 네트워크 단절, EP rank에 30배 쏠린 MoE OOM, cyber 클러스터 k8s 장애 | 적중 |
| 17 | r1은 처벌이 아니라 면책 (4.3) | Penalty Module이 "모델 탓이 아닌 infra 실패"를 mask. r1이라는 이름은 없음 | 방향 적중 |
| 18 | 티어 $f$ = 1 / 0.6 / 0.25 / 0 (4.3) | $f \in (0, 1]$, 값 비공개. hack은 $f$가 아니라 reward를 0으로 리셋 | 부분. 13.3 재적합으로 1 / **0.625** / **0.2**, hack은 실패 처리. 논문(14장)으로는 $T_1$ 1·0.9, $T_2$ 0.85→0.4 선형(평균 0.625), $T_3$ 0.2 |
| 19 | 순서: $f \cdot A$ → 재중심화 → $k$ (4.4) | $f \cdot A$ → 통과끼리 $\lambda$로 양수 질량 복원 → cap → 재중심화 | 부분. cap이 안 걸리면 재중심화는 아무 일도 안 함. "균일 factor 소멸" 결론은 그대로 |
| 20 | reward는 안 건드리고 advantage만 (4) | hack은 effective reward = 0 후 그룹 통계 재계산 | 부분. 대시보드 `rewards`는 리셋 전 값 |
| 21 | 길이 페널티: 1M 데이터셋의 절대 길이 soft overlong 램프, 최대 −0.8 (11.3) | **group-relative** 길이 페널티 Eq.4. 통과율 > A인 그룹의 **성공** 롤아웃만, 성공 길이의 B 분위수 기준 | 부분. 존재는 맞음, 형태는 다름 (13.4) |
| 22 | loss 집계 미확정, 토큰 평균 가정 (6) | **prompt-mean**: 그룹 토큰 수로 나눈 뒤 프롬프트 평균. 길이 증가 억제가 목적 | 미확정으로 둔 게 맞았음. `pg_loss` ≠ −token-mean(adv)의 이유 |
| 23 | TIS $C = 10$, $[0.1, 10]$ 밖을 자름 (6) | 양수와 음수 advantage에 각각 $[\epsilon^l, \epsilon^h]$ 마스크, 초기값 **$[0.2, 5.0]$**, 엔트로피 따라 런타임 조정 | **오답.** 2배 틀림. 후반 추정 6.5~8을 "다른 항이 섞임"으로 본 것도 경계 조정이었을 수 있음 |
| 24 | signed가 노리는 것: 테스트 수정, 하드코딩 등 (5) | **포맷 위반과 tool call 오류** (깨진 markup, 없는 도구 이름, 잘못된 인자) | **오답** |
| 25 | 옵티마이저 Adam (7) | **Muon**(Muown) + Adam 혼합. Nesterov 0.95, Newton–Schulz 10회, grad clip 1.0, Adam $\beta_1 = \beta_2 = 0.95$ | **오답** |
| 26 | 스텝당 약 2B 토큰 (8) | 2.7B~3.7B | 오답 (초반 수치로 과소) |
| 27 | 턴 단위 credit: set / mul 재작성, 규칙 기반 action 경로 (12) | Penalty Module: Rule(수작업 또는 모델 judge) × Strategy(mask / set / scale / subtract / monitor) × 계층(segment / context / sequence), early stop | 구조는 적중. judge pass2 기반 턴 credit은 리포트에 없고, GAR은 "시퀀스 단위 advantage를 모든 토큰에 broadcast"라고 명시 |
| 28 | 30스텝은 계획된 예산 (11.1) | 5장 제목 "You Only RL Once", 30스텝 타임라인. 이유는 명시 안 함 | 모순 없음, 미확인 |
| 29 | 코드 routed-off 2개(m1dt, yfch)는 judge 대상 아님 (4.1) | 고통과율 과제는 **GRS**: $R = R^{\text{test}} \cdot S^{\text{sol}} \cdot S^{\text{beh}}$ (오프라인 루브릭) | **놓침.** 데이터로 재확인됨 (13.3) |

집계: 29개 중 적중 16, 방향이나 구조만 적중 2, 부분 적중 4, 미확정으로 둔 게 맞은 것 1, 모순 없음 1, 오답 4, 놓침 1. 뼈대(샘플링, advantage, judge 재배분, 부호별 토큰 페널티, 비동기)는 거의 다 맞췄고, 틀린 건 대시보드에 값이 직접 안 찍히는 부분(옵티마이저, IS 경계, 지목 대상)이었습니다.

### 13.2 리포트로 확정된 실제 수식

**GAR** (Groupwise Advantage Redistribution, 4.3.2절). $P = \{i : R_i = 1\}$, hack 확정 롤아웃은 먼저 $R_i \leftarrow 0$:

$$
A_i = R_i - \bar R, \qquad
\lambda = \min\!\left(K,\ \frac{\sum_{j \in P} A_j}{\sum_{j \in P} f_j A_j}\right), \qquad
A'_i = \begin{cases} \lambda f_i A_i, & i \in P \\ A_i, & i \notin P \end{cases}, \qquad
A^{\text{new}}_i = A'_i - \tfrac{1}{G}\textstyle\sum_j A'_j
$$

**GRS** (Groupwise Reward Synthesis, 4.3.1절). 통과율이 높은 일부 코드 과제:

$$
R_i = R^{\text{test}}_i \cdot S^{\text{sol}}_i \cdot S^{\text{beh}}_i
$$

**길이 페널티** (4.3.3절, Eq.4). $|P_q|/G > A$인 그룹에서 $\ell^\star_q = \text{Quantile}_{B/100}\{\ell_j : j \in P_q\}$:

$$
\tilde R_i = R_i - \mathbb{1}[i \in P_q]\ X \cdot \operatorname{clip}\!\left(\frac{\ell_i / \ell^\star_q - 1 - \delta}{s - \delta},\ 0,\ 1\right)^{\gamma}
$$

**segment 단위 행동 페널티** (Eq.5, 우리 5장의 signed). $h_{i,t} = 1$은 지목 토큰:

$$
\tilde A_{i,t} = \begin{cases}
\alpha (1 - h_{i,t}) A_i, & A_i > 0 \\
\big(\beta (1 - h_{i,t}) + \kappa h_{i,t}\big) A_i, & A_i < 0
\end{cases}, \qquad
\alpha = \min\!\left(\alpha_{\max},\ 1 + \frac{\sum_{H_+} A_i}{\sum_{C_+} A_i}\right), \quad
\beta = \max\!\left(\beta_{\min},\ 1 - \frac{(\kappa - 1)\sum_{H_-} |A_i|}{\sum_{C_-} |A_i|}\right)
$$

5장의 후보 표로 보면 양수 쪽은 "0으로 만들기"($\delta = A'_i$), 음수 쪽은 "비율"($\varepsilon = (\kappa - 1)|A'_i|$)의 혼합이었습니다. 관측 $\bar\varepsilon / \overline{|A'|} \approx 2.3$을 그대로 읽으면 $\kappa \approx 3.3$이지만, 지목 토큰이 $|A|$가 큰 롤아웃에 몰렸다면 더 작습니다 **[추정]**. clamp는 대칭 $\pm c$가 아니라 $\alpha_{\max}$(위)와 $\beta_{\min}$(아래) 한쪽씩이고, 관측 범위에서 $\alpha_{\max} > 1.003$, $\beta_{\min} < 0.896$입니다.

**손실** (Eq.1, 5.1절). prompt-mean 집계, 부호별 IS 마스크:

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{q}\!\left[\frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} r_{i,t}\, M_{i,t}\, \tilde A_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right], \qquad
r_{i,t} = \text{sg}\!\left[\frac{\pi_\theta(o_{i,t})}{\mu_{\theta_{\text{old}}}(o_{i,t})}\right]
$$

$$
M = \mathbb{1}\big[(A \ge 0 \wedge \epsilon^l_+ \le r \le \epsilon^h_+) \vee (A < 0 \wedge \epsilon^l_- \le r \le \epsilon^h_-)\big], \qquad [\epsilon^l_\pm, \epsilon^h_\pm]_{\text{init}} = [0.2,\ 5.0]
$$

엔트로피가 낮으면 양수 경계를 넓히고 음수 경계를 좁히며, 높으면 반대로 합니다. 대시보드의 `pg_tis_clipfrac_{pos,neg}_{low,high}` 네 태그가 이 네 경계에 해당합니다.

### 13.3 리포트 공식으로 대시보드 다시 맞추기

**$K$, $f(T_2)$, $f(T_3)$.** 이진 보상이고 cap에 안 걸리면 $\lambda = |P| / \sum_P f$입니다. 그러면 GAR 코드 데이터셋 9개의 `advantages/max`는 $(|P|, \{f\}, K)$ 조합으로 정확히 계산되는 유한한 값이어야 합니다. pro 30스텝 × 9개 = 270점을 조합 전수 탐색으로 맞췄습니다.

| 관측 최댓값 | 횟수 | 설명 | 계산 |
|---|---|---|---|
| 1.08333 | 73 | 통과 3, $f = (1, 0.625, 0.625)$ | $\tfrac{3}{2.25} \times 0.8125$ |
| 1.04348 | 10 | 통과 4, $f = (1, 0.625, 0.625, 0.625)$ | $\tfrac{4}{2.875} \times 0.75$ |
| 1.14023 | 6 | 통과 4, $f = (1, 0.625, 0.625, 0.2)$, **cap 발동** | $\lambda = 1.633 \to 1.5$. $0.75 \times (1.5 + 0.325/16)$ |
| 1.10795 | 4 | 통과 3, $f = (1, 1, 0.2)$ | $\tfrac{3}{2.2} \times 0.8125$ |
| 1.32344 | 5 | 통과 2, $f = (1, 0.2)$, **cap 발동** | $\lambda = 1.667 \to 1.5$. 재중심화로 $+0.0109$: $0.875 \times (1.5 + 0.2/16)$ |
| 1.26445 | 3 | 통과 3, $f = (1, 0.2, 0.2)$, **cap 발동** | $0.8125 \times (1.5 + 0.9/16)$ |

- **[확정]** $(f(T_2), f(T_3), K) = (0.625, 0.2, 1.5)$가 0.9 이상 고유값 39개 중 12개, 전체 270점 중 158점을 소수 5자리까지 맞춥니다. 무작위 값이 우연히 맞을 확률은 0.4%입니다. 경쟁 후보 $(0.6, 0.25)$는 7개에 그칩니다. 4.3의 회귀값(0.587~0.634, 0.24 ± 0.15)은 이 값의 근사였습니다.
- **[확정]** cap에 걸린 세 행은 "cap → 재중심화" 순서가 아니면 나올 수 없는 값이라 리포트의 순서도 데이터로 확인됩니다.
- 설명 안 되는 나머지(1.12246 × 26, 1.12617 × 19 등)는 길이 페널티로 성공 reward가 1보다 작아진 그룹이거나 r1 마스킹이 섞인 그룹으로 보입니다 **[추정]**.

**GRS 데이터셋.** 이진 보상이면 `advantages/max` × 16이 GAR 적용 전에는 정수여야 합니다. judge 대상이 아니었던 m1dt와 yfch는 30스텝 내내 비정수(예: 10.61, 5.63, 6.94)이고, `rewards/mean`도 코드 중 가장 높은 편(yfch 0.57~0.78)입니다. 고통과율 과제에 쓴다는 GRS의 연속 보상 $R^{\text{test}} S^{\text{sol}} S^{\text{beh}}$로 설명됩니다 **[사실상 확정]**. 256k 한도인 4onq는 대부분 정수라 GRS가 아닙니다.

### 13.4 길이 페널티: 남은 불일치

리포트의 길이 페널티는 **성공 롤아웃만** 깎습니다. $X \le 1$이면 성공 reward는 $1 - X \ge 0$ 밑으로 내려가지 않으므로, 11.3에서 본 "한도 근처 셀의 최솟값이 정확히 −0.8"은 Eq.4로 설명되지 않습니다. 두 가지로 읽을 수 있습니다.

| 해석 | 맞는 관측 | 안 맞는 관측 |
|---|---|---|
| (a) 리포트에 없는 절대 길이 overlong 감점이 따로 있음 (11.3 그대로) | 절대 길이 구간별로 단조. 400k 미만 722셀에서 음수 0건 | 리포트는 길이 페널티를 하나만 설명함. x7wh는 437k부터 음수 |
| (b) Eq.4의 $X \approx 1.8$: 매우 긴 성공은 $1 - 1.8 = -0.8$ | 리포트 수식만으로 설명됨. 기준이 데이터셋과 프롬프트마다 달라 x7wh 예외도 자연스러움. SWE 과제는 잘려도 레포 상태로 채점되므로 잘린 성공이 가능 | 성공이 실패(0)보다 낮아지는 설계라 공격적임. 256k 데이터셋은 한도에 닿아도 음수가 없음 |

집계값만으로는 둘을 가를 수 없습니다. 확실한 건 두 가지입니다. 11.3의 "−0.8 램프" 수식은 리포트 기준으로 확인되지 않으며, 성공 롤아웃에 거는 group-relative 페널티를 우리는 놓쳤습니다. 리포트 Figure 8(GAR 유무 비교)에서 GAR만으로도 턴 수와 길이 증가가 크게 억제되므로, 길이 제어는 이 페널티, GAR, prompt-mean 집계 세 가지가 함께 맡고 있었습니다.

### 13.5 대시보드로는 볼 수 없었던 것

- **Router freeze.** RL 중 MoE router를 고정. 풀어 두면 20스텝 동안 layer 9의 CV가 0.78에서 2.0으로, 차가운 expert가 0.5%에서 22%로 늘어남. 7장의 "죽은 expert 거의 없음"은 이 결과였습니다.
- **학습-추론 일치.** MXFP4 expert QDQ, R3(routing replay), top-p 후보 집합 replay. 8장에서 staleness 0일 때 바닥 KL을 0.002~0.003으로 본 것이 이 장치들의 잔차입니다.
- **reward hack 대응은 주로 학습 밖에서.** mid-training 정렬 데이터, 환경 정리(캐시, git 이력, 네트워크 격리), 탐색 전담 hack agent로 막힐 때까지 반복, 학습 중 오프라인 감사. 학습 안의 장치는 GAR의 hack → 0 하나이고, 확정 hack 비중은 전 구간 2% 미만. 11.3의 "hack 시도율 38%가 안 줄었다"는 시도 기준이라 이 수치와 모순되지 않습니다.
- **Multi-harness.** mini-harness 4개로 학습(대시보드 `harness-A`~`D`). 학습에 안 쓴 codex, claude code, mini-swe-agent의 DeepSWE 평균이 약 50%에서 66%로 오름.
- **스케줄링 수식.** 소스별 oversampling $p_i = \operatorname{clip}(c\, t_i - 1, p_{\min}, p_{\max})$, 가중치 $w_i = \alpha B_i / r_i + (1 - \alpha)(B_i - A_i)_+ / r_i$, $\alpha = 0.5$.
- **grader가 accept보다 먼저.** grader가 그룹 reward를 다시 쓴 뒤에 sampler가 통과율로 accept하고, 그 다음 hook이 길이 페널티 → advantage → shaping 순으로 적용합니다. advantage가 전부 0인 그룹은 기본으로 버립니다. 4.1 funnel을 "accept 뒤 judge"로 그린 것은 태그 집계 순서였을 뿐, 실제 파이프라인 순서와는 다릅니다.

---

## 14. GAGAR 논문 대조 (2026-09-30)

[arXiv 2609.32577](https://arxiv.org/abs/2609.32577) "Groupwise Agentic Grading and Advantage Redistribution for Code Agent RL" (Xiaomi LLM Core, 2026-09-26). 리포트 4.3.2절 GAR의 원 논문입니다. 이 장부터는 **논문을 레퍼런스로 삼고**, 대시보드 역산은 논문과 모순이 없는지만 확인합니다.

### 14.1 논문이 공개한 규칙과 상수

- **대상.** 동적 샘플링으로 남은 성공·실패 혼합 그룹. infra 때문에 채점할 수 없는 궤적은 그룹 통계 **전에** 마스킹하고, 평가 자체가 깨진 그룹은 학습에서 뺍니다. 외부·유출 답 의존이 확정되면 reward를 0으로 바꾼 뒤 그룹 통계를 다시 계산합니다. 품질 순위와는 별개인 무결성 교정입니다.
- **채점기.** RL 전 MiMo-V2.6-Pro SFT 체크포인트 하나를 Flash와 Pro 학습 모두에 씁니다. 처음에는 Claude Opus 5로 그룹당 약 2,000초가 걸렸고, SFT로 학습한 채점기로 약 600초까지 줄였습니다. 채점은 partial rollout과 겹쳐 돌립니다.
- **채점 절차.** 턴별 요약으로 볼 곳을 고르고, 궤적의 해당 부분을 읽고, 패치를 레포 코드·테스트 로그와 교차 확인하고, 필요하면 표적 검사를 직접 실행합니다. 부정 평가는 패치 위치, 궤적 이벤트, 실행 결과 중 하나를 근거로 인용해야 합니다. 실패 궤적도 비교 맥락으로 보지만 순위는 통과한 것만 받습니다.
- **점수.** 다섯 기준을 1~5 정수로 매기고, 초기 순위는 $W = 0.30\,s^{\text{app}} + 0.25\,s^{\text{prec}} + 0.20\,s^{\text{min}} + 0.15\,s^{\text{side}} + 0.10\,s^{\text{style}}$입니다. 채점기가 근거를 대고 순서를 바꿀 수 있고, 동률을 허용합니다.
- **티어** (이 순서로 판정)
  - $T_3$: 요청 안 한 재작성이나 테스트 맞춤 우회가 확정됨, 접근 적절성 최하점, 또는 최소성과 부작용 회피가 둘 다 2 이하
  - $T_1$: 나머지 중 모든 기준 4 이상이고 심각한 프로세스 문제·미해결 회귀가 없음
  - $T_2$: 그 외
- **factor** (Flash 설정). $k_i$는 티어 안에서 동률 그룹의 0부터 센 순위, $K_2$는 $T_2$ 안 동률 그룹 수:

$$
f_i = \begin{cases}
1, & T_1,\ k_i = 0 \\
f_{\text{runner}} = 0.9, & T_1,\ k_i > 0 \\
f_{\max} - (f_{\max} - f_{\min})\,\dfrac{k_i}{K_2 - 1}, & T_2,\ K_2 > 1 \quad (f_{\max} = 0.85,\ f_{\min} = 0.4) \\
f_{\max} = 0.85, & T_2,\ K_2 = 1 \\
f_{\text{low}} = 0.2, & T_3
\end{cases}
$$

- **재배분과 상한.** $S_+ = \sum_P A_i$, $\lambda = S_+ / \sum_P f_j A_j$, $\lambda_{\text{bnd}} = \min(\lambda, 1.5)$. $B_i = \lambda_{\text{bnd}} f_i A_i$ (통과) 또는 $A_i$ (실패), $A^{\text{train}}_i = B_i - \frac{1}{n}\sum_j B_j$. 상한이 안 걸리면 마지막 빼기는 0입니다.
- **비이진 보상** (식 8, 길이 페널티와 함께 쓸 때). $a_i = r_i - \bar r$, 통과 궤적은 $a_i^+ = \max(a_i, 0)$에 $f$를 곱해 재분배하고 $\lambda_{\text{bnd}} = \min(\sum_P a^+ / \sum_P f a^+,\ 1.5)$. 분모가 0이면 재배분하지 않습니다.
- **보상 공간 등가식** (A.3). 통과 $R'_i = \bar R + (1 - \bar R) f_i / \bar f_P$, 실패 $0$으로 두고 평균을 빼면 같은 advantage가 나옵니다. 보상을 그냥 $f_i$로 할인하는 방식은 실패 advantage와 통과 사이 비율을 바꾸므로 등가가 아닙니다.

### 14.2 대시보드 역산과의 대조: 모순 없음

| 역산 (13.3) | 논문 | 판정 |
|---|---|---|
| 상한 $K = 1.5$ | $\lambda_{\max} = 1.5$ | 적중 |
| $f(T_3) = 0.2$ | $f_{\text{low}} = 0.2$ | 적중 |
| $f(T_2) = 0.625$ 고정 | $T_2$는 순위에 따라 0.85 → 0.4 선형 | **모순 없음.** `advantages/max`는 통과 $f$의 합에만 달려 있고, 순위가 모두 다른 $T_2$들의 평균은 정확히 $(0.85 + 0.4)/2 = 0.625$ |
| $f(T_1) = 1$ | 1등 1, 나머지 0.9 | 부분 |
| cap → 재중심화 순서 | 식 7 | 적중 |
| hack → reward 0 | 같음 | 적중 |
| 판정 시간 평균 655초 | SFT 채점기 약 600초 | 적중 |
| r1 마스킹 | $f \in (0, 1]$이라 $f = 0$은 없음. infra 무효 궤적은 통계 전 마스킹 | 13.1의 17번처럼 r1을 infra 면책으로 읽으면 모순 없음 |

논문 규칙(Flash 상수)으로 13.3 관측값을 다시 계산했습니다 (`papers/gar_paper_fit.py`, 조합 전수 탐색, $n = 16$):

| 관측 최댓값 | 횟수 | 논문 규칙의 그룹 구성 |
|---|---|---|
| 1.08333 | 73 | 통과 3: $T_1$ + $T_2$ 두 순위, $f = (1, 0.85, 0.4)$ |
| 1.04348 | 10 | 통과 4: $T_1$ + $T_2$ 세 순위, $f = (1, 0.85, 0.625, 0.4)$ |
| 1.14023 | 6 | 통과 4: $f = (1, 0.85, 0.4, 0.2)$, 상한 발동 |
| 1.10795 | 4 | 통과 3: $T_1$ 동률 둘 + $T_3$, $f = (1, 1, 0.2)$ |
| 1.32344 | 5 | 통과 2: $f = (1, 0.2)$, 상한 발동 |
| 1.26445 | 3 | 통과 3: $f = (1, 0.2, 0.2)$, 상한 발동 |
| **1.12617** | **19** | 통과 4: $T_1$ + $T_2$ 두 동률 그룹(1개, 2개), $f = (1, 0.85, 0.4, 0.4)$. **0.625 모델로는 설명 못 하던 값** |
| 1.12246 | 26 | 여전히 설명 안 됨. 식 8(길이 페널티로 비이진 보상) 그룹으로 추정 **[추정]** |

Pro 대시보드 값이 Flash 상수로 맞으므로, Pro도 같은 상수를 썼을 가능성이 높습니다 **[추정]**.

### 14.3 실험 결과 (Flash 310B, Code 단독, 배치 128 × 16, token-mean 집계)

- GAR 없는 기준선은 28스텝에서 중단: DeepSWE가 20스텝 58.5%에서 50.2%로 떨어지고 궤적 길이가 폭증. GAR은 28스텝 62.2%, 44스텝 63.4%. SWE-bench Pro는 기준선이 약 59%에서 정체, GAR은 52스텝 62.5%
- 28스텝 DeepSWE 평균 턴 132.3 → 111.6 (−15.6%), 토큰 191.9k → 172.9k (−9.9%)
- 합 보존 없이 $f$만 곱한 ablation: 엔트로피 0.36 → 0.91, 학습 롤아웃 길이 47k → 114k, pg_loss 평균 0.030 (전체 방법 0.002). 합 보존이 안정성의 핵심
- 품질 감사 (Claude Opus 5, 30과제): 가중 점수 3.70 → 4.03, 통과 후보 간 평균 승률 69.8%, $T_1$ 비중 25.4% → 34.3%

### 14.4 우리 구현에 주는 것

- GAR은 채점 모델만 빼면 논문대로 구현할 수 있습니다. 채점기는 강한 공개 모델로 대신하면 됩니다(논문도 처음에 Claude Opus 5).
- infra 무효 궤적을 그룹 통계 전에 빼는 원칙은 `FIXES.md` 1번 수정과 같습니다.
- 식 8은 통과 궤적의 음수 advantage를 0으로 자릅니다. 길이 페널티 때문에 평균보다 낮아진 통과 궤적은 길이 신호를 일부 잃고, 이때는 양수 합 보존도 정확히 성립하지 않습니다(논문도 명시).
- 논문의 Code 단독 실험은 token-mean 집계, 리포트 본 학습은 prompt-mean입니다.

---

## 부록: 근거가 된 주요 태그

| 결론 | 태그 |
|---|---|
| 그룹 크기 16, std 나눗셈 없음 | `critic/*/advantages/max` = 0.9375 |
| reward에 KL 없음 | `critic/score/*` ≡ `critic/rewards/*` |
| accept 조건 | `dynsam/num_measurable`, `dynsam/passrate/{zero,one}`, `dynsam/*/num_accepted/step` |
| 이월 관계 | `dynsam/*/num_accepted/{step,carryover,held}` |
| 재중심화 | `penalty/stage_credit_group/select_adv_group_sum_abs_mean` |
| factor와 renorm | `select_factor_mean`, `select_renorm_k_mean`, `select_renorm_capped_rate` |
| signed 스케일 공식 | `penalty/signed/{pos,neg}_scale`, `train/adv_{pos,neg}_sum_{pre,post}_penalty` |
| 단일 업데이트 | `training/actor_optimizer_steps`, `actor/ppo_kl`, `actor/pg_clipfrac` |
| 토큰 평균 loss | `actor/pg_loss` vs `critic/advantages/mean` |
| TIS threshold | `actor/pg_tis_clipfrac` vs `train_infer_diff/new_infer/F(tau=10)` |
| staleness 지배 | `partial/avg_staleness` vs `train_infer_diff/new_infer/kl` |
| judge 라우팅 범위 | `api/live` 의 데이터셋별 `judged` 필드 |
| judge funnel | `groups_total`, `routed/*`, `select_groups_skipped_uniform`, `judge_pending`, `groups_attempted`, `groups_failed_{pod,select}`, `groups_judged` |
| hack/티어는 통과만 심사 | `select_hack_attempt` / `select_hack_attempt_rate` 의 분모 = 11,903 |
| r1은 실패 대상 | `select_r1_masked` / `select_r1_rate` 의 분모 ≈ 1.1 × (rows − passes) |
| 티어별 $f$ | `select_factor_mean` ~ `select_tier_share_*` 회귀, 217점, $R^2$ 0.953 |
| soft overlong 램프 | `ctx_total_length/*/max` vs `critic/*/rewards/min` 셀 단위 단조 관계, `clip_ratio`로 한도 도달 확인 |
| 종료 사유 | `api/status` `run.mode`, `run.end`, `trained_cum` = 30 × 25,088 |
