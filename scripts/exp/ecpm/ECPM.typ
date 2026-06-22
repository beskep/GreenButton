#import "@preview/touying:0.7.4": *
#import "@preview/metropolyst:0.1.0": *
#import "@preview/splash:0.5.0": tol-bright, tol-vibrant

#show: metropolyst-theme.with(
  font: "Source Han Sans KR",
  accent-color: tol-vibrant.orange,
  header-background-color: rgb("#546E7A"),
  header-weight: "bold",
  header-right: text(
    size: 0.6em,
    fill: luma(100%, 80%),
    weight: 400,
    utils.display-current-heading(level: 1),
  ),
  footer-progress: true,
  date-size: 1em,
  main-background-color: white,
  config-info(
    author: [ ],
    date: [2026-06-18],
    title: [그린버튼 ECPM 모델 분석],
    subtitle: [],
  ),
)

#set text(
  font: ("Source Sans 3", "Source Han Sans KR"),
  weight: 400,
  size: 16pt,
  lang: "ko",
)

#show heading.where(level: 1): set heading(numbering: "1.")
#show outline: set text(size: 13.5pt)

#show math.equation: set text(
  font: ("New Computer Modern Math", "Source Han Sans KR"),
)
#show raw: set text(font: ("Sarasa Term K", "AdwaitaMono Nerd Font"))
#show raw.where(block: false): box.with(
  fill: luma(250),
  stroke: luma(200) + 0.5pt,
  inset: (x: 2pt, y: 1pt),
  outset: (y: 2pt),
  radius: 2pt,
)

#set enum(spacing: 1em)
#set list(spacing: 1.2em, marker: (sym.bullet, sym.bullet.stroked))

#show strong: it => { text(it.body, weight: 500, tol-bright.blue) }
#show emph: it => {
  show regex("[\p{Hangul}\p{Han}\p{Hiragana}\p{Katakana}]"): it => {
    box(skew(it, ax: -12deg, reflow: false), width: 0.9em)
  }
  text(it, fill: tol-bright.blue)
}
#show image: set align(center)

#show "->": sym.arrow.long

// =============================================================================

#let r2 = $r^2$

#let ti = $T_i$
#let tiw = $T_italic("iw")$
#let te = $T_e$
#let dt = $Delta T$
#let pv = $P_v$
#let eui = $"EUI"$

#let eb = $E_b$
#let bh = $beta_h$
#let bc = $beta_c$
#let th = $T_h$
#let tc = $T_c$

// =============================================================================

#title-slide()

== 목차 <touying:hidden>
#components.adaptive-columns(outline(depth: 2, title: none))

= 전처리 & EDA

== 데이터 전처리

#cols(columns: (1fr, 1fr))[
  === 한전 파주지사 KEPCO
  - BEMS 전력 사용량 -> EUI [kWh/m²]
  - BEMS 실내 온도 (1층 민원실) [°C]

  === 한국에너지공단 KEA
  - AMI 전력 사용량 -> EUI [kWh/m²]
  - BEMS 실내 온도 (다수 지점의 중위수) [°C]
  - _새벽 ESS 충전 문제 해결 필요_
][
  === 공통
  - 실내 온도 근무시간 평균 계산 (#ti or #tiw)
  - 기상청 ASOS 기상자료 (파주·울산)
    - 일간 기온 [°C] (#te)
    - 일사량 [MJ/m²s] ($I$)
    - 수증기 분압 [hPa] (#pv)
  - 주말, 공휴일 데이터 제외
]

== 요일별 사용량

#image("assets/00.EDA.EUI-weekday.svg", height: 100%)

== Pair Grid Plot: KEPCO
#slide(align: top, composer: 2)[
  #image("assets/00.EDA.WeatherPairGrid-KEPCO.png")
][
  - #te, #ti, $dt=te-ti$, #pv 간 강한 상관관계
  - 모두 CPM(#te vs #eui)와 유사한 모델 생성 가능
]

== Pair Grid Plot: KEA
#slide(align: top, composer: 2)[
  #image("assets/00.EDA.WeatherPairGrid-KEA.png")
][
  - KEPCO보다 큰 산포
]

// =============================================================================

= Change Point Model (CPM) 분석

== CPM 분석 방법

$ E = eb + bh (th - te)^+ + bc (te - tc)^+ $

- 일반적인 5-point CPM 식 적용 (#te vs #eui)
- Differential Evolution 최적화 방법으로 change point 결정 (`scipy.optimize.differential_evolution`)
  - 잔차 제곱합을 최소화하는 #th, #tc 탐색
  - 유효하지 않은 모델을 제외하기 위해 잔차에 다음 항목 추가 (ECPM 최적화에도 적용)
    - $c times max(0, #th - #tc)^2$ -> #th;가 #tc;보다 크면 패널티
    - $c times sum(min(0, beta_i)^2)$ -> 기울기 $beta$가 0보다 작으면 패널티
  - (상수 $c$는 $10^8$ 적용)

== [KEPCO] CPM

#cols(columns: 2)[
  #image("assets/KEPCO T=Te model=CPM scatter.svg")
][
  #text(raw(read("assets/KEPCO T=Te model=CPM OLS.txt")), size: 0.65em)
]

== [KEPCO] CPM 잔차
#[
  #set list(spacing: 0.8em)
  - 난방(주황색), 기저(회색), 냉방(파란색) 구간 구분
  - *실내온도 #ti;와 잔차 간 상관관계 보이지 않음* -> CPM에 #ti 항 추가가 어려움
  - 일사량($I$)과 잔차 음의 상관관계 (태양광 영향 가능성)
]
#image("assets/KEPCO T=Te model=CPM residual.svg", height: 75%)

== [KEA] CPM

#cols(columns: 2)[
  #image("assets/KEA T=Te model=CPM scatter.svg")
][
  #text(raw(read("assets/KEA T=Te model=CPM OLS.txt")), size: 0.65em)
]

== [KEA] CPM 잔차

- 실내온도, 일사량 KEPCO와 같은 경향
#image("assets/KEA T=Te model=CPM residual.svg", height: 90%)

// =============================================================================

= Extended Change Point Model #text(size: 0.8em)[(ECPM)] 분석

== Additive 모델

$
  E = eb & + bh  &   (th - te)^+ + bc & (te - tc)^+ \
         & + bh' & (th' - ti)^+ + bc' & (ti - tc')^+
$
- #te;와 같은 형태의 #ti change point 항목 추가
- $th, tc, th', tc'$ 각각 최적화

== [KEPCO] Additive ECPM

- #r2;가 0.8199에서 0.8257로 소폭 증가
- `x3`=$(th' - ti)$의 `coef`가 0 \
  -> `x3` 항목은 영향 없음 (냉·난방·기저 모든 구간에 $(ti - tc')^+$만 영향)

#cols(columns: 2)[
  #image("assets/KEPCO T=Te model=ADD scatter.svg")
][
  #text(raw(read("assets/KEPCO T=Te model=ADD OLS.txt")), size: 0.6em)
]

== [KEPCO] Additive 잔차

#image("assets/KEPCO T=Te model=ADD residual.svg", height: 80%)

== [KEA] Additive ECPM

- #r2 0.5196 -> 0.5295
- `x4`=$tc'$의 `coef`가 0
- CPM 대비 냉방 구간에 변화 없음 -> 모델 일관성 없음

#cols(columns: 2)[
  #image("assets/KEA T=Te model=ADD scatter.svg")
][
  #text(raw(read("assets/KEA T=Te model=ADD OLS.txt")), size: 0.6em)
]

== [KEA] Additive 잔차

#image("assets/KEA T=Te model=ADD residual.svg", height: 90%)

// =============================================================================

== Multiplicative 모델

$ E = & eb + bh' (dt + th') (th - te)^+ + bc' (dt + tc') (te - tc)^+ $

- $dt = ti - te$ #h(1em) _(양수 변수를 쓰기 위해 $te - ti$ 대신 사용)_
- 냉난방 민감도 $bh, bc$가 실내외 온도차 #dt;에 영향을 받는다 가정
  - $bh = bh' times (dt + th')$
  - $bc = bc' times (dt + tc')$
  - 물리적 근거는 없음
- 경희대 모델의 change point shift 문제를 해결하기 위해 냉난방 #dt 항에 $th'$와 $tc'$를 더해줌

== [KEPCO] Multiplicative ECPM

- #r2; 0.8199 -> 0.8136 하락
- CPM 직선과 Multiplicative ECPM 예측 결과 간 편차가 작음 -> #ti;의 영향이 제한적
- `x3`, `x4` ($th', tc'$)가 임의로 설정한 bound (최대 50°C)에 수렴해서
  *#ti;의 영향을 최소화* #text(size: 0.9em)[(bound 범위를 늘려도 같은 결과)]

#cols(columns: 2)[
  #image("assets/KEPCO T=Te model=MULT scatter.svg")
][
  #text(raw(read("assets/KEPCO T=Te model=MULT OLS.txt")), size: 0.6em)
]

== [KEPCO] Multiplicative 잔차

#image("assets/KEPCO T=Te model=MULT residual.svg", height: 80%)

== [KEA] Multiplicative ECPM

- #r2 0.5196 -> 0.5214
- KEPCO와 마찬가지로 #ti;의 영향이 제한적

#cols(columns: 2)[
  #image("assets/KEA T=Te model=MULT scatter.svg")
][
  #text(raw(read("assets/KEA T=Te model=MULT OLS.txt")), size: 0.6em)
]

== [KEA] Multiplicative 잔차

#image("assets/KEA T=Te model=MULT residual.svg", height: 90%)

// =============================================================================

= 추가 테스트 모델

== #sym.Delta;T CPM

$ E = eb + bh (dt - te)^+ + bc (dt - tc)^+ $

- 일반 CPM에 외기온 #te 대신 실내외 온도차 #dt;를 적용해서 실내온도 반영
- (경희대 첫 시도에선 고정된 #ti;를 가정했으나, 본 분석에선 실측한 #ti 반영)

== [KEPCO] #sym.Delta;T CPM

- #r2; 0.8199 -> 0.8237

#cols(columns: 2)[
  #image("assets/KEPCO T=dT model=CPM scatter.svg")
][
  #text(raw(read("assets/KEPCO T=dT model=CPM OLS.txt")), size: 0.6em)
]

== [KEPCO] #sym.Delta;T CPM 잔차

#image("assets/KEPCO T=dT model=CPM residual.svg", height: 80%)

== [KEA] #sym.Delta;T CPM

- #r2 0.5196 -> 0.4671

#cols(columns: 2)[
  #image("assets/KEA T=dT model=CPM scatter.svg")
][
  #text(raw(read("assets/KEA T=dT model=CPM OLS.txt")), size: 0.6em)
]

== [KEA] #sym.Delta;T CPM 잔차

#image("assets/KEA T=dT model=CPM residual.svg", height: 90%)

// =============================================================================

== 기상자료 추가 CPM

$ E = eb + bh (th - te)^+ + bc (te - tc)^+ + beta_I I + beta_P pv $

- CPM에 일사량 $I$, 수증기 분압 #pv 독립변수 추가
- 서비스 이용에는 제한적이나, 모델 정확도를 개선할 것으로 기대

== [KEPCO] 기상자료 추가 CPM

=== $I, pv$ 추가
- #r2; 0.8199 -> 0.8423
- `x4` (#pv)의 p-value = 0.063

#cols(columns: 2)[
  #image("assets/KEPCO T=Te model=CPM_ExtI+Pv scatter.svg")
][
  #text(raw(read("assets/KEPCO T=Te model=CPM_ExtI+Pv OLS.txt")), size: 0.6em)
]

== [KEPCO] $I, pv$ 개별 추가 모델 비교

#cols(columns: (1fr, 1fr))[
  === $I$ 추가
  #text(raw(read("assets/KEPCO T=Te model=CPM_ExtI OLS.txt")), size: 0.6em)
][
  === $pv$ 추가
  #text(raw(read("assets/KEPCO T=Te model=CPM_ExtPv OLS.txt")), size: 0.6em)
]

== [KEPCO] 기상자료 추가 CPM 잔차 ($I, pv$ 추가)

#image("assets/KEPCO T=Te model=CPM_ExtI+Pv residual.svg", height: 80%)

== [KEA] 기상자료 추가 CPM

=== $I, pv$ 추가
- #r2 0.5196 -> 0.6130

#cols(columns: 2)[
  #image("assets/KEA T=Te model=CPM_ExtI+Pv scatter.svg")
][
  #text(raw(read("assets/KEA T=Te model=CPM_ExtI+Pv OLS.txt")), size: 0.6em)
]

== [KEA] $I, pv$ 개별 추가 모델 비교

#cols(columns: (1fr, 1fr))[
  === $I$ 추가
  #text(raw(read("assets/KEA T=Te model=CPM_ExtI OLS.txt")), size: 0.6em)
][
  === $pv$ 추가
  #text(raw(read("assets/KEA T=Te model=CPM_ExtPv OLS.txt")), size: 0.6em)
]

== [KEA] 기상자료 추가 CPM ($I, pv$ 추가)

#image("assets/KEA T=Te model=CPM_ExtI+Pv residual.svg", height: 90%)

= 모델 비교

== KEPCO

#image("assets/01.KEPCO.svg", height: 100%)

== KEA

#image("assets/01.KEA.svg", height: 100%)
