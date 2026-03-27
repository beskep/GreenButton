// 2026-03-24 경희대학교 에너지 예측 분석안

#import "@preview/marginalia:0.3.1"

#set document(
  title: [그린버튼 과제 에너지 사용량 예측 모델 분석안],
  author: "박관용",
  date: datetime(year: 2026, month: 3, day: 24),
)
#set text(font: "Source Han Sans KR", lang: "ko")
#set par(first-line-indent: 1em, leading: 0.8em)
#set par(first-line-indent: (amount: 1em, all: true), leading: 0.8em)

#show title: set align(center)

#set heading(numbering: "1.")
#show heading: set text(fill: rgb("263238"))
#show heading: set block(above: 2em, below: 1em)

#set table(stroke: none)
#show table: set text(size: 10pt)

#show math.equation: set block(above: 1.8em, below: 1.8em)

#show: marginalia.setup.with(
  inner: (far: 5mm, width: 5mm, sep: 5mm),
  outer: (far: 5mm, width: 50mm, sep: 5mm),
  top: 20mm,
  bottom: 20mm,
)
#let note = marginalia.note.with(
  text-style: (size: 10pt, font: "Source Han Sans KR"),
  par-style: (first-line-indent: 0em),
)

#let ti = $T_italic("int")$
#let te = $T_italic("ext")$

#let kwhm2 = text(size: 9pt)[[kWh/m²]]
#let kwhm2c = text(size: 9pt)[[kWh/m²℃]]
#let degc = text(size: 9pt)[[℃]]

// =============================================================================

#marginalia.wideblock({
  title()
  v(1em)
  align(right)[2026-03-24 박관용 #h(2em)]
})

= CPM 모델 분석

ASHRAE Inverse Modeling Toolkit 등 Change Point Model 분석 방법과
일간 데이터로 건물의 냉난방 민감도 분석.

#let n1 = note(dy: -1.5em)[
  $E$ 에너지 사용량 #kwhm2 \
  $E_b$ 기저 사용량 #kwhm2 \
  $E_c, E_h$ 냉·난방 사용량 #kwhm2
]
#let n2 = note[
  $X^+ = max(0, X)$ \
  $te$ 외기 온도 #degc \
  $T_c, T_h$ 냉·난방 균형점 온도 #degc \
  $beta_c, beta_h$ 냉·난방 민감도 #kwhm2c
]

$
  E = & E_b + E_c    &                 + E_h &              & #h(1em)#n1 \
    = & E_b + beta_c & (te - T_c)^+ + beta_h & (T_h - te)^+ & #h(1em)#n2
$

에너지 사용량은 AMI 또는 BEMS 데이터, 외기 온도는 기상청 또는 BEMS 데이터 이용
(간절기 기저 부하 포함).

= 실내 온도#{ sym.dash.en }냉난방 민감도 분석

실내 온도 설정에 따른 냉난방 에너지 사용량의 변화를 평가하기 위해
냉난방 민감도를 실내 온도에 따른 함수로 해석.

$
  beta_c = & f_(beta c) (ti, te, beta_c, ...) #h(1em)#note[ #ti 실내 온도 [℃] ] \
  beta_h = & f_(beta h) (ti, te, beta_h, ...)
$

가장 간단한 모델로 냉난방 민감도가 실내외 온도차에 단순 비례한다는 가정을 먼저 분석.


$
  beta_c & = c_c (te - ti)^+ #h(1em)#note[$c_c$, $c_h$는 상수] \
  beta_h & = c_h (ti - te)^+
$

분석에 실증 데이터 (동·하절기) 및 일부 BEMS 실내 온도 계측 결과 사용.

= 에너지 사용량 추정 모델 (eCPM)

분석한 모델과 주어진 기온($te$), 설정 온도($ti$)를 통해 에너지 사용량#note[$E(ti, te)$]
또는 절감률#note[$1-E(ti, te) \/ E(te)$] 예측.

$
  E(ti, te) = E_b & + f_(beta c) (...) times (te - T_c)^+ \
                  & + f_(beta h) (...) times (T_h - te)^+
$

단순 비례 모델이 타당한 경우,

$
  E(ti, te) = E_b & + c_c (te - ti)^+ (te - T_c)^+ \
                  & + c_h (ti - te)^+ (T_h - te)^+
$
