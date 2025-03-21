from __future__ import annotations

import io

import polars as pl
import polars.testing
import pytest

from greenbutton import sensors

TR7 = """"Date/Time","Date/Time","No.1","No.2"
"Date/Time","Date/Time","TR-72wf Ch.1","TR-72wf Ch.2"
"","","°C","%"
"2024-03-20 13:51:10","45371.5771990741","25.2","30.3"
"2024-03-20 13:53:10","45371.5785879630","24.8","25.6"
"2024-03-20 13:55:10","45371.5799768519","25.0","24.0"
"2024-03-20 13:57:10","45371.5813657407","25.1","24.1"
"2024-03-20 13:59:10","45371.5827546296","25.2","22.7"
"2024-03-20 14:01:10","45371.5841435185","25.2","21.9"
"2024-03-20 14:03:10","45371.5855324074","25.3","21.2"
"2024-03-20 14:05:10","45371.5869212963","25.2","21.1"
"2024-03-20 14:07:10","45371.5883101852","25.1","21.1"
"2024-03-20 14:09:10","45371.5896990741","25.1","21.1"
"""


PMV_TESTO = """
날짜/시간,PMV,PPD [%],159 [°C],159 [%RH],177 TC1 [°C],840 [m/s],159 [bar],159 [ppm],159 이슬점 [°C],159 습구 [°C],159 [g/m³],177 [bar],177 [hPa],840 [°C],840 [bar],
24. 2. 26. 10:55:57 오전,-2.0,76.8,19.7,39.5,17.1,0.02,1.0232,807,5.6,12.0,6.72,1.0251,0.007,20.2,1.0235,
24. 2. 26. 10:56:21 오전,-1.9,72.1,19.8,38.7,17.5,0.02,1.0233,844,5.4,11.9,6.62,1.0251,0.007,20.3,1.0235,
24. 2. 26. 10:57:21 오전,-1.8,67.0,20.0,38.7,18.2,0.02,1.0232,873,5.5,12.1,6.70,1.0250,0.007,20.4,1.0235,
24. 2. 26. 10:58:21 오전,-1.7,61.8,20.1,37.9,18.8,0.02,1.0231,864,5.3,12.0,6.60,1.0250,0.007,20.6,1.0234,
24. 2. 26. 10:59:21 오전,-1.7,61.8,20.1,37.0,19.1,0.04,1.0231,818,5.0,11.9,6.44,1.0250,0.007,20.6,1.0235,
24. 2. 26. 11:00:21 오전,-1.6,56.3,20.2,36.7,19.4,0.03,1.0231,790,5.0,11.9,6.43,1.0249,0.007,20.7,1.0235,
24. 2. 26. 11:01:21 오전,-1.5,50.9,20.3,36.7,19.7,0.05,1.0232,776,5.0,12.0,6.47,1.0249,0.007,20.8,1.0236,
24. 2. 26. 11:02:21 오전,-1.5,50.9,20.4,36.6,19.8,0.03,1.0232,806,5.1,12.1,6.49,1.0249,0.007,20.9,1.0236,
24. 2. 26. 11:03:21 오전,-1.4,45.5,20.5,36.2,20.0,0.01,1.0231,804,5.0,12.1,6.45,1.0249,0.007,21.0,1.0235,
24. 2. 26. 11:04:21 오전,-1.4,45.5,20.6,36.2,20.1,0.04,1.0230,787,5.1,12.2,6.49,1.0249,0.007,21.1,1.0234,
전체 평균,-1.2,35.2,21.2,30.9,20.9,0.03,1.0221,727,3.4,11.8,5.74,1.0236,0.010,21.9,1.0224,

< 일반 정보 >,
앱 버전,측정 프로그램 이름,측정 주기,설명,
V 14.51.14.62512,실내 쾌적도 - PMV/PPD,24. 2. 26. 오후 4:28,,

< 기업 >,
구/동,국가,고객 데이터,이메일 주소,성명,기술자 이름,전화,팩스,홈페이지 주소,주소,
,,,,,,,,,,

< 고객 >,
구/동,국가,고객 데이터,이메일 주소,성명,기술자 이름,전화,팩스,홈페이지 주소,주소,
,,,,,,,,,,

< 측정기 정보 >,
장치 이름,일련번호,버전,
testo 400,63647177,V 14.51.14.62512,

< 측정기 >,
성명,일련번호,펌웨어 버전,측정 단위,
CO₂,58627159,V 0.8.16,온도,
CO₂,58627159,V 0.8.16,상대 습도,
CO₂,58627159,V 0.8.16,압력,
CO₂,58627159,V 0.8.16,CO₂ 농도(첫 경보 한계),
난류도,63355840,V 1.5.0,풍속,
난류도,63355840,V 1.5.0,온도,
난류도,63355840,V 1.5.0,압력,
testo 400,63647177,V 1.1.4,압력,
testo 400,63647177,V 1.1.4,차압,
testo 400,63647177,V 1.1.4,온도 TC1,

< 측정점 >,
성명,시스템 번호,시스템 타입,시스템 제조업체,일련 번호 시스템,제조일,메모,
,,,,,,,

< 측정 파라미터 >,
측정 모드,측정 주기,의복 단열,에너지 대사량,시작 시간,종료 시간,기간,
시간적,1 분 0 초,일반 비즈니스 복장 (0.154 m²K/W / clo=1.00),편안히 앉은 상태 (52 W/m² / met=0.9),24. 2. 26. 10:55:57 오전,24. 2. 26. 4:23:21 오후,0 d 5 시 27 분 24 초,

< 이미지 >,
성명,
"""  # noqa: E501


PMV_DELTA_OHM = """Dump Log n.= 24.07.18_11.00.07A
Model HD32.3TC
/*
PMV Index
Firm.Ver.=01.60
Firm.Date=2024/05/02
SN=24017899
User ID=DELTAOHM
Cal.=Factory
MET= 1.00
CLO= 0.69
Ch.1;Probe = Pt100 Tg_150;Probe cal.=2024/06/19;Probe SN=24017348
Ch.2;Probe = Hot wire ;Probe cal.=2024/04/19;Probe SN=24010781
Ch.3;Probe = RH-Pt100 R;Probe cal.=2024/06/11;Probe SN=24015698
Ch.4;Probe = not present;Probe cal.=not present;Probe SN=not present
Ch.5;Probe = not present;Probe cal.=not present;Probe SN=not present
Ch.6;Probe = not present;Probe cal.=not present;Probe SN=not present
Ch.7;Probe = not present;Probe cal.=not present;Probe SN=not present
Ch.8;Probe = not present;Probe cal.=not present;Probe SN=not present
*/
Sample interval= 300sec;Tw;C;Tg;C;Ta;C;Pr;hPa;RH;%;Va;m/s;Tr;C;WBGT(i);C;WBGT(o);C;WCI;C;PMV;;PPD;%;CO2;ppm;PM1.0;ug/m3;PM2_5;ug/m3;PM10;ug/m3;COV2H_50;%;COV2D_50;%;COV2H_99.99;%;COV2D_99.99;%;COV2H_99.9999;%;COV2D_99.9999;%;COV2H_99.999999;%;COV2D_99.999999;%;HI;C;UTCI;C;PET;C;VOC;;
Date=2024/07/18 11:00:07;19.6;26.3;26.1;ERR.;54.4;0.03;26.3;21.6;21.6;ERR.;0.49;10.0;ERR.;ERR.;ERR.;ERR.;  7.83;  0.33;104.01;  4.33;156.02;  6.50;208.02;  8.67;  26.8;  26.4;  26.7;ERR.
Date=2024/07/18 11:05:07;19.6;26.2;25.8;ERR.;55.6;0.05;26.3;21.6;21.5;ERR.;0.45;9.2;ERR.;ERR.;ERR.;ERR.;  7.83;  0.33;104.04;  4.33;156.06;  6.50;208.08;  8.67;  26.6;  26.2;  26.6;ERR.
Date=2024/07/18 11:10:07;19.6;26.0;25.8;ERR.;55.7;0.03;26.1;21.5;21.5;ERR.;0.42;8.7;ERR.;ERR.;ERR.;ERR.;  7.81;  0.33;103.84;  4.33;155.75;  6.49;207.67;  8.65;  26.6;  26.2;  26.6;ERR.
Date=2024/07/18 11:15:07;19.6;26.0;25.8;ERR.;55.8;0.06;26.1;21.5;21.5;ERR.;0.42;8.7;ERR.;ERR.;ERR.;ERR.;  7.80;  0.32;103.63;  4.32;155.45;  6.48;207.26;  8.64;  26.6;  26.2;  26.5;ERR.
Date=2024/07/18 11:20:07;19.3;26.0;25.8;ERR.;54.1;0.04;26.1;21.3;21.3;ERR.;0.40;8.3;ERR.;ERR.;ERR.;ERR.;  8.06;  0.34;107.10;  4.46;160.66;  6.69;214.21;  8.93;  26.5;  26.1;  26.4;ERR.
Date=2024/07/18 11:25:07;19.4;26.0;25.8;ERR.;54.5;0.05;26.1;21.4;21.4;ERR.;0.41;8.5;ERR.;ERR.;ERR.;ERR.;  8.00;  0.33;106.29;  4.43;159.43;  6.64;212.57;  8.86;  26.6;  26.1;  26.4;ERR.
Date=2024/07/18 11:30:07;19.2;26.0;25.8;ERR.;53.9;0.03;26.1;21.2;21.2;ERR.;0.40;8.3;ERR.;ERR.;ERR.;ERR.;  8.09;  0.34;107.51;  4.48;161.27;  6.72;215.02;  8.96;  26.5;  26.0;  26.4;ERR.
Date=2024/07/18 11:35:07;19.3;26.0;25.7;ERR.;54.9;0.03;26.0;21.3;21.3;ERR.;0.38;8.0;ERR.;ERR.;ERR.;ERR.;  8.00;  0.33;106.30;  4.43;159.44;  6.64;212.59;  8.86;  26.5;  26.0;  26.4;ERR.
Date=2024/07/18 11:40:07;19.5;25.9;25.7;ERR.;56.1;0.09;26.0;21.4;21.4;ERR.;0.36;7.7;ERR.;ERR.;ERR.;ERR.;  7.82;  0.33;103.85;  4.33;155.77;  6.49;207.69;  8.65;  26.5;  26.1;  26.4;ERR.
-->End of Log Session 24.07.18_11.00.07 <--
"""  # noqa: E501


def test_read_tr7():
    source = io.StringIO(TR7)
    df = sensors.read_tr7(source, unpivot=True)

    assert df.height == 20  # noqa: PLR2004
    for column in ['datetime', 'variable', 'value', 'unit']:
        assert column in df.columns


@pytest.mark.parametrize('source', ['str', 'bytes'])
def test_read_testo_pmv(source):
    src: io.StringIO | io.BytesIO = (
        io.StringIO(PMV_TESTO) if source == 'str' else io.BytesIO(PMV_TESTO.encode())
    )
    df = sensors.TestoPMV(src).dataframe

    for column in ['datetime', 'variable', 'value', 'unit']:
        assert column in df.columns


def test_read_testo_pmv_format_error():
    with pytest.raises(sensors.DataFormatError):
        print(sensors.TestoPMV(io.StringIO(TR7)).dataframe)


@pytest.mark.parametrize('source', ['str', 'bytes'])
def test_read_delta_ohm_pmv(source):
    src: io.StringIO | io.BytesIO = (
        io.StringIO(PMV_DELTA_OHM)
        if source == 'str'
        else io.BytesIO(PMV_DELTA_OHM.encode())
    )
    df = sensors.DeltaOhmPMV(source=src).dataframe

    for column in ['datetime', 'variable', 'value', 'unit']:
        assert column in df.columns


def test_read_delta_ohm_pmv_format_error():
    with pytest.raises(sensors.DataFormatError):
        print(sensors.DeltaOhmPMV(io.StringIO(TR7)).dataframe)


def test_dataframe_pmv():
    testo_pmv = sensors.TestoPMV(io.StringIO(PMV_TESTO))
    measured = testo_pmv.dataframe.pivot('variable', index='datetime', values='value')

    df_pmv = sensors.DataFramePMV(
        tdb='온도',
        tr='흑구온도',
        vel='기류',
        rh='상대습도',
        met=0.9,
        clo=1.0,
        units='SI',
    )
    calculated = df_pmv(measured.with_columns(pl.col('상대습도') * 100))

    polars.testing.assert_series_equal(measured['PMV'], calculated['PMV'], rtol=0.1)
