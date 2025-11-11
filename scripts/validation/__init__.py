"""
2025-11 정량적 목표 (건물 에너지소비량 예측 오차율) 분석.

전처리
------
전처리 후 데이터에 다음 필드 포함.
- 필수
    - building
    - date
    - energy
    - temperature
- 기타
    - weather_station
    - is_holiday

data 폴더에 다음 파일 저장.
- 00.[name].parquet
- 01.glimpse-[name].txt
- 02 이후: 기타 그래프 등
"""
