from __future__ import annotations

import re
from typing import ClassVar


class MetropolitanGov:
    pattern = re.compile(
        r'서울(특별)?시|(부산|대구|인천|광주|대전|울산)광역시'
        r'|세종(특별자치)?시|(강원|전(라)?북|제주)(특별자치)?도'
        r'|경기도|충청[남북]도|전라남도|경상[남북]도'
    )

    asos_region_mapping: ClassVar[dict[str, str]] = {
        '강원': '강원',
        '경상남도': '경남',
        '경남': '경남',
        '부산': '경남',
        '울산': '경남',
        '경상북도': '경북',
        '경북': '경북',
        '대구': '경북',
        '경기': '서울경기',
        '서울': '서울경기',
        '인천': '서울경기',
        '광주': '전남',
        '전라남도': '전남',
        '전남': '전남',
        '전라북도': '전북',
        '전북': '전북',
        '제주': '제주',
        '대전': '충남',
        '세종': '충남',
        '충청남도': '충남',
        '충남': '충남',
        '충청북도': '충북',
        '충북': '충북',
    }

    @classmethod
    def search(cls, address: str):
        if m := cls.pattern.search(address):
            return m.group()
        return None

    @classmethod
    def asos_region(cls, region: str):
        for key, value in cls.asos_region_mapping.items():
            if key in region:
                return value

        raise ValueError(region)
