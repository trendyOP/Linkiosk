# ── text_normalizer.py ──────────────────────────────────────────
import unicodedata
import re
from typing import List

# 1) OCR에서 자주 헷갈리는 글자 매핑표
CHAR_MAP = str.maketrans({
    # 영문 대소문자/숫자 혼동
    '|': 'I', 'Ⅰ': 'I', 'l': 'I', 'ᄿ': 'I', 
    'Ø': 'O', 'Ο': 'O', 'ο': 'o',
    '€': 'E',
    # 기호 → 글자/공백
    '“': '"', '”': '"', '‘': "'", '’': "'",
    '×': 'x', '✕': 'x', '✖': 'x',
    '‐': '-', '-': '-', '‒': '-', '–': '-', '—': '-',
    # 백틱 제거 (OCR 오인식)
    '`': '', '´': '', '′': '', '‵': '',
})
_KEEP_RE = re.compile(r'[^0-9a-z가-힣]', re.I)

def normalize_token(token: str) -> str:
    if not token:
        return ''
    txt = token.translate(CHAR_MAP)

    # **NFKC** 또는 **NFC** 로 ‘재조합’ ▶ 한글이 한 글자 단위로 유지
    txt = unicodedata.normalize('NFKC', txt)

    # (라틴 문자 악센트 제거용) 결합 문자만 삭제
    txt = ''.join(c for c in txt if not unicodedata.combining(c))

    txt = _KEEP_RE.sub('', txt).lower()
    return txt
