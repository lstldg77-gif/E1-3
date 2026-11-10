#!/usr/bin/env python3
"""Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 입력 패턴이 어떤 필터(Cross / X)와
더 유사한지 판정하는 콘솔 시뮬레이터.

- 모드 1: 3×3 필터 2개(A, B)와 패턴을 콘솔로 입력받아 판정
- 모드 2: data.json 의 필터/패턴을 읽어 일괄 판정 + PASS/FAIL 리포트
- 모드 3: (보너스) N×N Cross/X 패턴 자동 생성 + 성능 분석

표준 라이브러리만 사용한다 (NumPy 등 외부 라이브러리 사용 금지).

사용법:
    python main.py                 # 기본 data.json (main.py 와 같은 폴더)
    python main.py other.json      # 모드 2 에서 다른 JSON 파일 사용
"""

import json
import math
import os
import re
import sys
import time
import unicodedata

# ---------------------------------------------------------------------------
# 상수 / 정책
# ---------------------------------------------------------------------------
EPSILON = 1e-9          # 점수 비교 허용오차: |a - b| < EPSILON 이면 동점
REPEAT = 10             # 성능 측정 반복 횟수 (최소 10회)
PERF_SIZES = (3, 5, 13, 25)

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 외부 데이터에서 들어오는 다양한 라벨 표기 → 내부 표준 라벨
#   expected 값: '+' → Cross, 'x' → X
#   filter 키 : 'cross' → Cross, 'x' → X
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

SECTION_LINE = "#" + "-" * 39

PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")
FILTER_KEY_RE = re.compile(r"^size_(\d+)$")


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
class Matrix:
    """n×n 정사각 행렬 (패턴/필터 공용 저장 구조)."""

    def __init__(self, size, fill=0.0):
        if not isinstance(size, int) or size < 1:
            raise ValueError("행렬 크기는 1 이상의 정수여야 합니다.")
        self.size = size
        self.data = [[float(fill)] * size for _ in range(size)]

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트를 검증한 뒤 Matrix 로 변환한다. 문제가 있으면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 배열(리스트의 리스트)이 아닙니다.")
        n = len(rows)
        mat = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"{r + 1}번째 행이 배열이 아닙니다.")
            if len(row) != n:
                raise ValueError(
                    f"정사각 행렬이 아닙니다: 행 {n}개, {r + 1}번째 행의 열 {len(row)}개"
                )
            for c, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"({r + 1},{c + 1}) 위치의 값 {value!r} 은(는) 숫자가 아닙니다.")
                if not math.isfinite(value):
                    raise ValueError(f"({r + 1},{c + 1}) 위치의 값이 유한한 숫자가 아닙니다.")
                mat.set(r, c, value)
        return mat

    def get(self, row, col):
        return self.data[row][col]

    def set(self, row, col, value):
        self.data[row][col] = float(value)

    def to_flat(self):
        """보너스: 2차원 → 1차원(길이 N²) 변환 (row-major)."""
        flat = []
        for row in self.data:
            flat.extend(row)
        return flat

    def __str__(self):
        return "\n".join(" ".join(_fmt_cell(v) for v in row) for row in self.data)


def _fmt_cell(value):
    return str(int(value)) if value == int(value) else repr(value)


# ---------------------------------------------------------------------------
# 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_label(raw):
    """'+', 'cross', 'x', 'Cross', 'X' 등 → 'Cross' / 'X'. 알 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# MAC 연산 (반복문 직접 구현)
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """2차원 배열 버전: 같은 위치끼리 곱한 뒤 모두 더한다. O(N²)."""
    n = len(pattern)
    if n != len(filt):
        raise ValueError("패턴과 필터의 크기가 다릅니다.")
    total = 0.0
    for i in range(n):
        p_row = pattern[i]
        f_row = filt[i]
        for j in range(n):
            total += p_row[j] * f_row[j]
    return total


def mac_flat(pattern_flat, filter_flat):
    """보너스: 1차원(길이 N²) 배열 버전. 인덱스 계산이 한 단계로 단순해진다."""
    length = len(pattern_flat)
    if length != len(filter_flat):
        raise ValueError("패턴과 필터의 길이가 다릅니다.")
    total = 0.0
    for k in range(length):
        total += pattern_flat[k] * filter_flat[k]
    return total


# ---------------------------------------------------------------------------
# 점수 비교 정책 (epsilon)
# ---------------------------------------------------------------------------
def compare_scores(score_a, score_b, eps=EPSILON):
    """a > b → 1, a < b → -1, |a-b| < eps → 0 (동점)."""
    if abs(score_a - score_b) < eps:
        return 0
    return 1 if score_a > score_b else -1


def judge(score_cross, score_x):
    result = compare_scores(score_cross, score_x)
    if result == 0:
        return LABEL_UNDECIDED
    return LABEL_CROSS if result > 0 else LABEL_X


# ---------------------------------------------------------------------------
# 성능 측정
# ---------------------------------------------------------------------------
def measure_avg_ms(func, *args, repeat=REPEAT):
    """func(*args) 를 repeat 회 호출한 평균 시간(ms). I/O 는 측정 구간에서 제외."""
    func(*args)  # 워밍업 (첫 호출 오버헤드 제외)
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


def print_perf_table(cases):
    """cases: [(n, pattern Matrix, filter Matrix), ...]"""
    rows = []
    for n, pattern, filt in cases:
        avg_2d = measure_avg_ms(mac, pattern.data, filt.data)
        p_flat, f_flat = pattern.to_flat(), filt.to_flat()   # 변환 비용은 측정 제외
        avg_1d = measure_avg_ms(mac_flat, p_flat, f_flat)
        rows.append((n, avg_2d, avg_1d))

    print(_pad("크기", 10) + _pad("평균 시간(ms)", 16, True)
          + _pad("연산 횟수(N²)", 16, True) + _pad("1회당(ns)", 12, True))
    print("-" * 54)
    for n, avg_2d, _ in rows:
        ops = n * n
        per_op_ns = avg_2d * 1e6 / ops
        print(_pad(f"{n}×{n}", 10) + f"{avg_2d:>16.4f}{ops:>16}{per_op_ns:>12.1f}")

    print()
    print(f"[보너스] 2D vs 1D(평탄화) 비교 (동일 입력, 평균/{REPEAT}회)")
    print(_pad("크기", 10) + _pad("2D(ms)", 12, True) + _pad("1D(ms)", 12, True)
          + _pad("변화율", 10, True))
    print("-" * 44)
    for n, avg_2d, avg_1d in rows:
        change = (avg_1d - avg_2d) / avg_2d * 100 if avg_2d > 0 else 0.0
        print(_pad(f"{n}×{n}", 10) + f"{avg_2d:>12.4f}{avg_1d:>12.4f}{change:>+9.1f}%")


def _pad(text, width, right=False):
    """한글(전각) 문자를 2칸으로 계산해 정렬한다."""
    shown = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)
    space = " " * max(0, width - shown)
    return space + text if right else text + space


# ---------------------------------------------------------------------------
# 보너스: 패턴 생성기
# ---------------------------------------------------------------------------
def generate_cross(n):
    """가운데 행/열이 1 인 N×N 십자가. (짝수 N 이면 n//2 번째 행/열 사용)"""
    mat = Matrix(n)
    mid = n // 2
    for i in range(n):
        for j in range(n):
            if i == mid or j == mid:
                mat.set(i, j, 1)
    return mat


def generate_x(n):
    """두 대각선이 1 인 N×N X 패턴."""
    mat = Matrix(n)
    for i in range(n):
        for j in range(n):
            if i == j or i + j == n - 1:
                mat.set(i, j, 1)
    return mat


# ---------------------------------------------------------------------------
# 콘솔 입출력 유틸
# ---------------------------------------------------------------------------
def section(title):
    print()
    print(SECTION_LINE)
    print(f"# {title}")
    print(SECTION_LINE)


def read_line(prompt=""):
    try:
        # 파이프 입력(PowerShell 등) 시 붙을 수 있는 BOM 제거
        return input(prompt).lstrip("﻿")
    except EOFError:
        print("\n입력이 종료되어 프로그램을 끝냅니다.")
        sys.exit(0)


def parse_row(line, n):
    tokens = line.split()
    if len(tokens) != n:
        raise ValueError(f"숫자 {len(tokens)}개가 입력됨")
    values = []
    for tok in tokens:
        try:
            value = float(tok)
        except ValueError:
            raise ValueError(f"'{tok}' 은(는) 숫자가 아님") from None
        if not math.isfinite(value):
            raise ValueError(f"'{tok}' 은(는) 유한한 숫자가 아님")
        values.append(value)
    return values


def input_matrix(title, n=3):
    """n줄을 한 줄씩 입력받는다. 형식 오류인 줄은 안내 후 그 줄만 재입력."""
    print(f"{title} ({n}줄 입력, 공백 구분)")
    rows = []
    while len(rows) < n:
        line = read_line()
        try:
            rows.append(parse_row(line, n))
        except ValueError as err:
            print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. ({err})")
            print(f"  → {len(rows) + 1}번째 줄을 다시 입력하세요.")
    return Matrix.from_rows(rows)


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ---------------------------------------------------------------------------
def run_user_mode():
    n = 3
    section("[1] 필터 입력")
    filter_a = input_matrix("필터 A", n)
    print()
    filter_b = input_matrix("필터 B", n)

    print()
    print("[저장 확인]")
    print("필터 A:")
    print(filter_a)
    print("필터 B:")
    print(filter_b)

    section("[2] 패턴 입력")
    pattern = input_matrix("패턴", n)

    score_a = mac(pattern.data, filter_a.data)
    score_b = mac(pattern.data, filter_b.data)
    time_a = measure_avg_ms(mac, pattern.data, filter_a.data)
    time_b = measure_avg_ms(mac, pattern.data, filter_b.data)

    result = compare_scores(score_a, score_b)
    section("[3] MAC 결과" + (" (판정 불가)" if result == 0 else ""))
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(f"연산 시간(MAC 1회 평균/{REPEAT}회): {(time_a + time_b) / 2:.4f} ms")
    if result == 0:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON:g})")
    else:
        print(f"판정: {'A' if result > 0 else 'B'}")

    section(f"[4] 성능 분석 (평균/{REPEAT}회)")
    print_perf_table([(n, pattern, filter_a)])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError as err:
        print(f"[오류] JSON 파싱 실패 ({path}): {err}")
    except OSError as err:
        print(f"[오류] 파일을 읽을 수 없습니다 ({path}): {err}")
    return None


def load_filters(data):
    """filters 섹션 로드. 반환: ({N: {'Cross': Matrix, 'X': Matrix}}, {N: 실패 사유})"""
    filters, errors = {}, {}
    raw = data.get("filters")
    if not isinstance(raw, dict):
        print("✗ 'filters' 항목이 없거나 객체(dict)가 아닙니다.")
        return filters, errors

    def sort_key(key):
        m = FILTER_KEY_RE.match(key)
        return (0, int(m.group(1))) if m else (1, 0)

    for key in sorted(raw, key=sort_key):
        m = FILTER_KEY_RE.match(key)
        if not m:
            print(f"✗ {key}: 키 형식 오류 (size_N 형태가 아님) → 무시")
            continue
        n = int(m.group(1))
        entry = raw[key]
        if not isinstance(entry, dict):
            errors[n] = f"{key} 필터가 객체(dict)가 아님"
            print(f"✗ {key:<8} 필터 로드 실패: {errors[n]}")
            continue

        loaded, problems = {}, []
        for fkey, fval in entry.items():
            label = normalize_label(fkey)
            if label is None:
                problems.append(f"알 수 없는 필터 키 '{fkey}'")
                continue
            try:
                mat = Matrix.from_rows(fval)
            except ValueError as err:
                problems.append(f"{fkey} 필터 형식 오류 - {err}")
                continue
            if mat.size != n:
                problems.append(f"{fkey} 필터 크기 {mat.size}×{mat.size} ≠ 키의 {n}×{n}")
                continue
            loaded[label] = mat

        missing = [lbl for lbl in (LABEL_CROSS, LABEL_X) if lbl not in loaded]
        if missing:
            reason = "; ".join(problems) if problems else f"필터 누락: {', '.join(missing)}"
            errors[n] = reason
            print(f"✗ {key:<8} 필터 로드 실패: {reason}")
        else:
            filters[n] = loaded
            note = f" (경고: {'; '.join(problems)})" if problems else ""
            print(f"✓ {key:<8} 필터 로드 완료 ({LABEL_CROSS}, {LABEL_X}){note}")
    return filters, errors


def evaluate_case(key, entry, filters, filter_errors):
    """패턴 1건을 판정한다. 어떤 오류도 예외로 새지 않고 FAIL 결과로 돌려준다."""
    def fail(reason):
        print(f"판정: - | FAIL ({reason})")
        return {"id": key, "passed": False, "reason": reason}

    m = PATTERN_KEY_RE.match(key)
    if not m:
        return fail("키 형식 오류: size_{N}_{idx} 형태가 아님")
    n = int(m.group(1))

    if not isinstance(entry, dict):
        return fail("스키마 오류: 패턴 항목이 객체(dict)가 아님")
    if "input" not in entry:
        return fail("스키마 오류: 'input' 누락")
    if "expected" not in entry:
        return fail("스키마 오류: 'expected' 누락")

    expected = normalize_label(entry["expected"])
    if expected is None:
        return fail(f"라벨 오류: 알 수 없는 expected 값 {entry['expected']!r}")

    if n not in filters:
        reason = filter_errors.get(n, f"size_{n} 필터가 없음")
        return fail(f"필터 없음: {reason}")

    try:
        pattern = Matrix.from_rows(entry["input"])
    except ValueError as err:
        return fail(f"입력 형식 오류: {err}")

    if pattern.size != n:
        return fail(f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, 필터 {n}×{n}")

    score_cross = mac(pattern.data, filters[n][LABEL_CROSS].data)
    score_x = mac(pattern.data, filters[n][LABEL_X].data)
    verdict = judge(score_cross, score_x)

    print(f"Cross 점수: {score_cross!r}")
    print(f"X 점수: {score_x!r}")
    if verdict == LABEL_UNDECIDED:
        print(f"판정: {verdict} | expected: {expected} | FAIL (동점 규칙)")
        return {"id": key, "passed": False,
                "reason": f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| < {EPSILON:g})"}
    if verdict == expected:
        print(f"판정: {verdict} | expected: {expected} | PASS")
        return {"id": key, "passed": True, "reason": ""}
    print(f"판정: {verdict} | expected: {expected} | FAIL")
    return {"id": key, "passed": False, "reason": f"판정 불일치 (판정 {verdict}, expected {expected})"}


def analyze_patterns(data, filters, filter_errors):
    results = []
    raw = data.get("patterns")
    if not isinstance(raw, dict):
        print("✗ 'patterns' 항목이 없거나 객체(dict)가 아닙니다.")
        return results

    def sort_key(key):
        m = PATTERN_KEY_RE.match(key)
        return (0, int(m.group(1)), int(m.group(2)), key) if m else (1, 0, 0, key)

    for key in sorted(raw, key=sort_key):
        print(f"--- {key} ---")
        try:
            result = evaluate_case(key, raw[key], filters, filter_errors)
        except Exception as err:  # 최후 방어선: 한 케이스 때문에 전체가 죽지 않도록
            reason = f"예기치 못한 오류: {type(err).__name__}: {err}"
            print(f"판정: - | FAIL ({reason})")
            result = {"id": key, "passed": False, "reason": reason}
        results.append(result)
    return results


def print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")
    if failed:
        print()
        print("실패 케이스:")
        for r in results:
            if not r["passed"]:
                print(f"- {r['id']}: {r['reason']}")
    print()
    print('(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션 참고)')


def run_json_mode(path):
    data = load_json(path)
    if data is None:
        return
    if not isinstance(data, dict):
        print("[오류] JSON 최상위 구조가 객체(dict)가 아닙니다.")
        return

    section("[1] 필터 로드")
    filters, filter_errors = load_filters(data)

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data, filters, filter_errors)

    section(f"[3] 성능 분석 (평균/{REPEAT}회)")
    cases = []
    for n in PERF_SIZES:
        # data.json 에 해당 크기 필터가 있으면 사용, 없으면(3×3 등) 생성기로 만든다.
        filt = filters[n][LABEL_CROSS] if n in filters else generate_cross(n)
        cases.append((n, generate_cross(n), filt))
    print_perf_table(cases)

    section("[4] 결과 요약")
    print_summary(results)


# ---------------------------------------------------------------------------
# 모드 3 (보너스): 패턴 생성기
# ---------------------------------------------------------------------------
def run_generator_mode():
    while True:
        raw = read_line("생성할 크기 N (2 이상의 정수): ").strip()
        if raw.isdigit() and int(raw) >= 2:
            n = int(raw)
            break
        print("입력 형식 오류: 2 이상의 정수를 입력하세요.")

    cross, x = generate_cross(n), generate_x(n)
    section(f"[1] {n}×{n} 패턴 생성")
    if n <= 25:
        print("Cross:")
        print(cross)
        print("X:")
        print(x)
    else:
        print("(N > 25 이므로 패턴 출력 생략)")

    section("[2] 생성 패턴 교차 판정")
    for name, pattern, expected in (("Cross 패턴", cross, LABEL_CROSS), ("X 패턴", x, LABEL_X)):
        sc = mac(pattern.data, cross.data)
        sx = mac(pattern.data, x.data)
        verdict = judge(sc, sx)
        status = "PASS" if verdict == expected else "FAIL"
        print(f"{name}: Cross 점수 {sc!r} / X 점수 {sx!r} → {verdict} ({status})")

    section(f"[3] 성능 분석 (평균/{REPEAT}회)")
    sizes = sorted(set(PERF_SIZES) | {n})
    print_perf_table([(s, generate_cross(s), generate_cross(s)) for s in sizes])


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def configure_stdout():
    # Windows 콘솔(cp949) 등에서 특수문자 출력 시 프로그램이 죽지 않도록 한다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv):
    configure_stdout()
    json_path = argv[1] if len(argv) > 1 else DEFAULT_JSON

    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 (보너스)")
    while True:
        choice = read_line("선택: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("1, 2, 3 중 하나를 입력하세요.")

    if choice == "1":
        run_user_mode()
    elif choice == "2":
        run_json_mode(json_path)
    else:
        run_generator_mode()


if __name__ == "__main__":
    main(sys.argv)
