#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advisor.py — CLI «Starting Hands Advisor» для NLHE (6-max/9-max)

Один файл, без зависимостей. Реализует:
- advise: рекомендация по руке/позиции/сценарию
- range:   показать диапазон для позиции/сценария
- export:  экспорт диапазона в матрицу 13×13 (ASCII/Unicode)
- explain: объяснить источник решения
- validate: самопроверка пресетов/правил

ВНИМАНИЕ: это обучающий упрощённый чартик, не GTO.
Лицензия: MIT.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

# -----------------------------
# Константы и базовые структуры
# -----------------------------

RANKS: List[str] = "A K Q J T 9 8 7 6 5 4 3 2".split()
RANK_TO_I: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}  # ниже индекс — слабее ранг
I_TO_RANK: Dict[int, str] = {i: r for r, i in RANK_TO_I.items()}

POSITIONS_6 = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
POSITIONS_9 = ["UTG", "UTG1", "MP", "MP1", "HJ", "CO", "BTN", "SB", "BB"]

Action = Literal["FOLD", "CALL", "RAISE", "3BET", "4BET"]
Confidence = Literal["LOW", "MED", "HIGH"]

# Тип «рука» в нормальной записи: "AKs", "QJo", "99" (пары — без суффикса)
HandStr = str

# -----------------------------
# Утилиты по картам/рукам
# -----------------------------


def _is_rank(ch: str) -> bool:
    return ch in RANK_TO_I


def parse_cards_to_hand_str(cards: str) -> HandStr:
    """
    Парсим строку вида "AsKh" / "9d9c" / "7h5s" и нормализуем в "AKs"/"99"/"AJo".
    Масти учитываются только для suited/offsuit определения.
    """
    s = cards.strip()
    if len(s) not in (4, 5):  # допускаем "AsKh" (4) или "A s K h" (с пробелом)
        raise ValueError("Ожидались две карты, формат вроде 'AsKh' или '9d9c'")
    # оставляем только символы рангов и мастей
    s = "".join(ch for ch in s if ch.isalnum())
    if len(s) != 4:
        raise ValueError("Некорректный формат карт (должно быть ровно 2 карты)")

    r1, m1, r2, m2 = s[0].upper(), s[1].lower(), s[2].upper(), s[3].lower()
    if not (_is_rank(r1) and _is_rank(r2)):
        raise ValueError("Неизвестный ранг карты. Допустимы A,K,Q,J,T,9..2")
    if m1 not in "shdc" or m2 not in "shdc":
        raise ValueError("Неизвестная масть. Допустимы s,h,d,c")

    # нормализация по старшинству рангов
    i1, i2 = RANK_TO_I[r1], RANK_TO_I[r2]
    if i1 == i2:  # пара
        return f"{r1}{r2}"
    # определяем suited/offsuit
    suited = (m1 == m2)
    hi, lo = (r1, r2) if i1 < i2 else (r2, r1)  # меньший индекс — старше
    return f"{hi}{lo}{'s' if suited else 'o'}"


def normalize_hand_notation(s: str) -> HandStr:
    """
    Нормализуем ручной ввод типа "kqs", "Qa Ko" в компактную запись.
    Ввод может быть без мастей для пар: "99".
    """
    s = s.replace(" ", "").upper()
    # Если пара
    if len(s) == 2 and s[0] == s[1] and _is_rank(s[0]):
        return s
    if len(s) == 3:
        r1, r2, k = s[0], s[1], s[2].lower()
        if not (_is_rank(r1) and _is_rank(r2) and k in ("S", "O", "s", "o")):
            raise ValueError("Некорректный формат руки. Пример: 'AKs', 'QJo', '99'")
        i1, i2 = RANK_TO_I[r1], RANK_TO_I[r2]
        hi, lo = (r1, r2) if i1 < i2 else (r2, r1)
        return f"{hi}{lo}{k.lower()}"
    raise ValueError("Некорректный формат руки. Пример: 'AKs', 'QJo', '99'")


def all_pairs() -> List[HandStr]:
    return [f"{r}{r}" for r in RANKS]


def all_suited() -> List[HandStr]:
    out = []
    for i, hi in enumerate(RANKS[:-1]):
        for lo in RANKS[i + 1 :]:
            out.append(f"{hi}{lo}s")
    return out


def all_offsuit() -> List[HandStr]:
    out = []
    for i, hi in enumerate(RANKS[:-1]):
        for lo in RANKS[i + 1 :]:
            out.append(f"{hi}{lo}o")
    return out


def rank_range(start: str, end: str) -> List[str]:
    """Список рангов от start до end по силе (включительно), например T..6."""
    i0, i1 = RANK_TO_I[start], RANK_TO_I[end]
    if i0 > i1:
        # например T до 6 — i0<i1 (т.к. A=0)
        # если пришло наоборот — меняем
        i0, i1 = i1, i0
    return [I_TO_RANK[i] for i in range(i0, i1 + 1)]


def make_hand(hi: str, lo: str, kind: str) -> HandStr:
    i1, i2 = RANK_TO_I[hi], RANK_TO_I[lo]
    assert i1 < i2, "для несетовых рук hi должен быть старше lo"
    return f"{hi}{lo}{kind}"


# ---------------------------------
# Разворачивание шаблонов из чартов
# ---------------------------------


def expand_token(token: str) -> Set[HandStr]:
    """
    Разворачивает записи вида:
      - пары: "55+"  -> 55,66,77,88,99,TT,JJ,QQ,KK,AA
      - одномастные/разномастные: "ATs+", "KJo+"
      - одиночные: "KQo", "JTs"
      - диапазоны коннекторов: "T9s-76s" (включительно)
    Возвращает множество нормализованных рук.
    """
    token = token.strip()
    if not token:
        return set()

    # Пары
    if len(token) in (2, 3) and token[0] == token[1] and _is_rank(token[0]):
        if token.endswith("+"):
            start = token[:2]
            start_i = RANK_TO_I[start[0]]
            return {f"{I_TO_RANK[i]}{I_TO_RANK[i]}" for i in range(start_i, RANK_TO_I["A"] + 1)}
        return {token[:2]}

    # Диапазон вида XYs-UVs (только для suited/offsuit)
    if "-" in token:
        left, right = token.split("-", 1)
        left = left.strip()
        right = right.strip()
        if len(left) == 3 and len(right) == 3 and left[2] == right[2] and left[2] in "soSO":
            k = left[2].lower()
            # Пример: T9s-76s => hi: T..7, lo: 9..6 (соответственно)
            hi_from, lo_from = left[0], left[1]
            hi_to, lo_to = right[0], right[1]
            hi_list = rank_range(hi_to, hi_from)  # например 7..T
            lo_list = rank_range(lo_to, lo_from)  # например 6..9
            if len(hi_list) != len(lo_list):
                # Защита от ошибок ввода
                raise ValueError(f"Невалидный диапазон: {token}")
            out: Set[HandStr] = set()
            for hi, lo in zip(reversed(hi_list), reversed(lo_list)):
                # hi всегда старше lo
                if RANK_TO_I[hi] < RANK_TO_I[lo]:
                    out.add(make_hand(hi, lo, k))
                else:
                    # если некорректная пара рангов — пропускаем
                    pass
            return out

    # Суффикс «+» для не пар (например ATo+, ATs+)
    if token.endswith("+") and len(token) == 4:
        base = token[:3]
        kind = base[-1].lower()
        if kind not in ("s", "o"):
            raise ValueError(f"Ожидался 's' или 'o' в {token}")
        hi, lo = base[0].upper(), base[1].upper()
        out: Set[HandStr] = set()
        # увеличиваем младшую карту вниз по силе (например ATo+ => ATo, AJo, AQo, AKo недопустимо — hi фикс)
        # На практике «+» для не пар означает: фиксированный hi, lo — от указанного вверх до один ниже hi
        for lo_rank in RANKS[RANK_TO_I[lo] : RANK_TO_I[hi]]:
            out.add(make_hand(hi, lo_rank, kind))
        return out

    # Одиночная непарная рука типа "KQo" / "JTs"
    if len(token) == 3 and _is_rank(token[0]) and _is_rank(token[1]) and token[2].lower() in ("s", "o"):
        hi, lo = token[0].upper(), token[1].upper()
        k = token[2].lower()
        if RANK_TO_I[hi] == RANK_TO_I[lo]:
            raise ValueError(f"Непарные руки должны иметь разные ранги: {token}")
        # нормализация порядка
        if RANK_TO_I[hi] > RANK_TO_I[lo]:
            hi, lo = lo, hi
        return {make_hand(hi, lo, k)}

    raise ValueError(f"Не удалось разобрать токен чарта: '{token}'")


def expand_many(tokens: Iterable[str]) -> Set[HandStr]:
    out: Set[HandStr] = set()
    for t in tokens:
        out |= expand_token(t)
    return out


# ---------------------------------
# Пресеты (минимально достаточные)
# ---------------------------------

# Примерный базовый 6-max пресет «Basic6»
BASIC6 = {
    "table": "6max",
    "open": {
        "UTG": ["77+", "AJo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs", "T9s", "98s"],
        "MP": ["66+", "AJo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs", "T9s-98s", "87s"],
        "CO": ["55+", "ATo+", "ATs+", "KJo+", "KTs+", "QTs+", "JTs", "T9s-87s", "76s", "65s"],
        "BTN": ["22+", "A2s+", "ATo+", "K9s+", "KJo+", "Q9s+", "QJo", "J9s+", "T9s-54s"],
        "SB": ["22+", "A2s+", "A5o+", "K9s+", "KTo+", "Q9s+", "QTo+", "J9s+", "T9s-54s"],
        # BB обычно не first-in в рейз без лимперов — условно используем как изолейт ≈ BTN
        "BB": ["22+", "A2s+", "ATo+", "K9s+", "KJo+", "Q9s+", "QJo", "J9s+", "T9s-65s"],
    },
    # Защита против открытия
    "defend": {
        # BB против BTN open 2.5bb
        "BB_vs_BTN_open_2.5": {
            "call": [
                "A2o+", "A2s+",
                "K9s+", "KTo+", "KJo+",  # offsuit KTo/KJo как край
                "Q9s+", "QTo+", "QJo",
                "J9s+", "JTo",
                "T9s-54s",
                "22+",
            ],
            "3bet": ["AJo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs", "99+"],
        },
        # SB против CO open 2.5bb — больше 3бет/фолд
        "SB_vs_CO_open_2.5": {
            "call": ["AJs", "KQs", "QJs", "JTs", "TT+"],  # узкий колл
            "3bet": ["AQo+", "ATs+", "KQo", "KTs+", "QTs+", "A5s-A2s", "99+"],
        },
    },
    # Против 3бета (упрощённо)
    "vs3bet": {
        # Открытие CO и нас 3бетят с BTN (условный кейс)
        "CO_vs_BTN_3bet": {
            "continue": ["AQo+", "AQs+", "TT+", "KQs"],
            "fourbet": ["AKo", "AKs", "QQ+", "A5s-A2s"],  # вэлью + блефовые Axs
        }
    },
    # Против 4бета (минимально)
    "vs4bet": {
        "BTN_vs_SB_4bet": {
            "continue": ["AKo", "AKs", "QQ+"],
            "fivebet": ["KK+", "AKs"],  # чисто декоративно в рамках ТЗ
        }
    },
}

# Компактный 9-max — используем консервативные открытия и унаследуем остальное по умолчанию
BASIC9 = {
    "table": "9max",
    "open": {
        "UTG": ["88+", "AQo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs"],
        "UTG1": ["77+", "AQo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs", "T9s"],
        "MP": ["66+", "AJo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs", "T9s-98s"],
        "MP1": ["66+", "AJo+", "ATs+", "KQo", "KTs+", "QTs+", "JTs", "T9s-98s", "87s"],
        "HJ": ["55+", "ATo+", "ATs+", "KJo+", "KTs+", "QTs+", "JTs", "T9s-87s", "76s"],
        "CO": ["55+", "ATo+", "ATs+", "KJo+", "KTs+", "QTs+", "JTs", "T9s-87s", "76s", "65s"],
        "BTN": ["22+", "A2s+", "ATo+", "K9s+", "KJo+", "Q9s+", "QJo", "J9s+", "T9s-54s"],
        "SB": ["22+", "A2s+", "A5o+", "K9s+", "KTo+", "Q9s+", "QTo+", "J9s+", "T9s-54s"],
        "BB": ["22+", "A2s+", "ATo+", "K9s+", "KJo+", "Q9s+", "QJo", "J9s+", "T9s-65s"],
    },
    "defend": BASIC6["defend"],
    "vs3bet": BASIC6["vs3bet"],
    "vs4bet": BASIC6["vs4bet"],
}

PRESETS = {
    "Basic6": BASIC6,
    "Basic9": BASIC9,
}


# ---------------------------------
# Построение индексов решений
# ---------------------------------

@dataclass(frozen=True)
class Context:
    table: Literal["6max", "9max"]
    pos: str
    action: Literal["first_in", "vs"]  # first_in (open) или vs (против действия)
    vs: Optional[Literal["open", "3bet", "4bet"]] = None
    opp_from: Optional[str] = None
    sizing: float = 2.5
    ante: float = 0.0
    preset_name: str = "Basic6"
    strict: bool = False
    verbose: bool = False


@dataclass
class Decision:
    action: Action
    sizing_bb: Optional[float]
    alt: Optional[Action]
    confidence: Confidence
    source: str
    notes: List[str]

    def to_machine_line(self) -> str:
        size = f" ({self.sizing_bb}bb)" if self.sizing_bb else ""
        alt = f" | ALT: {self.alt}" if self.alt else ""
        return f"RECOMMENDATION: {self.action}{size}{alt} | CONF: {self.confidence}"

    def to_json(self) -> str:
        return json.dumps(
            {
                "action": self.action,
                "sizing_bb": self.sizing_bb,
                "confidence": self.confidence,
                "source": self.source,
                "notes": self.notes,
                "alt": self.alt,
            },
            ensure_ascii=False,
        )


class Chart:
    """
    Основной класс: загрузка пресета, индексация, рекомендации.
    """

    def __init__(self, preset_name: str):
        if preset_name not in PRESETS:
            raise ValueError(f"Неизвестный пресет: {preset_name}")
        self.name = preset_name
        self.cfg = PRESETS[preset_name]
        self.table = self.cfg["table"]
        self.positions = POSITIONS_6 if self.table == "6max" else POSITIONS_9

        # Индексы для быстрого поиска
        # open_index[(pos, hand)] = Action
        self.open_index: Dict[Tuple[str, HandStr], Action] = {}
        # defend_index[("BB_vs_BTN_open_2.5", hand)] = Action (CALL/3BET/FOLD)
        self.defend_index: Dict[Tuple[str, HandStr], Action] = {}
        # vs3bet_index[(key, hand)] -> {"continue"/"fourbet": Action}
        self.vs3bet_index: Dict[Tuple[str, HandStr], str] = {}
        # vs4bet_index аналогично
        self.vs4bet_index: Dict[Tuple[str, HandStr], str] = {}

        self._build_indices()

    # ---------- Индексация ----------

    def _build_indices(self) -> None:
        # Open
        for pos, tokens in self.cfg.get("open", {}).items():
            hands = expand_many(tokens)
            for h in hands:
                self.open_index[(pos, h)] = "RAISE"

        # Defend vs Open
        for key, bucket in self.cfg.get("defend", {}).items():
            for act_key in ("call", "3bet"):
                tokens = bucket.get(act_key, [])
                hands = expand_many(tokens)
                for h in hands:
                    action: Action = "CALL" if act_key == "call" else "3BET"
                    self.defend_index[(key, h)] = action

        # vs 3bet
        for key, bucket in self.cfg.get("vs3bet", {}).items():
            for act_key in ("continue", "fourbet"):
                tokens = bucket.get(act_key, [])
                hands = expand_many(tokens)
                for h in hands:
                    self.vs3bet_index[(key, h)] = act_key  # строковый ярлык

        # vs 4bet
        for key, bucket in self.cfg.get("vs4bet", {}).items():
            for act_key in ("continue", "fivebet"):
                tokens = bucket.get(act_key, [])
                hands = expand_many(tokens)
                for h in hands:
                    self.vs4bet_index[(key, h)] = act_key

    # ---------- Публичные методы ----------

    def get_range(self, scenario: str, pos: Optional[str] = None) -> Set[HandStr]:
        """
        Вернуть множество рук для:
         - scenario=="open" и задан pos — диапазон открытия
         - scenario ключом вида "BB_vs_BTN_open_2.5" — защита против открытия
        """
        if scenario == "open":
            if not pos:
                raise ValueError("Для 'open' необходимо указать позицию")
            tokens = self.cfg.get("open", {}).get(pos, [])
            return expand_many(tokens)
        # иначе считаем, что это ключ защитного диапазона
        bucket = self.cfg.get("defend", {}).get(scenario, {})
        out: Set[HandStr] = set()
        out |= expand_many(bucket.get("call", []))
        out |= expand_many(bucket.get("3bet", []))
        return out

    def advise(self, hand: HandStr, ctx: Context) -> Decision:
        """
        Главный метод выдачи рекомендации по контексту.
        """
        notes: List[str] = []
        conf: Confidence = "HIGH"
        sizing = ctx.sizing

        # first_in -> open
        if ctx.action == "first_in":
            source = f"{self.name}.open.{ctx.pos}"
            # Корректируем open под анте (простая эвристика): при ante>0 слегка расширяем,
            # но т.к. у нас статический список, мы просто снижаем требование к «близости» в эвристике
            if ctx.ante > 0:
                notes.append(f"ante={ctx.ante}BB: допускается расширение")
                # конфиденс чуть ниже, если рука вне таблицы, но проходит эвристику
            if (ctx.pos, hand) in self.open_index:
                return Decision("RAISE", sizing, "FOLD", "HIGH", source, notes)
            # эвристика (если не strict)
            if ctx.strict:
                return Decision("FOLD", None, None, "HIGH", source, notes + ["strict: off-chart -> FOLD"])
            action, conf2, reason = self._heuristic_open(hand, ctx.pos, ante=ctx.ante)
            notes.append(reason)
            return Decision(action, sizing if action == "RAISE" else None, "FOLD" if action != "FOLD" else None, conf2, source, notes)

        # vs smth
        if ctx.action == "vs":
            if ctx.vs == "open":
                key = f"{ctx.pos}_vs_{ctx.opp_from}_open_{sizing:.1f}"
                # Округлённые ключи по 0.5 bb (чтобы не взрываться от дробей)
                key_rounded = f"{ctx.pos}_vs_{ctx.opp_from}_open_{round(sizing*2)/2:.1f}"
                source = f"{self.name}.defend.{key_rounded}"
                # прямое попадание
                if (key_rounded, hand) in self.defend_index:
                    act = self.defend_index[(key_rounded, hand)]
                    alt = "3BET" if act == "CALL" else "CALL"
                    return Decision(act, None, alt, "HIGH", source, notes)

                # При отсутствии точного key — попробуем ближайшие типовые ключи (2.5/3.0)
                for sz in (2.0, 2.2, 2.5, 3.0):
                    k = f"{ctx.pos}_vs_{ctx.opp_from}_open_{sz:.1f}"
                    if (k, hand) in self.defend_index:
                        act = self.defend_index[(k, hand)]
                        # если sizing ниже 2.3 и мы BB vs BTN — колл чуточку шире (повышаем склонность к CALL)
                        if ctx.pos == "BB" and ctx.opp_from in ("BTN", "CO") and ctx.sizing <= 2.2 and act == "FOLD":
                            return Decision("CALL", None, "3BET", "LOW", f"{self.name}.defend.{k}", notes + ["sizing низкий: расширение BB defend"])
                        return Decision(act, None, ("3BET" if act == "CALL" else "CALL"), "MED", f"{self.name}.defend.{k}", notes + ["аппроксимация по sizing"])
                # эвристика по соседним рангам
                if ctx.strict:
                    return Decision("FOLD", None, None, "HIGH", source, notes + ["strict: off-chart -> FOLD"])
                act, conf2, reason = self._heuristic_defend(hand, ctx)
                return Decision(act, None, ("3BET" if act == "CALL" else "CALL") if act != "FOLD" else None, conf2, source, notes + [reason])

            if ctx.vs == "3bet":
                key = f"{ctx.pos}_vs_{ctx.opp_from}_3bet"
                source = f"{self.name}.vs3bet.{key}"
                if (key, hand) in self.vs3bet_index:
                    tag = self.vs3bet_index[(key, hand)]
                    if tag == "continue":
                        return Decision("CALL", None, "4BET", "HIGH", source, ["continue vs 3bet"])
                    else:
                        return Decision("4BET", None, "CALL", "HIGH", source, ["fourbet vs 3bet"])
                if ctx.strict:
                    return Decision("FOLD", None, None, "HIGH", source, ["strict: off-chart -> FOLD"])
                # простая эвристика: сильные бродвеи/пары продолжаем
                if hand in expand_many(["AQo+", "AQs+", "TT+", "KQs"]):
                    return Decision("CALL", None, "4BET", "MED", source, ["эвристика continue vs 3bet"])
                if hand in expand_many(["A5s-A2s", "AKs", "QQ+"]):
                    return Decision("4BET", None, "CALL", "LOW", source, ["эвристика fourbet vs 3bet"])
                return Decision("FOLD", None, None, "LOW", source, ["эвристика: слабая рука vs 3bet"])

            if ctx.vs == "4bet":
                key = f"{ctx.pos}_vs_{ctx.opp_from}_4bet"
                source = f"{self.name}.vs4bet.{key}"
                if (key, hand) in self.vs4bet_index:
                    tag = self.vs4bet_index[(key, hand)]
                    if tag == "continue":
                        return Decision("CALL", None, "5BET", "HIGH", source, ["continue vs 4bet"])
                    else:
                        return Decision("5BET", None, "CALL", "HIGH", source, ["fivebet vs 4bet"])
                if ctx.strict:
                    return Decision("FOLD", None, None, "HIGH", source, ["strict: off-chart -> FOLD"])
                if hand in expand_many(["AKo", "AKs", "QQ+"]):
                    return Decision("CALL", None, "5BET", "MED", source, ["эвристика continue vs 4bet"])
                return Decision("FOLD", None, None, "LOW", source, ["эвристика: фолд vs 4bet"])

        # если дошли сюда — что-то не так
        return Decision("FOLD", None, None, "LOW", f"{self.name}.internal", ["неизвестный контекст"])

    # ---------- Объяснение ----------

    def explain(self, hand: HandStr, ctx: Context) -> str:
        d = self.advise(hand, ctx)
        lines = [
            f"- Preset: {d.source}",
            f"- Action: {d.action} (conf={d.confidence})",
        ]
        if d.alt:
            lines.append(f"- Alternative: {d.alt}")
        if d.sizing_bb:
            lines.append(f"- Sizing: {d.sizing_bb}bb")
        if d.notes:
            for n in d.notes:
                lines.append(f"- Note: {n}")
        return "\n".join(lines)

    # ---------- Экспорт матрицы ----------

    def export_matrix(self, scenario: str, pos: Optional[str], fmt: Literal["ascii", "unicode"]) -> str:
        rng = self.get_range(scenario, pos)
        # маппинг руки -> буква (R/C/3/F/?)
        # для open все из rng = RAISE, прочее = FOLD
        # для defend rng = CALL или 3BET в сумме — пометим как «C»/«3», иначе «F»
        # Для defend различить CALL и 3BET можно, если scenario в defend и есть индексы
        def cell_label(hi: str, lo: str, k: str) -> str:
            h = (f"{hi}{lo}" if k == "p" else f"{hi}{lo}{k}")
            if scenario == "open":
                return "R" if h in rng else "F"
            # defend
            label = "F"
            # ищем ключи defend, где этот h встречается
            bucket = self.cfg.get("defend", {}).get(scenario, {})
            if not bucket:
                return "?"
            if h in expand_many(bucket.get("3bet", [])):
                return "3"
            if h in expand_many(bucket.get("call", [])):
                return "C"
            return label

        # формируем матрицу 13×13
        header = "   " + " ".join(f"{r:>2}" for r in RANKS)
        lines = [header]
        for i, r_hi in enumerate(RANKS):
            row = [f"{r_hi:>2}"]
            for j, r_lo in enumerate(RANKS):
                if i == j:
                    # пары по диагонали
                    ch = cell_label(r_hi, r_lo, "p")
                elif i < j:
                    # верхний треугольник — suited (hi старше lo)
                    ch = cell_label(r_hi, r_lo, "s") if RANK_TO_I[r_hi] < RANK_TO_I[r_lo] else cell_label(r_lo, r_hi, "s")
                else:
                    # нижний — offsuit
                    ch = cell_label(r_lo, r_hi, "o") if RANK_TO_I[r_lo] < RANK_TO_I[r_hi] else cell_label(r_hi, r_lo, "o")
                row.append(f"{ch:>2}")
            lines.append(" ".join(row))

        box_top = "┌" + "─" * (len(lines[0]) - 2) + "┐"
        box_bot = "└" + "─" * (len(lines[0]) - 2) + "┘"
        body = "\n".join(lines)
        if fmt == "unicode":
            return f"{box_top}\n{body}\n{box_bot}"
        return body

    # ---------- Валидация ----------

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Базовые проверки:
         - корректность развёртки всех токенов
         - отсутствие конфликтов в open (одна и та же рука — всегда RAISE)
         - defend: не выдаём одну и ту же руку и в call, и в 3bet одновременно
        """
        errors: List[str] = []

        # Проверка open
        seen_open: Dict[Tuple[str, HandStr], str] = {}
        for pos, tokens in self.cfg.get("open", {}).items():
            try:
                hands = expand_many(tokens)
            except Exception as e:
                errors.append(f"open[{pos}] токен-ошибка: {e}")
                continue
            for h in hands:
                key = (pos, h)
                if key in seen_open:
                    errors.append(f"open[{pos}] дубль руки {h}")
                seen_open[key] = "RAISE"

        # Defend: пересечения call vs 3bet
        for key, bucket in self.cfg.get("defend", {}).items():
            try:
                call_h = expand_many(bucket.get("call", []))
                three_h = expand_many(bucket.get("3bet", []))
            except Exception as e:
                errors.append(f"defend[{key}] токен-ошибка: {e}")
                continue
            both = call_h & three_h
            if both:
                errors.append(f"defend[{key}] конфликт в руках: {sorted(both)}")

        # vs3bet / vs4bet — просто токены
        for sec in ("vs3bet", "vs4bet"):
            for key, bucket in self.cfg.get(sec, {}).items():
                for k in list(bucket.keys()):
                    try:
                        _ = expand_many(bucket.get(k, []))
                    except Exception as e:
                        errors.append(f"{sec}[{key}] токен-ошибка: {e}")

        return (len(errors) == 0, errors)

    # ---------- Эвристики ----------

    def _heuristic_open(self, hand: HandStr, pos: str, ante: float) -> Tuple[Action, Confidence, str]:
        """
        Простая эвристика для first_in:
         - если рука «соседняя» с чартом позиции — RAISE (MED/LOW), иначе FOLD.
         - с анте слегка либерализуем.
        """
        base_set = self.get_range("open", pos)
        if hand in base_set:
            return ("RAISE", "HIGH", "точное совпадение")
        # соседи: уменьшить младшую карту на шаг (для A9s vs ATo+ и т.п.)
        neigh: Set[HandStr] = set()
        try:
            if len(hand) == 2:  # пара
                r = hand[0]
                idx = RANK_TO_I[r]
                if idx + 1 < len(RANKS):
                    neigh.add(f"{I_TO_RANK[idx+1]}{I_TO_RANK[idx+1]}")  # ниже по силе
            else:
                hi, lo, k = hand[0], hand[1], hand[2]
                lo_i = RANK_TO_I[lo]
                # сосед вверх (усиление) и вниз (ослабление)
                if lo_i - 1 >= 0:
                    neigh.add(make_hand(hi, I_TO_RANK[lo_i - 1], k))
                if lo_i + 1 < RANK_TO_I[hi]:
                    neigh.add(make_hand(hi, I_TO_RANK[lo_i + 1], k))
        except Exception:
            pass
        overlap = neigh & base_set
        if overlap:
            conf: Confidence = "MED" if ante == 0 else "MED"
            reason = f"соседство с чартом ({sorted(overlap)[0]})"
            # при наличии анте можно позволить RAISE даже с небольшим отклонением
            if ante > 0:
                return ("RAISE", "MED", reason + " + анте")
            else:
                return ("RAISE", conf, reason)
        return ("FOLD", "LOW", "off-chart без соседства")

    def _heuristic_defend(self, hand: HandStr, ctx: Context) -> Tuple[Action, Confidence, str]:
        """
        Для BB vs BTN low sizing — тяготеем к CALL для A-x, KTo, QTo, JTo, низких s-коннекторов.
        В остальных случаях — FOLD как дефолт.
        """
        if ctx.pos == "BB" and ctx.opp_from in ("BTN", "CO") and ctx.sizing <= 2.2:
            if hand in expand_many(
                ["A2o+", "A2s+", "KTo", "QTo", "JTo", "T9s-54s", "22+"]
            ):
                return ("CALL", "LOW", "эвристика BB defend vs маленький open")
        return ("FOLD", "LOW", "эвристика: по умолчанию FOLD")


# -----------------------------
# CLI
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="advisor.py",
        description="Starting Hands Advisor (NLHE 6-max/9-max)",
    )
    p.add_argument("--preset", default="Basic6", choices=list(PRESETS.keys()), help="Выбор пресета")
    p.add_argument("--table", choices=["6max", "9max"], help="Переопределить тип стола (иначе как в пресете)")
    p.add_argument("--strict", action="store_true", help="Строго следовать чарту (без эвристик)")
    p.add_argument("--verbose", action="store_true", help="Подробное объяснение решения")
    p.add_argument("--json", action="store_true", help="Вывод в JSON формате")

    sub = p.add_subparsers(dest="cmd", required=True)

    # advise
    sp = sub.add_parser("advise", help="Рекомендация по руке/позиции/контексту")
    sp.add_argument("--cards", required=True, help="Две карты, например 'AsKh' или 'AKs' или '99'")
    sp.add_argument("--pos", required=True, help="Ваша позиция за столом")
    sp.add_argument("--action", choices=["first_in", "vs"], required=True, help="Сценарий: first_in или vs")
    sp.add_argument("--vs", choices=["open", "3bet", "4bet"], help="Против какого действия играем (для --action vs)")
    sp.add_argument("--from", dest="opp_from", help="Позиция оппонента, совершившего действие (для --vs)")
    sp.add_argument("--sizing", type=float, default=2.5, help="Размер открытия/3бета в ББ")
    sp.add_argument("--ante", type=float, default=0.0, help="Анте в ББ (0 по умолчанию)")

    # range
    sp2 = sub.add_parser("range", help="Вывести набор рук для позиции/сценария")
    sp2.add_argument("--pos", help="Позиция (для scenario=open)")
    sp2.add_argument("--scenario", required=True, help="open или ключ типа 'BB_vs_BTN_open_2.5'")

    # export
    sp3 = sub.add_parser("export", help="Экспорт матрицы 13×13 (ASCII/Unicode)")
    sp3.add_argument("--pos", help="Позиция (для scenario=open)")
    sp3.add_argument("--scenario", required=True, help="open или ключ типа 'BB_vs_BTN_open_2.5'")
    sp3.add_argument("--format", choices=["ascii", "unicode"], default="unicode", help="Формат вывода")

    # explain
    sp4 = sub.add_parser("explain", help="Объяснить решение")
    sp4.add_argument("--cards", required=True)
    sp4.add_argument("--pos", required=True)
    sp4.add_argument("--action", choices=["first_in", "vs"], required=True)
    sp4.add_argument("--vs", choices=["open", "3bet", "4bet"])
    sp4.add_argument("--from", dest="opp_from")
    sp4.add_argument("--sizing", type=float, default=2.5)
    sp4.add_argument("--ante", type=float, default=0.0)

    # validate
    sub.add_parser("validate", help="Проверка корректности пресета/логики")

    return p


def ensure_position(pos: str, table: Literal["6max", "9max"]) -> None:
    valid = POSITIONS_6 if table == "6max" else POSITIONS_9
    if pos not in valid:
        raise ValueError(f"Недопустимая позиция '{pos}' для {table}. Допустимы: {', '.join(valid)}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Загрузка пресета
    try:
        chart = Chart(args.preset)
    except Exception as e:
        print(f"Ошибка пресета: {e}", file=sys.stderr)
        return 4

    table: Literal["6max", "9max"] = chart.table
    if args.table:
        table = args.table  # пользователь может переопределить

    # Команда: validate
    if args.cmd == "validate":
        ok, errs = chart.validate()
        if ok:
            print("OK: пресет валиден.")
            return 0
        for e in errs:
            print("ERR:", e, file=sys.stderr)
        return 4

    # Команда: range
    if args.cmd == "range":
        scenario = args.scenario
        pos = args.pos
        if scenario == "open":
            if not pos:
                print("Для scenario=open требуется --pos", file=sys.stderr)
                return 2
            ensure_position(pos, table)
        try:
            rng = chart.get_range(scenario, pos)
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 4
        # Печатаем упорядоченно
        def key_hand(h: HandStr) -> Tuple[int, int, int]:
            if len(h) == 2:
                i = RANK_TO_I[h[0]]
                return (0, i, i)
            hi, lo, k = h[0], h[1], h[2]
            return (1 if k == "s" else 2, RANK_TO_I[hi], RANK_TO_I[lo])

        for h in sorted(rng, key=key_hand):
            print(h)
        return 0

    # Команда: export
    if args.cmd == "export":
        scenario = args.scenario
        pos = args.pos
        if scenario == "open":
            if not pos:
                print("Для scenario=open требуется --pos", file=sys.stderr)
                return 2
            ensure_position(pos, table)
        try:
            mat = chart.export_matrix(scenario, pos, args.format)
        except Exception as e:
            print(f"Ошибка экспорта: {e}", file=sys.stderr)
            return 4
        print(mat)
        return 0

    # Команда: advise / explain
    if args.cmd in ("advise", "explain"):
        # Валидация позиции
        ensure_position(args.pos, table)
        # Разбор руки
        try:
            # поддержим оба формата: "AsKh" или "AKs"/"99"
            cards = args.cards.strip()
            if len(cards) in (3, 2):  # "AKs" или "99"
                hand = normalize_hand_notation(cards)
            else:
                hand = parse_cards_to_hand_str(cards)
        except Exception as e:
            print(f"Некорректные карты: {e}", file=sys.stderr)
            return 2

        # Проверка совместимости флагов
        if args.action == "vs" and not args.vs:
            print("Для '--action vs' требуется указать '--vs {open|3bet|4bet}'", file=sys.stderr)
            return 2
        if args.action == "vs" and args.vs in ("open", "3bet", "4bet") and not args.opp_from:
            print("Для '--vs' требуется указать источник через '--from <pos>'", file=sys.stderr)
            return 2

        ctx = Context(
            table=table,
            pos=args.pos,
            action=args.action,
            vs=args.vs,
            opp_from=args.opp_from,
            sizing=float(args.sizing),
            ante=float(args.ante),
            preset_name=chart.name,
            strict=bool(args.strict),
            verbose=bool(args.verbose),
        )

        try:
            if args.cmd == "advise":
                dec = chart.advise(hand, ctx)
                if args.json:
                    print(dec.to_json())
                else:
                    print(dec.to_machine_line())
                    if args.verbose:
                        print(chart.explain(hand, ctx))
                return 0 if dec.action else 4
            else:
                # explain
                text = chart.explain(hand, ctx)
                print(text)
                return 0
        except ValueError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 4
        except Exception as e:
            print(f"Внутренняя ошибка: {e}", file=sys.stderr)
            return 4

    # Непредвиденное
    print("Неизвестная команда", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())