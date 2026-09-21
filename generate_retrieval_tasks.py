"""
Generates retrieval task datasets for Punjabi (Shahmukhi) and Urdu:
  - NameIndex   (100 items): 50-name list, ask for name at position N
  - MiddleMatch (100 items): 40-name list from 10-name pool, ask for name between A and B
  - ScriptMixed  (20 items): English instruction + native-script name list

Outputs saved to datasets/ as xlsx files.
Run: python generate_retrieval_tasks.py
"""

import sys
import io
import random
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATASETS_DIR = Path("D:/research/datasets")
RANDOM_SEED = 42

# ── Name lists ─────────────────────────────────────────────────────────────────

PUNJABI_50_NAMES = [
    "دلشاد بانو", "امجد فاروقی", "حمیرا اعوان", "طفیل گوندل", "عبدالحکیم",
    "ارشد چیمہ", "شاہدہ پروین", "فردوس بی بی", "اکرم گجر", "رخسانہ کوکب",
    "نسیم اختر", "شمیم اختر", "ممتاز بھٹی", "فرحت ناز", "شبیر حسین",
    "الطاف حسین", "انور شاہ", "نجمہ سعید", "غلام نبی", "اقبال ندیم",
    "شبنم گل", "اسلم پراچہ", "گلشن آرا", "نذیر لودھی", "مختار لغاری",
    "منظور الٰہی", "روبینہ بیگم", "اشرف چوہدری", "محمد یوسف", "لیاقت بلوچ",
    "نسرین مہر", "حافظ عبدالستار", "نصرت جہاں", "زیتون مائی", "رابعہ بی بی",
    "سجاد حیدر", "صبیحہ سلطانہ", "صفیہ بیگم", "مسعود کنجاہ", "سرفراز وڑائچ",
    "تسلیم فاطمہ", "ریاض کھوکھر", "ثمینہ یاسمین", "مقبول بٹ", "عاصمہ پروین",
    "حسینہ بیوی", "کلثوم بیگم", "بشیر وٹو", "خدیجہ زہرا", "پروین سلطانہ",
]

PUNJABI_10_POOL = [
    "احمد خان", "زینب اختر", "عمران شاہ", "پروین سلطانہ", "بلال حسین",
    "رابعہ بی بی", "فاطمہ بیگم", "انور شاہ", "صفیہ بیگم", "غلام نبی",
]

URDU_50_NAMES = [
    "عائشہ صدیقی", "محمد علی", "فاطمہ زہرا", "احمد رضا", "مریم انصاری",
    "عمر فاروق", "سارہ خان", "یوسف میر", "زینب حسین", "نعمان شیخ",
    "حنا قریشی", "ابراہیم سید", "صفیہ ملک", "طارق محمود", "نادیہ احمد",
    "کامران بیگ", "روبینہ طاہر", "شاہد اقبال", "نورین چوہدری", "عمران بٹ",
    "منیرہ خانم", "رفیق حیدر", "شہلا رانا", "نجم الدین", "پرویز الٰہی",
    "ثریا بیگم", "ارسلان علی", "سمیرا خالد", "ریحان انور", "بشریٰ کاظمی",
    "زاہد منیر", "آمنہ تاشفین", "قمر زمان", "صبا ناصر", "ظفر عباس",
    "ماریہ گل", "اسامہ طارق", "لبنیٰ شاہ", "وقار احمد", "حمزہ اعوان",
    "نیلوفر رضوی", "مصطفیٰ کریمی", "غزالہ بخاری", "شعیب اختر", "فریدہ پروین",
    "عدنان حق", "توقیر جاوید", "رمشا خانم", "ذوالفقار علی", "پاکیزہ بتول",
]

URDU_10_POOL = [
    "علی حسن", "مریم بیگم", "طارق عزیز", "نادیہ رضا", "کامران شاہ",
    "صبا خانم", "جاوید انور", "رشیدہ قریشی", "منصور اقبال", "شہناز بتول",
]

PASHTO_50_NAMES = [
    "زرغونه کاکړ", "ملالۍ میوند", "خوشحال خان", "احمد شاہ", "میرویس خان",
    "شاه زمان", "نازو توخۍ", "تورپیکۍ", "بټو خان", "ګل پاڼه",
    "اسفندیار ولي", "رحمان بابا", "اجمل خټک", "حمید مومند", "دریا خان",
    "سردار علي", "شیر محمد", "شریف خان", "اختر محمد", "جان محمد",
    "شاه محمود", "عبدالحی", "پاچا خان", "غني خان", "ولي خان",
    "سیف الرحمن", "نور محمد", "حیات خان", "عطا محمد", "سید جمال",
    "پیر محمد", "خان زمان", "فریدون خان", "عمر گل", "بخت جمال",
    "ظاهر شاه", "داود خان", "حبیب الله", "نجیب الله", "برهان الدین",
    "صالح محمد", "عزیز الله", "عصمت الله", "نصرت الله", "سمیع الله",
    "حکمت یار", "فضل حق", "نعمت الله", "هدایت الله", "ثناء الله",
]

PASHTO_10_POOL = [
    "احمد شاہ", "زرغونه کاکړ", "خوشحال خان", "ملالۍ میوند", "میرویس خان",
    "شاه زمان", "رحمان بابا", "غني خان", "تورپیکۍ", "حمید مومند",
]

BALOCHI_50_NAMES = [
    "میر چاکر", "میر گوہرام", "بالاچ گورگیج", "شے مرید", "ہانی بیگم",
    "بیبگر رند", "شہ داد", "میر حمل", "مہرلب", "ماہ گنج",
    "گل بی بی", "زرینہ بلوچ", "مراد بخش", "خداداد بلوچ", "عطا شاد",
    "ظفر علی", "غلام محمد", "حبیب جالب", "میر گل خان", "اکبر بارکزئی",
    "یوسف عزیز", "کریم بخش", "رحیم داد", "صالح محمد", "داد شاہ",
    "حمل ہوت", "سبزل بلوچ", "واحد بخش", "قادر بخش", "تاج محمد",
    "دوشنبہ", "شنبے خان", "سنجر خان", "سہراب خان", "نوروز خان",
    "امیر بخش", "عاصم بلوچ", "شیرین گل", "بانک کریمہ", "درناز",
    "شاہ خاتون", "بی بگر", "لعل بخش", "نود بندگ", "مراد خان",
    "خیر محمد", "اللہ بخش", "پیر داد", "روشن دین", "محمد بخش",
]

BALOCHI_10_POOL = [
    "میر چاکر", "ہانی بیگم", "بالاچ گورگیج", "شے مرید", "بیبگر رند",
    "میر حمل", "عطا شاد", "میر گل خان", "مہرلب", "گل بی بی",
]

# ── Query templates ─────────────────────────────────────────────────────────────

def nameindex_query(pos: int, lang: str) -> str:
    if lang == "pa":
        ordinal = "پہلا" if pos == 1 else f"{pos}واں"
        return f"فہرست وچ {ordinal} ناں کیہڑا اے؟"
    elif lang == "ur":
        ordinal = "پہلا" if pos == 1 else f"{pos}واں"
        return f"فہرست میں {ordinal} نام کیا ہے؟"
    elif lang == "ps":
        ordinal = "لومړی" if pos == 1 else f"{pos}م"
        return f"په لیست کې {ordinal} نوم څه دی؟"
    elif lang == "bal":
        ordinal = "اولی" if pos == 1 else f"{pos}می"
        return f"تہر بند ءَ {ordinal} نام چے اِنت؟"
    return f"What is the {pos} name?"


def middlematch_query(name_a: str, name_b: str, lang: str) -> str:
    if lang == "pa":
        return f"فہرست وچ {name_a} تے {name_b} دے وچکار کیہڑا ناں اے؟"
    elif lang == "ur":
        return f"فہرست میں {name_a} اور {name_b} کے درمیان کیا نام ہے؟"
    elif lang == "ps":
        return f"په لیست کې د {name_a} او {name_b} ترمنځ کوم نوم دی؟"
    elif lang == "bal":
        return f"تہر بند ءَ {name_a} ءُ {name_b} ءِ نیام ءَ چے نام اِنت؟"
    return f"What name is between {name_a} and {name_b}?"


def english_ordinal(n: int) -> str:
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    return f"{n}{suffix}"


# ── List formatting ─────────────────────────────────────────────────────────────

ARABIC_COMMA = "،"


def names_to_numbered_list(names: list[str]) -> str:
    """Return a numbered list as a single string: '1. name\n2. name\n...'"""
    return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))


def names_to_inline(names: list[str]) -> str:
    """Return names joined with Arabic comma + space (matches pilot format)."""
    return (ARABIC_COMMA + " ").join(names)


# ── xlsx helpers ────────────────────────────────────────────────────────────────

YELLOW = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
BOLD = Font(bold=True)
RTL = Alignment(horizontal="right", wrap_text=True, readingOrder=2)
LTR = Alignment(wrap_text=True)


def _header_row(ws, headers: list[str]):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = BOLD
        cell.alignment = LTR


def save_xlsx(items: list[dict], path: Path, sheet_title: str):
    """Save a list of dicts (keys: #, Long Data, Query, Correct Answer, Language, Task) to xlsx."""
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_title
    headers = ["#", "Long Data", "Query", "Correct Answer", "Language", "Task"]
    _header_row(ws, headers)

    for item in items:
        row_num = item["#"] + 1
        for c, key in enumerate(headers, 1):
            cell = ws.cell(row=row_num, column=c, value=item[key])
            if c == 2:
                cell.fill = YELLOW
                cell.alignment = RTL
            elif c == 3:
                cell.alignment = RTL
            else:
                cell.alignment = LTR

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 25
    ws.column_dimensions["E"].width = 10
    ws.column_dimensions["F"].width = 14
    wb.save(path)
    print(f"  Saved {len(items)} items -> {path}")


# ── NameIndex generator ─────────────────────────────────────────────────────────

def gen_nameindex(names_50: list[str], lang: str, n: int = 100) -> list[dict]:
    """
    Generate n NameIndex items. 10 positions × 10 shuffles = 100.
    Positions: [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    """
    rng = random.Random(RANDOM_SEED)
    positions = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    items = []
    shuffles_per_pos = n // len(positions)

    for pos in positions:
        for _ in range(shuffles_per_pos):
            shuffled = names_50[:]
            rng.shuffle(shuffled)
            answer = shuffled[pos - 1]
            long_data = names_to_numbered_list(shuffled)
            query = nameindex_query(pos, lang)
            items.append({
                "#": len(items) + 1,
                "Long Data": long_data,
                "Query": query,
                "Correct Answer": answer,
                "Language": lang,
                "Task": "NameIndex",
            })

    return items[:n]


# ── MiddleMatch generator ───────────────────────────────────────────────────────

def _make_40_list(pool: list[str], rng: random.Random) -> list[str]:
    """Sample 40 names from pool (with replacement), no two consecutive the same."""
    result = []
    prev = None
    pool_no_prev = pool[:]
    while len(result) < 40:
        if prev is not None:
            choices = [p for p in pool if p != prev]
        else:
            choices = pool[:]
        name = rng.choice(choices)
        result.append(name)
        prev = name
    return result


def gen_middlematch(pool_10: list[str], lang: str, n: int = 100) -> list[dict]:
    """
    Generate n MiddleMatch items. 5 random 40-name lists × 20 triplets each = 100.
    Each row asks for the name between two anchor names in the list.
    """
    rng = random.Random(RANDOM_SEED + 1)
    items = []
    lists_needed = (n + 19) // 20

    for list_idx in range(lists_needed):
        name_list = _make_40_list(pool_10, rng)
        long_data = names_to_numbered_list(name_list)

        # Collect all valid consecutive triplets (avoid ambiguous A,B pairs)
        seen_pairs: dict[tuple, str] = {}
        triplets = []
        for i in range(1, len(name_list) - 1):
            a, mid, b = name_list[i - 1], name_list[i], name_list[i + 1]
            pair = (a, b)
            if pair not in seen_pairs:
                seen_pairs[pair] = mid
                triplets.append((a, mid, b))
            elif seen_pairs[pair] == mid:
                triplets.append((a, mid, b))
            # skip ambiguous pair (same A,B but different middles in this list)

        # Sample up to 20 triplets, spread across the list
        step = max(1, len(triplets) // 20)
        selected = triplets[::step][:20]

        for a, mid, b in selected:
            if len(items) >= n:
                break
            query = middlematch_query(a, b, lang)
            items.append({
                "#": len(items) + 1,
                "Long Data": long_data,
                "Query": query,
                "Correct Answer": mid,
                "Language": lang,
                "Task": "MiddleMatch",
            })
        if len(items) >= n:
            break

    return items[:n]


# ── ScriptMixed generator ───────────────────────────────────────────────────────

def gen_scriptmixed(names_50: list[str], lang: str, n: int = 20) -> list[dict]:
    """
    Generate n ScriptMixed items.
    English instruction + native-script numbered name list; ask for position N.
    Positions: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] × 2 shuffles = 20.
    """
    rng = random.Random(RANDOM_SEED + 2)
    positions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    items = []
    shuffles_per_pos = n // len(positions)

    for pos in positions:
        for _ in range(shuffles_per_pos):
            shuffled = names_50[:]
            rng.shuffle(shuffled)
            answer = shuffled[pos - 1]
            name_list_str = names_to_numbered_list(shuffled)
            ord_str = english_ordinal(pos)
            prompt = (
                f"Here's a list of names:\n{name_list_str}\n\n"
                f"What is the {ord_str} name in the list?\n"
                f"Reply with one name only."
            )
            items.append({
                "#": len(items) + 1,
                "Long Data": prompt,
                "Query": f"What is the {ord_str} name in the list?",
                "Correct Answer": answer,
                "Language": lang,
                "Task": "ScriptMixed",
            })

    return items[:n]


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    DATASETS_DIR.mkdir(exist_ok=True)

    configs = [
        {
            "lang": "pa",
            "label": "Punjabi",
            "folder": "punjabi",
            "names_50": PUNJABI_50_NAMES,
            "pool_10": PUNJABI_10_POOL,
        },
        {
            "lang": "ur",
            "label": "Urdu",
            "folder": "urdu",
            "names_50": URDU_50_NAMES,
            "pool_10": URDU_10_POOL,
        },
        {
            "lang": "ps",
            "label": "Pashto",
            "folder": "pashto",
            "names_50": PASHTO_50_NAMES,
            "pool_10": PASHTO_10_POOL,
        },
        {
            "lang": "bal",
            "label": "Balochi",
            "folder": "balochi",
            "names_50": BALOCHI_50_NAMES,
            "pool_10": BALOCHI_10_POOL,
        },
    ]

    for cfg in configs:
        lang = cfg["lang"]
        label = cfg["label"]
        out_dir = DATASETS_DIR / cfg["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{label}] Generating retrieval tasks in {out_dir}...")

        ni = gen_nameindex(cfg["names_50"], lang, n=100)
        save_xlsx(ni, out_dir / f"NameIndex_{label}.xlsx", f"NameIndex {label}")

        mm = gen_middlematch(cfg["pool_10"], lang, n=100)
        save_xlsx(mm, out_dir / f"MiddleMatch_{label}.xlsx", f"MiddleMatch {label}")

        sm = gen_scriptmixed(cfg["names_50"], lang, n=20)
        save_xlsx(sm, out_dir / f"ScriptMixed_{label}.xlsx", f"ScriptMixed {label}")

    print("\nDone. All retrieval task files written across all 4 languages.")


if __name__ == "__main__":
    main()
