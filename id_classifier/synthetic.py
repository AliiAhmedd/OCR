"""SyntheticSource: fake ID-like cards drawn with Pillow, for demos and tests.

Nothing here is a real document or a real person:
- the only text is the country name, the document title, field labels and "SPECIMEN"
- every personal field is a grey bar, and the "photo" is a grey silhouette
- the MRZ lines contain only filler ("<", "0", "SPECIMEN") and the country's 3-letter code

The true label is stored inside each PNG (text field LABEL_KEY). MockClassifier reads it, which lets us
test the whole pipeline without a model. These images prove the pipeline works end-to-end;
they do NOT say how well a real model will do. That comparison needs the labeled sample of real images.
"""

from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont 
from PIL.PngImagePlugin import PngInfo # needed for PNG text fields, which hold the true label in each synthetic image

from id_classifier.sources import ImageKey, ImageSource, failed_record, record_from_bytes
from id_classifier.types import GROUND_TRUTH_COLUMNS, NO_DOCUMENT, UNKNOWN, ImageRecord

LABEL_KEY = "id_classifier_synthetic_label"  # PNG text field holding the true label as JSON
FIRST_TRANSACTION_ID = 900_000_001           # synthetic transaction ids start here, easy to recognise
CANVAS_SIZE = (960, 640)                     # every synthetic "photo" is 960 x 640 pixels
NEGATIVE_KINDS = ("blank", "noise", "shapes")  # images with no document in them. required for the pipeline to learn to reject non-document images.                                              


@dataclass(frozen=True)
class CountryStyle:
    code: str                       # ISO alpha-2 label, e.g. "TR"
    alpha3: str                     # 3-letter code printed in the MRZ lines, e.g. "TUR"
    header: str                     # big header text on the card
    id_title: str                   # document title under the header
    colour: tuple[int, int, int]    # header colour, different per country


COUNTRY_STYLES = {
    style.code: style
    for style in (
        CountryStyle("TR", "TUR", "REPUBLIC OF TURKIYE", "IDENTITY CARD", (200, 30, 45)),
        CountryStyle("TN", "TUN", "REPUBLIC OF TUNISIA", "NATIONAL IDENTITY CARD", (150, 20, 70)),
        CountryStyle("SA", "SAU", "KINGDOM OF SAUDI ARABIA", "NATIONAL ID CARD", (0, 110, 60)),
        CountryStyle("JO", "JOR", "HASHEMITE KINGDOM OF JORDAN", "PERSONAL IDENTITY CARD", (40, 40, 40)),
        CountryStyle("SD", "SDN", "REPUBLIC OF THE SUDAN", "NATIONAL ID CARD", (0, 90, 160)),
    )
}


@dataclass(frozen=True)
class SyntheticSample:
    """The plan for one synthetic image (it is only drawn when fetched or written)."""

    transaction_id: int
    image_id: str            # "front" / "back" for ID cards, "page" for passports, "front" for negatives
    document_type: str
    issuing_country: str
    document_side: str
    style: str               # country code for documents, or a NEGATIVE_KINDS value

    @property
    def reference(self) -> str:
        return f"synthetic://{self.transaction_id}/{self.image_id}"

    def label(self) -> dict:
        return {
            "sample_id": f"{self.transaction_id}:{self.image_id}",
            "document_type": self.document_type,
            "issuing_country": self.issuing_country,
            "document_side": self.document_side,
        }


def plan_samples(per_country: int, negatives: int, countries: list[str]) -> list[SyntheticSample]:
    """Decides which images exist. Same arguments -> same list, always."""
    samples = []
    next_id = FIRST_TRANSACTION_ID
    for code in countries: 
        if code not in COUNTRY_STYLES:
            raise ValueError(f"No synthetic style for country {code!r}. Available: {sorted(COUNTRY_STYLES)}") #country_styles is a dict of CountryStyle objects, keyed by country code. If the user asks for a country that isn't in that dict, err is raised
        for _ in range(per_country):                     # one transaction = front AND back of one ID card
            for side in ("front", "back"):
                samples.append(SyntheticSample(next_id, side, "national_id", code, side, code))
            next_id += 1
        for _ in range(max(1, per_country // 2)):        # fewer passports than ID cards
            samples.append(SyntheticSample(next_id, "page", "passport", code, "n/a", code))
            next_id += 1
    for i in range(negatives):                           # no document at all: the pipeline must reject these
        samples.append(SyntheticSample(next_id, "front", NO_DOCUMENT, UNKNOWN, "n/a", NEGATIVE_KINDS[i % len(NEGATIVE_KINDS)])) # i modulo len(NEGATIVE_KINDS) gives a mix of blank, noise and shapes so that 
        next_id += 1
    return samples


# ---------------------------------------------------------------- drawing helpers

def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)          # scalable built-in font (Pillow >= 10.1)
    except (TypeError, OSError, ImportError):
        return ImageFont.load_default()                   # fallback: small fixed-size bitmap font


def _text(draw: ImageDraw.ImageDraw, xy, text: str, size: int, fill, max_width: int | None = None) -> None:
    """Draws text, shrinking the font until it fits in max_width."""
    font = _font(size)
    while max_width and size > 8 and draw.textlength(text, font=font) > max_width:
        size -= 2
        font = _font(size)
    draw.text(xy, text, font=font, fill=fill)


def _paper(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(235, 250), rng.randint(230, 245), rng.randint(215, 235))  # off-white card colour


def _blank_card(width: int, height: int, fill) -> tuple[Image.Image, Image.Image, ImageDraw.ImageDraw]:
    """A card image plus a mask with rounded corners (the mask decides which pixels get pasted)."""
    card = Image.new("RGB", (width, height), fill)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, width - 1, height - 1], radius=round(height * 0.05), fill=255)
    return card, mask, ImageDraw.Draw(card)


def _photo_placeholder(draw: ImageDraw.ImageDraw, box: list[int]) -> None:
    """Grey box with a head-and-shoulders silhouette. No face, no person."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.rectangle(box, fill=(200, 200, 200), outline=(120, 120, 120), width=2)
    draw.ellipse([x0 + w * 0.3, y0 + h * 0.15, x1 - w * 0.3, y0 + h * 0.45], fill=(150, 150, 150))            # head
    draw.pieslice([x0 + w * 0.12, y0 + h * 0.52, x1 - w * 0.12, y0 + h * 1.44], 180, 360, fill=(150, 150, 150))  # shoulders


def _field(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, max_width: int, rng: random.Random) -> None:
    """A field label ("SURNAME") with a grey bar where the value would be."""
    _text(draw, (x, y), label, size=14, fill=(90, 90, 90))
    draw.rectangle([x, y + 20, x + round(max_width * rng.uniform(0.4, 1.0)), y + 36], fill=(150, 150, 150))


def _pad(text: str, length: int) -> str:
    return (text + "<" * length)[:length]  # MRZ lines are fixed-length, padded with "<"


def _card_size(rng: random.Random, ratio: float) -> tuple[int, int]:
    width = rng.randint(600, 700)
    return width, round(width / ratio)


def _card_front(style: CountryStyle, rng: random.Random):
    width, height = _card_size(rng, 1.586)                  # ID-1 card proportions (85.6 x 54 mm)
    card, mask, draw = _blank_card(width, height, _paper(rng))
    band = round(height * 0.2)
    draw.rectangle([0, 0, width, band], fill=style.colour)  # coloured header band
    _text(draw, (20, round(band * 0.2)), style.header, size=round(band * 0.42), fill="white", max_width=width - 40)
    _text(draw, (20, band + 8), style.id_title, size=round(height * 0.06), fill=style.colour, max_width=width - 40)
    photo = [20, band + round(height * 0.16), 20 + round(width * 0.26), height - 20]
    _photo_placeholder(draw, photo)
    x, y = photo[2] + 24, photo[1]
    for label in ("SURNAME", "GIVEN NAMES", "DATE OF BIRTH", "DOCUMENT NO."):
        _field(draw, x, y, label, width - x - 24, rng)
        y += round((photo[3] - photo[1]) / 4)
    _text(draw, (width - 180, height - 40), "SPECIMEN", size=26, fill=(210, 120, 120))
    return card, mask


def _card_back(style: CountryStyle, rng: random.Random):
    width, height = _card_size(rng, 1.586)
    card, mask, draw = _blank_card(width, height, _paper(rng))
    strip = round(height * 0.08)
    draw.rectangle([0, height - strip, width, height], fill=style.colour)  # thin coloured strip at the bottom
    x = 24
    while x < width * 0.6:                                   # fake barcode: random black stripes
        stripe = rng.randint(1, 5)
        draw.rectangle([x, 24, x + stripe, 24 + round(height * 0.2)], fill=(20, 20, 20))
        x += stripe + rng.randint(2, 5)
    y = 40 + round(height * 0.2)
    for label in ("ADDRESS", "PLACE OF BIRTH", "ISSUING AUTHORITY"):
        _field(draw, 24, y, label, width - 48, rng)
        y += 52
    # MRZ: three 30-character lines like a real ID card (TD1 format), filled with placeholders only
    lines = (_pad(f"I<{style.alpha3}", 30), _pad(f"0000000<0{style.alpha3}", 30), _pad("SPECIMEN<<SAMPLE", 30))
    top = height - strip - 3 * 26 - 10
    for i, line in enumerate(lines):
        _text(draw, (24, top + i * 26), line, size=22, fill=(30, 30, 30), max_width=width - 48)
    return card, mask


def _passport_page(style: CountryStyle, rng: random.Random):
    width, height = _card_size(rng, 1.42)                    # passport data page proportions (125 x 88 mm)
    paper = tuple(round(p * 0.9 + c * 0.1) for p, c in zip(_paper(rng), style.colour))  # paper lightly tinted
    card, mask, draw = _blank_card(width, height, paper)
    _text(draw, (20, 14), "PASSPORT", size=34, fill=style.colour)
    _text(draw, (20, 56), style.header, size=22, fill=(60, 60, 60), max_width=width - 40)
    mrz_top = height - 2 * 28 - 18
    photo = [20, 96, 20 + round(width * 0.27), mrz_top - 14]
    _photo_placeholder(draw, photo)
    x, y = photo[2] + 24, photo[1]
    for label in ("SURNAME", "GIVEN NAMES", "NATIONALITY", "DATE OF EXPIRY"):
        _field(draw, x, y, label, width - x - 24, rng)
        y += round((photo[3] - photo[1]) / 4)
    # MRZ: two 44-character lines like a real passport (TD3 format), placeholders only
    lines = (_pad(f"P<{style.alpha3}SPECIMEN<<SAMPLE", 44), _pad(f"000000000<0{style.alpha3}", 44))
    for i, line in enumerate(lines):
        _text(draw, (20, mrz_top + i * 28), line, size=22, fill=(30, 30, 30), max_width=width - 40)
    return card, mask


def _background(rng: random.Random) -> Image.Image:
    """A plain 'table surface' with a few faint lines."""
    base = rng.randint(120, 210)
    tint = (base, base - rng.randint(0, 20), base - rng.randint(0, 40))
    canvas = Image.new("RGB", CANVAS_SIZE, tint)
    draw = ImageDraw.Draw(canvas)
    darker = tuple(max(0, c - 15) for c in tint)
    for _ in range(rng.randint(3, 8)):
        y = rng.randint(0, CANVAS_SIZE[1])
        draw.line([(0, y), (CANVAS_SIZE[0], y + rng.randint(-60, 60))], fill=darker, width=rng.randint(1, 4))
    return canvas


def _draw_negative(canvas: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind == "blank":
        return canvas                                         # only the background
    if kind == "noise":
        # random colour blocks, blown up and smoothed; generated from rng so it is reproducible
        small = Image.frombytes("RGB", (96, 64), rng.randbytes(96 * 64 * 3))
        return small.resize(CANVAS_SIZE, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)                             # "shapes": random coloured shapes, no text
    for _ in range(rng.randint(10, 25)):
        x0, y0 = rng.randint(0, CANVAS_SIZE[0] - 40), rng.randint(0, CANVAS_SIZE[1] - 40)
        box = [x0, y0, x0 + rng.randint(20, 300), y0 + rng.randint(20, 200)]
        colour = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)(box, fill=colour)
    return canvas


def _place(canvas: Image.Image, card: Image.Image, mask: Image.Image, rng: random.Random) -> None:
    """Pastes the card on the background at a random position with a small tilt, like a phone photo."""
    angle = rng.uniform(-6, 6)
    card = card.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    mask = mask.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    x = rng.randint(0, max(0, canvas.width - card.width))
    y = rng.randint(0, max(0, canvas.height - card.height))
    canvas.paste(card, (x, y), mask)


def render(sample: SyntheticSample, seed: int) -> Image.Image:
    """Draws one sample. Same sample + same seed -> same image."""
    rng = random.Random(f"{seed}|{sample.transaction_id}|{sample.image_id}")  # string seeds are stable across runs
    canvas = _background(rng)
    if sample.document_type == NO_DOCUMENT:
        return _draw_negative(canvas, sample.style, rng)
    style = COUNTRY_STYLES[sample.style]
    if sample.document_type == "passport":
        card, mask = _passport_page(style, rng)
    elif sample.document_side == "front":
        card, mask = _card_front(style, rng)
    else:
        card, mask = _card_back(style, rng)
    _place(canvas, card, mask, rng)
    return canvas


def render_png_bytes(sample: SyntheticSample, seed: int) -> bytes:
    """The PNG file content for one sample, with the true label in a PNG text field."""
    info = PngInfo()
    info.add_text(LABEL_KEY, json.dumps(sample.label()))
    buffer = io.BytesIO()
    render(sample, seed).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


# ---------------------------------------------------------------- the source itself

class SyntheticSource(ImageSource):
    """Generates the images in memory on fetch(). Use write_dataset() to save them to disk with a ground-truth CSV."""

    def __init__(self, per_country: int = 4, negatives: int = 8, seed: int = 42, countries: list[str] | None = None):
        self.seed = seed
        self.samples = plan_samples(per_country, negatives, countries or list(COUNTRY_STYLES))
        self._by_key = {(s.transaction_id, s.image_id): s for s in self.samples}

    def list_keys(self) -> list[ImageKey]:
        return [(s.transaction_id, s.image_id, s.reference) for s in self.samples]

    def fetch(self, transaction_id: int, image_id: str, reference: str) -> ImageRecord:
        sample = self._by_key.get((transaction_id, image_id))
        if sample is None:
            return failed_record(transaction_id, image_id, reference, "unknown_synthetic_key")
        # go through PNG bytes, exactly like a real file, so hashing and decoding are exercised too
        return record_from_bytes(transaction_id, image_id, reference, render_png_bytes(sample, self.seed))


def write_dataset(source: SyntheticSource, out_dir: str | Path) -> Path:
    """Writes images/<transaction_id>_<image_id>.png and ground_truth.csv. Returns the CSV path."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ground_truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_COLUMNS)
        writer.writeheader()
        for sample in source.samples:
            file_name = f"{sample.transaction_id}_{sample.image_id}.png"
            (images_dir / file_name).write_bytes(render_png_bytes(sample, source.seed))
            writer.writerow({
                "transaction_id": sample.transaction_id,
                "image_id": sample.image_id,
                "image_path": f"images/{file_name}",   # relative to the CSV, so the folder can be moved
                "document_type": sample.document_type,
                "issuing_country": sample.issuing_country,
                "document_side": sample.document_side,
            })
    return csv_path
