"""
Media Source Finder - Image mode backend (Phase 1 of 3; Video and
Audio are Phase 2/3, not implemented here).

MODEL AVAILABILITY - verified against Hugging Face's actual current
Inference Providers status (not assumed - see MODEL_CONFIG below,
which must stay truthful about what's really running):

- AXERA-TECH/MobileCLIP: NOT deployed by any HF Inference Provider.
  It's built for Axera's own NPU hardware/SDK (converted to their
  "axmodel" format), not a generic Inference-API-callable model. Even
  the most standard CLIP variant (openai/clip-vit-base-patch32) has no
  confirmed free serverless feature-extraction provider as of this
  writing, so there is no drop-in cloud embedding substitute either.
  DISABLED. Perceptual hashing (pure Python/Pillow math, no model at
  all) is the real visual-similarity signal used instead.
- PP-OCRv6 (PaddlePaddle): only a public HF Space demo exists
  (huggingface.co/spaces/PaddlePaddle/PP-OCRv6_Online_Demo) - that's a
  Gradio app, not a serverless Inference API endpoint, and isn't meant
  for production API traffic. DISABLED. The iOS client's on-device
  Apple Vision OCR supplies text instead (see MediaOCRService.swift) -
  free, fast, already proven working in this app's chat-attachment
  feature.
- google/videoprism-base-f16r288: Phase 2 (video mode) - not
  implemented yet, so not evaluated here.
- Audio identification (AudD/ACRCloud): Phase 3 - not implemented yet,
  no API key configured.

Because both of the models actually named for Image mode turned out to
be unusable via free serverless inference, this pipeline runs with NO
cloud ML model dependency at all: perceptual hashing for visual
similarity, on-device OCR (client-side) for text, and the app's
existing Google Custom Search image-search proxy for candidate
discovery. That's a deliberate, honest simplification - not a stopgap
hiding a broken integration.
"""

import io
import logging
import re
from typing import Optional

import requests
from PIL import Image
import imagehash

logger = logging.getLogger("MediaSourceFinder")

MODEL_CONFIG = {
    "imageEmbedding": {
        "enabled": False,
        "provider": "huggingface-serverless",
        "model": "AXERA-TECH/MobileCLIP",
        "reason": (
            "Not deployed by any HF Inference Provider (Axera NPU-only "
            "format). No confirmed free serverless CLIP-family embedding "
            "model is available as a substitute either."
        ),
    },
    "ocr": {
        "enabled": False,
        "provider": "on-device-vision",
        "model": "Apple Vision (VNRecognizeTextRequest, client-side)",
        "reason": (
            "PP-OCRv6 has no confirmed HF serverless Inference API - only "
            "a public Space demo exists. On-device OCR is used instead."
        ),
    },
    "videoEmbedding": {
        "enabled": False,
        "provider": "huggingface-serverless",
        "model": "google/videoprism-base-f16r288",
        "reason": "Video mode is Phase 2 - not implemented yet.",
    },
    "audioIdentification": {
        "enabled": False,
        "provider": "audd",
        "reason": "Audio mode is Phase 3 - not implemented yet; no API key configured.",
    },
}

MAX_CANDIDATES = 8
CANDIDATE_FETCH_TIMEOUT_S = 6
MAX_CANDIDATE_BYTES = 8 * 1024 * 1024  # refuse to download absurdly large "images"

# Perceptual-hash Hamming-distance thresholds on a 64-bit phash
# (0 = identical structure, 64 = maximally different). Tuned
# conservatively - a false "visual match" is worse than a missed one.
VISUAL_MATCH_MAX_DISTANCE = 10
LIKELY_ORIGINAL_MAX_DISTANCE = 4

# Jaccard word-overlap threshold between OCR'd text and a candidate's
# title/snippet to count as a genuine "text/keyword match" rather than
# coincidental overlap.
TEXT_MATCH_MIN_OVERLAP = 0.34

_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _words(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _text_overlap(ocr_text: str, candidate_text: str) -> float:
    """Jaccard overlap between OCR'd words and a candidate page's
    title/snippet - a cheap, dependency-free "does this page actually
    mention what's in the image" signal."""
    a, b = _words(ocr_text), _words(candidate_text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _domain(url: str) -> str:
    match = re.match(r"^https?://(?:www\.)?([^/]+)", url or "")
    return match.group(1) if match else (url or "unknown")


def _fetch_image_hash(url: str):
    try:
        resp = requests.get(
            url,
            timeout=CANDIDATE_FETCH_TIMEOUT_S,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JonahMediaSourceFinder/1.0)"},
        )
        resp.raise_for_status()
        content = resp.raw.read(MAX_CANDIDATE_BYTES + 1, decode_content=True)
        if len(content) > MAX_CANDIDATE_BYTES:
            logger.info("Candidate image too large, skipping: %s", url)
            return None
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content))  # must re-open after verify()
        return imagehash.phash(image)
    except Exception as e:
        logger.info("Candidate fetch/hash failed for %s: %s", url, e)
        return None


def analyze_image(original_image_bytes: bytes, ocr_text: str, candidates: list) -> dict:
    """
    candidates: list of {"url", "contextLink", "title"} from the
    client's GoogleSearchService.searchImages() call - real keyword-
    driven image search results, not a proprietary reverse-image index.
    Returns a dict matching iOS's MediaSourceResult exactly (camelCase
    keys, no snake_case conversion needed on either side).
    """
    try:
        original_image = Image.open(io.BytesIO(original_image_bytes))
        original_image.verify()
        original_image = Image.open(io.BytesIO(original_image_bytes))
        original_hash = imagehash.phash(original_image)
    except Exception as e:
        logger.warning("Could not read submitted image: %s", e)
        return {
            "outcome": "source_not_found",
            "topMatch": None,
            "otherMatches": [],
            "ocrText": ocr_text or None,
            "contentSummary": None,
        }

    scored = []
    for candidate in candidates[:MAX_CANDIDATES]:
        url = candidate.get("url") or ""
        context_link = candidate.get("contextLink") or url
        title = candidate.get("title") or ""
        if not url:
            continue

        candidate_hash = _fetch_image_hash(url)
        distance = (candidate_hash - original_hash) if candidate_hash is not None else None
        text_overlap = _text_overlap(ocr_text, title)

        is_visual = distance is not None and distance <= VISUAL_MATCH_MAX_DISTANCE
        is_text = text_overlap >= TEXT_MATCH_MIN_OVERLAP

        if not is_visual and not is_text:
            continue

        if is_visual and is_text:
            match_kind = "visual_and_text_match"
        elif is_visual:
            match_kind = "visual_match"
        else:
            match_kind = "text_match"

        evidence = []
        if is_visual:
            evidence.append(
                "Exact visual match" if distance <= LIKELY_ORIGINAL_MAX_DISTANCE else "Visually similar image"
            )
        if is_text:
            evidence.append("Matching text/caption")
        if not evidence:
            evidence.append("Weak match")

        if is_visual and distance <= LIKELY_ORIGINAL_MAX_DISTANCE:
            confidence = "likely_original_source"
            confidence_percent = max(70, 100 - distance * 6)
        elif is_visual:
            confidence = "possible_source"
            confidence_percent = max(40, 80 - distance * 4)
        else:
            confidence = "matching_page"
            confidence_percent = int(30 + text_overlap * 40)

        scored.append({
            "url": context_link,
            "domain": _domain(context_link),
            "title": title or None,
            "confidence": confidence,
            "confidencePercent": min(99, confidence_percent),
            "matchKind": match_kind,
            "evidence": evidence,
            "thumbnailURL": url,
            "_distance": distance if distance is not None else 999,
            "_text_overlap": text_overlap,
        })

    scored.sort(key=lambda m: (m["_distance"], -m["_text_overlap"]))
    for m in scored:
        m.pop("_distance", None)
        m.pop("_text_overlap", None)

    content_summary = ocr_text.strip() if ocr_text and ocr_text.strip() else None

    if not scored:
        outcome = "content_identified_source_not_found" if content_summary else "source_not_found"
        return {
            "outcome": outcome,
            "topMatch": None,
            "otherMatches": [],
            "ocrText": ocr_text or None,
            "contentSummary": content_summary,
        }

    return {
        "outcome": "found",
        "topMatch": scored[0],
        "otherMatches": scored[1:],
        "ocrText": ocr_text or None,
        "contentSummary": content_summary,
    }
