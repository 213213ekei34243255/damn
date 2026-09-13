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
- nlpconnect/vit-gpt2-image-captioning: UNLIKE every model above, this
  one has documented serverless Inference API support (Hugging Face's
  own huggingface.js client library ships an imageToText() example
  using this exact model). ENABLED, with a runtime fallback: if
  HF_API_TOKEN isn't set, or the call fails/errors/cold-starts, this
  degrades to no caption rather than breaking the request - see
  generate_caption() below. Used ONLY as a text-query source when
  on-device OCR finds nothing (a content-only photo, e.g. an animal
  with no visible text) - it turns the photo into a caption like "a
  lion standing in grass" and that feeds into the SAME keyword search
  OCR text would otherwise drive. This is NOT reverse-image search -
  it can only help discover pages that happen to use similar words in
  their own text, same honest limitation as the OCR-driven path. True
  "find the exact page this photo was published on" requires a real
  visual-search index (Google Vision Web Detection / Bing Visual
  Search / TinEye), none of which exist for free - see the
  conversation this was scoped from for why that trade-off was made.

Because every image-embedding and OCR model actually named for Image
mode turned out to be unusable via free serverless inference, this
pipeline's visual-similarity signal is perceptual hashing (pure
Python/Pillow math, no model at all) plus on-device OCR (client-side)
plus this one captioning fallback - not the MobileCLIP/PP-OCRv6
pipeline originally specified. That's a deliberate, honest
simplification forced by real model availability - not a stopgap
hiding a broken integration.
"""

import io
import logging
import os
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
        "reason": (
            "Raw JAX research checkpoint (458MB) with no evidence of any "
            "HF Inference Provider deployment. Video mode (Phase 2) is "
            "implemented without it - see analyze_video_candidates(), "
            "which is text/keyword-only against OCR'd keyframes / an HF "
            "caption fallback, with no visual comparison step at all."
        ),
    },
    "audioIdentification": {
        "enabled": False,
        "provider": "audd",
        "reason": "Audio mode is Phase 3 - not implemented yet; no API key configured.",
    },
    "imageCaption": {
        "enabled": True,
        "provider": "huggingface-serverless",
        "model": "nlpconnect/vit-gpt2-image-captioning",
        "reason": (
            "Has documented serverless Inference API support (unlike "
            "MobileCLIP/BLIP/PP-OCRv6/VideoPrism). Requires HF_API_TOKEN "
            "to be set; degrades to no-caption at runtime if the token is "
            "missing, the call errors, or the model is cold-starting - "
            "never blocks the rest of the pipeline."
        ),
    },
}

MAX_CANDIDATES = 8
CANDIDATE_FETCH_TIMEOUT_S = 6
MAX_CANDIDATE_BYTES = 8 * 1024 * 1024  # refuse to download absurdly large "images"

# Free Hugging Face account access token (huggingface.co/settings/tokens)
# - NOT a paid Inference Endpoint. Server-side only; never sent to the
# client. Captioning is simply skipped if this isn't set.
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
HF_CAPTION_URL = f"https://api-inference.huggingface.co/models/{HF_CAPTION_MODEL}"
HF_CAPTION_TIMEOUT_S = 20


def generate_caption(image_bytes: bytes) -> Optional[str]:
    """
    Best-effort image-to-text caption via Hugging Face's free serverless
    Inference API - used only when on-device OCR found no usable text
    (see MediaSourceFinderService.swift). Returns None (never raises) on
    any failure: missing token, network error, bad response, or a cold-
    starting model (HF returns 503 with an estimated load time in that
    case - this does not wait/retry, since that would stall a live user
    request; the pipeline just proceeds without a caption that time).
    """
    if not HF_API_TOKEN:
        logger.info("[media-source] HF_API_TOKEN not set - captioning disabled")
        return None
    try:
        resp = requests.post(
            HF_CAPTION_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            data=image_bytes,
            timeout=HF_CAPTION_TIMEOUT_S,
        )
        if resp.status_code == 503:
            logger.info("[media-source] caption model is cold-starting, skipping this request")
            return None
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result and isinstance(result[0], dict):
            caption = result[0].get("generated_text")
            if caption:
                return caption.strip()
        logger.info("[media-source] caption response had no generated_text: %r", result)
        return None
    except Exception as e:
        logger.info("[media-source] caption generation failed: %s", e)
        return None

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

        logger.info(
            "[media-source] candidate url=%s hash_fetched=%s distance=%s text_overlap=%.2f accepted=%s",
            url, candidate_hash is not None, distance, text_overlap, (is_visual or is_text),
        )

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


def analyze_video_candidates(ocr_text: str, candidates: list) -> dict:
    """
    Video mode (Phase 2). Unlike analyze_image, there is NO visual
    comparison step here - fetching and decoding an arbitrary candidate
    page's video to compare against submitted keyframes isn't feasible
    within a request timeout, and VideoPrism has no confirmed free
    serverless inference either (see MODEL_CONFIG). Every result here is
    a text/keyword match against OCR'd keyframe text (or an HF-generated
    caption when no keyframe had visible text) - matchKind is always
    "text_match", and confidence never exceeds "possible_source" since
    there's no visual evidence to justify "likely_original_source".

    candidates: list of {"url", "title", "snippet"} from the client's
    GoogleSearchService.searchWeb() call - a plain web search for PAGES,
    not an image search, since video mode is looking for a page that
    might host/describe the video, not a matching thumbnail.
    """
    scored = []
    for candidate in candidates[:MAX_CANDIDATES]:
        url = candidate.get("url") or ""
        title = candidate.get("title") or ""
        snippet = candidate.get("snippet") or ""
        if not url:
            continue

        overlap = _text_overlap(ocr_text, f"{title} {snippet}")
        accepted = overlap >= TEXT_MATCH_MIN_OVERLAP

        logger.info(
            "[media-source] video candidate url=%s text_overlap=%.2f accepted=%s",
            url, overlap, accepted,
        )

        if not accepted:
            continue

        scored.append({
            "url": url,
            "domain": _domain(url),
            "title": title or None,
            "confidence": "possible_source",
            "confidencePercent": min(90, int(30 + overlap * 60)),
            "matchKind": "text_match",
            "evidence": ["Matching text/caption"],
            "thumbnailURL": None,
            "_overlap": overlap,
        })

    scored.sort(key=lambda m: -m["_overlap"])
    for m in scored:
        m.pop("_overlap", None)

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
