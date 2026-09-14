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

import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
from typing import Optional

import requests
from PIL import Image
import imagehash

# Reused for a cross-worker audio-result cache (see _get_cached_audio_
# result/_cache_audio_result below) - this backend already runs Redis
# for Veronica.py's chat history, so no new infrastructure is needed.
# If Redis itself is ever unreachable, every cache read/write is
# wrapped in a try/except that just skips caching - it degrades to
# "call the provider every time" rather than breaking audio ID.
from Veronica import redis_client

try:
    import acoustid
    _ACOUSTID_LIB_AVAILABLE = True
except ImportError:
    _ACOUSTID_LIB_AVAILABLE = False

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
        "enabled": False,  # recomputed at runtime from ACOUSTID_API_KEY/AUDD_API_TOKEN - see below
        "provider": "acoustid+audd",
        "model": "Chromaprint/AcoustID (free, tried first) then AudD standard endpoint (paid, fallback)",
        "reason": (
            "Real acoustic-fingerprint music recognition - not a generic "
            "HF audio model (no free HF model can identify arbitrary "
            "songs; see the note on microsoft/wavlm-base-plus - it's a "
            "speech/audio representation model, not a fingerprinting "
            "database, so it's not used here at all). AcoustID is free "
            "but its web-service tier is documented for non-commercial/"
            "open-source use only - a real ToS consideration for a paid "
            "app, not something this code resolves for you. Requires "
            "ACOUSTID_API_KEY and/or AUDD_API_TOKEN; returns "
            "outcome=not_configured (distinct from a genuine not_found) "
            "only when NEITHER is set - the UI must never blur those "
            "two into the same message."
        ),
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
    "visualSearch": {
        "enabled": False,  # recomputed at runtime from GOOGLE_VISION_API_KEY - see below
        "provider": "google-cloud-vision",
        "model": "Web Detection (images:annotate)",
        "reason": (
            "The one genuine reverse-image-search capability available to "
            "this feature - actually finds pages containing this exact or "
            "visually similar image via Google's own web index, unlike "
            "the free OCR+perceptual-hash+keyword-search pipeline below. "
            "A DIFFERENT Google product from the Custom Search JSON API "
            "this app already uses for text/image-by-keyword search - "
            "needs its own GOOGLE_VISION_API_KEY (Cloud Vision API "
            "enabled in Google Cloud Console; has its own free tier). "
            "Optional: the free pipeline remains the fallback when this "
            "isn't configured, or when Vision runs but finds nothing."
        ),
    },
}
MODEL_CONFIG["visualSearch"]["enabled"] = bool(os.environ.get("GOOGLE_VISION_API_KEY"))
MODEL_CONFIG["audioIdentification"]["enabled"] = bool(
    os.environ.get("ACOUSTID_API_KEY") or os.environ.get("AUDD_API_TOKEN")
)

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


# --------------------------------------------------------------------
# Google Cloud Vision Web Detection - genuine reverse-image search.
#
# This is a DIFFERENT Google product from the Custom Search JSON API
# (GoogleSearchService.swift's searchWeb/searchImages) this app already
# uses elsewhere - Custom Search only ever searches BY TEXT KEYWORDS,
# never by image content. Web Detection is the actual "search by
# providing an image" capability - what "reverse image search" and
# "Google Lens"-style tools are really built on. Needs its own
# GOOGLE_VISION_API_KEY (enable "Cloud Vision API" in Google Cloud
# Console; it has its own free tier separate from Custom Search's).
# --------------------------------------------------------------------

GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
GOOGLE_VISION_TIMEOUT_S = 20
# Vision Web Detection is billed per call - capping how many keyframes
# of a video get checked keeps a single video analysis from silently
# burning through a lot of quota.
MAX_VIDEO_KEYFRAMES_FOR_VISION = 3


def _web_detect(image_bytes: bytes) -> Optional[dict]:
    """
    Calls Web Detection on one image. Returns the raw `webDetection`
    object (fullMatchingImages / partialMatchingImages /
    pagesWithMatchingImages / visuallySimilarImages / bestGuessLabels),
    or None on any failure (missing key, network error, bad response,
    or the API itself returning an error for this image) - callers fall
    back to the free pipeline in every one of those cases, never raise.
    """
    if not GOOGLE_VISION_API_KEY:
        return None
    try:
        payload = {
            "requests": [{
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
            }]
        }
        resp = requests.post(
            GOOGLE_VISION_URL,
            params={"key": GOOGLE_VISION_API_KEY},
            json=payload,
            timeout=GOOGLE_VISION_TIMEOUT_S,
        )
        resp.raise_for_status()
        responses = (resp.json() or {}).get("responses") or []
        if not responses:
            return None
        if "error" in responses[0]:
            logger.warning("[media-source] Vision API returned an error: %s", responses[0]["error"])
            return None
        return responses[0].get("webDetection")
    except Exception as e:
        logger.info("[media-source] Vision Web Detection call failed: %s", e)
        return None


def _score_web_detection(web_detection: dict) -> list:
    """
    Turns one image's webDetection payload into the same scored-match
    shape the free pipeline produces, so the rest of the response (JSON
    shape, sorting) is identical regardless of which provider actually
    found the matches. A page with a fullMatchingImages entry is real,
    strong evidence (the exact image was found there) - never
    downgraded to a guess.
    """
    if not web_detection:
        return []

    scored = []
    for page in web_detection.get("pagesWithMatchingImages", [])[:MAX_CANDIDATES]:
        url = page.get("url") or ""
        if not url:
            continue

        full = [m.get("url") for m in page.get("fullMatchingImages", []) if m.get("url")]
        partial = [m.get("url") for m in page.get("partialMatchingImages", []) if m.get("url")]

        if full:
            confidence, confidence_percent = "likely_original_source", 95
            evidence = ["Exact visual match (Google Web Detection)"]
            thumbnail = full[0]
        elif partial:
            confidence, confidence_percent = "possible_source", 70
            evidence = ["Partial/cropped visual match (Google Web Detection)"]
            thumbnail = partial[0]
        else:
            continue

        scored.append({
            "url": url,
            "domain": _domain(url),
            "title": page.get("pageTitle") or None,
            "confidence": confidence,
            "confidencePercent": confidence_percent,
            "matchKind": "visual_match",
            "evidence": evidence,
            "thumbnailURL": thumbnail,
            "_rank": 0 if full else 1,
        })

    scored.sort(key=lambda m: m["_rank"])
    for m in scored:
        m.pop("_rank", None)
    return scored


def _best_guess_summary(web_detection: Optional[dict]) -> Optional[str]:
    """Google's own best-guess description of the image - genuinely
    better than the HF caption fallback when available, since it comes
    from the same system that just found (or didn't find) real matches
    for it, not a generic captioning model with no web knowledge."""
    if not web_detection:
        return None
    labels = [l.get("label") for l in (web_detection.get("bestGuessLabels") or []) if l.get("label")]
    return ", ".join(labels) if labels else None

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
    Tries Google Cloud Vision's Web Detection first (genuine reverse-
    image search) when GOOGLE_VISION_API_KEY is configured - this is
    what actually closes the gap the free pipeline can't: finding a
    source for a content-only photo with no useful text at all. Falls
    back to the free OCR+perceptual-hash+keyword-search pipeline
    (_analyze_image_free_pipeline) when Vision isn't configured, or
    when it runs but finds no matching pages.
    """
    web_detection = _web_detect(original_image_bytes)
    vision_summary = _best_guess_summary(web_detection)

    if web_detection is not None:
        scored = _score_web_detection(web_detection)
        if scored:
            content_summary = (ocr_text.strip() if ocr_text and ocr_text.strip() else None) or vision_summary
            logger.info("[media-source] Vision Web Detection found %d matching page(s)", len(scored))
            return {
                "outcome": "found",
                "topMatch": scored[0],
                "otherMatches": scored[1:],
                "ocrText": ocr_text or None,
                "contentSummary": content_summary,
            }
        logger.info("[media-source] Vision Web Detection configured but found no matching pages - falling back to free pipeline")

    result = _analyze_image_free_pipeline(original_image_bytes, ocr_text, candidates)
    if not result.get("contentSummary") and vision_summary:
        result["contentSummary"] = vision_summary
        if result["outcome"] == "source_not_found":
            result["outcome"] = "content_identified_source_not_found"
    return result


def _analyze_image_free_pipeline(original_image_bytes: bytes, ocr_text: str, candidates: list) -> dict:
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


def analyze_video(ocr_text: str, candidates: list, keyframe_images: list) -> dict:
    """
    keyframe_images: raw JPEG bytes for a handful of representative
    keyframes (already capped small client-side - each is a billed
    Vision API call, so only MAX_VIDEO_KEYFRAMES_FOR_VISION of them are
    checked here regardless of how many were passed in).

    Runs Web Detection on each keyframe and aggregates by destination
    page: a page whose image matched MULTIPLE keyframes is meaningfully
    stronger evidence than matching just one (the spec's own suggested
    "Multiple keyframes matched" evidence line), which the text-only
    pipeline below has no way to express at all. Falls back to
    analyze_video_candidates() (text-only) when Vision isn't
    configured, or ran but found nothing.
    """
    if GOOGLE_VISION_API_KEY and keyframe_images:
        page_matches = {}
        for frame_bytes in keyframe_images[:MAX_VIDEO_KEYFRAMES_FOR_VISION]:
            web_detection = _web_detect(frame_bytes)
            if not web_detection:
                continue
            for page in web_detection.get("pagesWithMatchingImages", []):
                url = page.get("url") or ""
                if not url:
                    continue
                full = [m.get("url") for m in page.get("fullMatchingImages", []) if m.get("url")]
                partial = [m.get("url") for m in page.get("partialMatchingImages", []) if m.get("url")]
                if not full and not partial:
                    continue

                entry = page_matches.setdefault(url, {
                    "count": 0, "title": page.get("pageTitle"),
                    "has_full": False, "thumbnail": (full or partial)[0],
                })
                entry["count"] += 1
                if full:
                    entry["has_full"] = True

        if page_matches:
            scored = []
            for url, info in page_matches.items():
                multi = info["count"] > 1
                if info["has_full"]:
                    confidence = "likely_original_source"
                    confidence_percent = 95 if multi else 85
                    evidence = ["Multiple keyframes matched (Google Web Detection)"] if multi else ["Exact visual match (Google Web Detection)"]
                else:
                    confidence = "possible_source"
                    confidence_percent = 75 if multi else 60
                    evidence = ["Multiple keyframes partially matched (Google Web Detection)"] if multi else ["Partial/cropped visual match (Google Web Detection)"]
                scored.append({
                    "url": url,
                    "domain": _domain(url),
                    "title": info["title"] or None,
                    "confidence": confidence,
                    "confidencePercent": confidence_percent,
                    "matchKind": "visual_match",
                    "evidence": evidence,
                    "thumbnailURL": info["thumbnail"],
                    "_rank": (0 if info["has_full"] else 1, -info["count"]),
                })
            scored.sort(key=lambda m: m["_rank"])
            for m in scored:
                m.pop("_rank", None)

            logger.info("[media-source] video Vision Web Detection found %d matching page(s) across keyframes", len(scored))
            content_summary = ocr_text.strip() if ocr_text and ocr_text.strip() else None
            return {
                "outcome": "found",
                "topMatch": scored[0],
                "otherMatches": scored[1:],
                "ocrText": ocr_text or None,
                "contentSummary": content_summary,
            }
        logger.info("[media-source] video Vision Web Detection configured but found nothing across keyframes - falling back to text-only pipeline")

    return analyze_video_candidates(ocr_text, candidates)


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


# --------------------------------------------------------------------
# Audio mode (Phase 3) - real acoustic-fingerprint music recognition.
# Two providers, tried in order:
#
#   1. AcoustID (free) - Chromaprint fingerprinting (the `fpcalc` binary,
#      installed via the Dockerfile's `libchromaprint-tools` package) +
#      AcoustID's crowd-sourced fingerprint database, tied to
#      MusicBrainz for metadata. Needs ACOUSTID_API_KEY - free
#      registration at acoustid.org/api-key, NOT the same as installing
#      Chromaprint itself. IMPORTANT: AcoustID's free web-service tier
#      is documented as being for non-commercial/open-source use only
#      (acoustid.org/webservice) - that's a real ToS consideration for
#      a paid app, not something this code can resolve for you.
#      Coverage is real but narrower than AudD's, since it depends on
#      volunteer-submitted fingerprints rather than a licensed catalog.
#   2. AudD (paid, api.audd.io) - broader commercial-catalog coverage,
#      used only if AcoustID isn't configured or didn't find anything.
#
# Neither is a Hugging Face model - no free HF model can identify an
# arbitrary song the way a real fingerprinting database can (see
# MODEL_CONFIG's note on microsoft/wavlm-base-plus). Adding a third
# provider later (e.g. ACRCloud) means adding another
# _identify_via_<provider>() function and a line in identify_audio()'s
# try-in-order chain - the route and the iOS side never need to change.
# --------------------------------------------------------------------

ACOUSTID_API_KEY = os.environ.get("ACOUSTID_API_KEY")
# AcoustID's own match score is 0-1; below this, treat it as no real
# match rather than surfacing a shaky guess as a confirmed song.
ACOUSTID_MIN_SCORE = 0.5

AUDD_API_TOKEN = os.environ.get("AUDD_API_TOKEN")
AUDD_URL = "https://api.audd.io/"
AUDD_TIMEOUT_S = 20
# AudD's own documented limit for the standard (non-enterprise) endpoint.
AUDD_MAX_BYTES = 10 * 1024 * 1024


def _empty_audio_result(outcome: str) -> dict:
    return {
        "outcome": outcome,
        "title": None,
        "artist": None,
        "album": None,
        "matchingSource": None,
    }


def _identify_via_acoustid(audio_bytes: bytes) -> Optional[dict]:
    """
    Returns None if AcoustID itself isn't usable right now (no API key,
    fpcalc/pyacoustid missing, or the request errored) - the caller
    reads None as "try the next provider," not as a real "no song"
    answer. Returns an actual result dict once AcoustID was genuinely
    asked and gave a definitive answer (identified or not_found).
    """
    if not ACOUSTID_API_KEY:
        return None
    if not _ACOUSTID_LIB_AVAILABLE:
        logger.warning("[media-source] ACOUSTID_API_KEY is set but the 'pyacoustid' package isn't installed")
        return None
    if not shutil.which("fpcalc"):
        logger.warning("[media-source] ACOUSTID_API_KEY is set but fpcalc isn't installed (check the Dockerfile has libchromaprint-tools)")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Calling fingerprint_file() + lookup() directly instead of the
        # higher-level acoustid.match() - match()'s own result parser
        # (parse_lookup_result) throws away AcoustID's actual error
        # detail on failure and raises WebServiceError("status: error"),
        # which tells us nothing. Doing it ourselves means a real
        # error.code/error.message from AcoustID's response shows up in
        # the logs instead of that useless generic string.
        #
        # force_fpcalc=True: the Dockerfile only installs the fpcalc CLI
        # tool, not the Chromaprint dynamic library pyacoustid would
        # otherwise prefer - forcing fpcalc avoids depending on a
        # library that isn't actually installed.
        duration, fingerprint = acoustid.fingerprint_file(tmp_path, force_fpcalc=True)
        response = acoustid.lookup(ACOUSTID_API_KEY, fingerprint, duration, meta="recordings")

        if response.get("status") != "ok":
            error = response.get("error") or {}
            logger.warning(
                "[media-source] AcoustID lookup rejected the request - code=%s message=%r (full response=%s). "
                "If message mentions an invalid/unknown key, this is very likely a personal ACCOUNT key rather "
                "than an APPLICATION key - AcoustID requires an application key (acoustid.org/api-key -> "
                "'Register a new application') for lookups specifically.",
                error.get("code"), error.get("message"), response,
            )
            return None

        best = None
        for r in response.get("results") or []:
            score = r.get("score", 0)
            for recording in r.get("recordings") or []:
                title = recording.get("title")
                if not title:
                    continue
                artists = recording.get("artists") or []
                artist = ", ".join(a.get("name") for a in artists if a.get("name")) or None
                if best is None or score > best[0]:
                    best = (score, title, artist)

        if best is None or best[0] < ACOUSTID_MIN_SCORE:
            logger.info("[media-source] AcoustID processed the audio - no confident match (best=%s)", best)
            return _empty_audio_result("not_found")

        score, title, artist = best
        logger.info("[media-source] AcoustID identified: %s - %s (score=%.2f)", artist, title, score)
        return {
            "outcome": "identified",
            "title": title,
            "artist": artist,
            "album": None,
            "matchingSource": "MusicBrainz (via AcoustID)",
        }

    except Exception as e:
        logger.info("[media-source] AcoustID lookup failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _identify_via_audd(audio_bytes: bytes) -> dict:
    """
    Always returns a real result dict (never None) - "not_configured"
    is itself a valid, final answer here since AudD is the last
    provider in the chain. Never fabricates a title/artist/album, and
    never invents a numeric confidence score - AudD's standard endpoint
    doesn't return one (matching is a binary fingerprint hit), so the
    client shows a fixed "verified match" label instead of a made-up
    percentage.
    """
    if not AUDD_API_TOKEN:
        return _empty_audio_result("not_configured")

    if len(audio_bytes) > AUDD_MAX_BYTES:
        logger.info("[media-source] audio file (%d bytes) exceeds AudD's 10MB standard-endpoint limit", len(audio_bytes))
        return _empty_audio_result("not_found")

    try:
        resp = requests.post(
            AUDD_URL,
            data={"api_token": AUDD_API_TOKEN, "return": "apple_music,spotify"},
            files={"file": ("audio", audio_bytes)},
            timeout=AUDD_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.info("[media-source] AudD request failed: %s", e)
        return _empty_audio_result("not_found")

    if payload.get("status") != "success":
        logger.warning("[media-source] AudD returned a non-success response: %s", payload)
        return _empty_audio_result("not_found")

    result = payload.get("result")
    if not result:
        logger.info("[media-source] AudD analyzed the audio - no fingerprint match (expected for speech/noise/non-music)")
        return _empty_audio_result("not_found")

    if result.get("spotify"):
        matching_source = "Spotify"
    elif result.get("apple_music"):
        matching_source = "Apple Music"
    else:
        matching_source = result.get("song_link")

    logger.info("[media-source] AudD identified: %s - %s", result.get("artist"), result.get("title"))
    return {
        "outcome": "identified",
        "title": result.get("title"),
        "artist": result.get("artist"),
        "album": result.get("album"),
        "matchingSource": matching_source,
    }


# --------------------------------------------------------------------
# Audio result cache - the actual "don't exhaust the free plan" tweak.
# Keyed by a SHA-256 of the raw audio bytes, so the exact same clip
# (submitted twice during testing, or uploaded by many different users
# if something goes viral) never triggers a second AcoustID/AudD call.
# Backed by the Redis instance this app already runs for Veronica.py's
# chat history - a plain in-process dict wouldn't help much here, since
# Gunicorn runs multiple worker processes (WEB_CONCURRENCY) that don't
# share memory, so a repeat request could easily land on a different
# worker than the one that saw it first. Redis is shared across all of
# them.
#
# "identified" results are cached for a long time (30 days) - a song's
# identity is a fact that doesn't change. "not_found" results are
# cached for a much shorter time (1 day) - that outcome depends on
# which providers are configured and their current database coverage,
# both of which CAN change (e.g. you add AUDD_API_TOKEN later), so a
# stale "not_found" shouldn't stick around too long. "not_configured"
# is never cached at all - it reflects your server's current setup,
# not anything about the audio itself.
# --------------------------------------------------------------------

_AUDIO_CACHE_TTL_IDENTIFIED_S = 30 * 24 * 60 * 60
_AUDIO_CACHE_TTL_NOT_FOUND_S = 1 * 24 * 60 * 60


def _audio_cache_key(audio_bytes: bytes) -> str:
    return "media-source:audio:" + hashlib.sha256(audio_bytes).hexdigest()


def _get_cached_audio_result(audio_bytes: bytes) -> Optional[dict]:
    try:
        raw = redis_client.get(_audio_cache_key(audio_bytes))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.info("[media-source] audio cache read failed (continuing without cache): %s", e)
        return None


def _cache_audio_result(audio_bytes: bytes, result: dict) -> None:
    ttl = _AUDIO_CACHE_TTL_IDENTIFIED_S if result["outcome"] == "identified" else _AUDIO_CACHE_TTL_NOT_FOUND_S
    try:
        redis_client.setex(_audio_cache_key(audio_bytes), ttl, json.dumps(result))
    except Exception as e:
        logger.info("[media-source] audio cache write failed (continuing without cache): %s", e)


def identify_audio(audio_bytes: bytes) -> dict:
    """
    Returns a dict with `outcome` of "identified" / "not_found" /
    "not_configured" - matches iOS's AudioIdentificationResult exactly.

    Checks the Redis-backed result cache first (see the block above) -
    a cache hit means NO provider call happens at all, which is the
    actual mechanism that keeps repeated/duplicate lookups from eating
    into a free-tier quota. On a miss, tries AcoustID (free) first,
    then AudD (paid) if AcoustID isn't configured or didn't find
    anything. "not_configured" and "not_found" are DELIBERATELY
    distinct and must never be blurred into the same UI message:
    not_configured means NEITHER provider is set up at all, while
    not_found means at least one of them actually analyzed the audio
    and found no fingerprint match - which is also the CORRECT,
    expected outcome for speech, traffic, wind, or any non-music
    recording, since both providers' databases are music-specific.
    """
    cached = _get_cached_audio_result(audio_bytes)
    if cached is not None:
        logger.info("[media-source] audio result served from cache (outcome=%s) - no provider call made", cached.get("outcome"))
        return cached

    acoustid_result = _identify_via_acoustid(audio_bytes)
    if acoustid_result is not None and acoustid_result["outcome"] == "identified":
        result = acoustid_result
    else:
        audd_result = _identify_via_audd(audio_bytes)
        if audd_result["outcome"] == "identified":
            result = audd_result
        elif acoustid_result is not None or audd_result["outcome"] != "not_configured":
            result = _empty_audio_result("not_found")
        else:
            logger.info("[media-source] neither ACOUSTID_API_KEY nor AUDD_API_TOKEN is set - audio identification not configured")
            result = _empty_audio_result("not_configured")

    if result["outcome"] != "not_configured":
        _cache_audio_result(audio_bytes, result)
    return result
