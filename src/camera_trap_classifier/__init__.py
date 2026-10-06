"""Local HTTP service that classifies camera-trap photos with Google SpeciesNet."""

from __future__ import annotations

import base64
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pycountry
from fastapi import FastAPI, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

__version__ = "0.2.1"

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_FRAMES = 16

# Which frame of a video speaks for it: anything alive beats an empty frame.
_KIND_RANK = {"animal": 4, "human": 3, "vehicle": 2, "unknown": 1, "blank": 0}

app = FastAPI(title="Camera trap classifier", version=__version__)

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load SpeciesNet once; the first call downloads the model if needed."""
    global _model
    if _model is None:
        from speciesnet import DEFAULT_MODEL, SpeciesNet

        _model = SpeciesNet(os.environ.get("CTC_MODEL", DEFAULT_MODEL))
    return _model


def _iso3(country: str | None) -> str | None:
    """Accept ISO 3166-1 alpha-2 or alpha-3 and return alpha-3."""
    if not country:
        return None
    code = country.upper()
    match = pycountry.countries.get(alpha_2=code) or pycountry.countries.get(alpha_3=code)
    if match is None:
        raise HTTPException(400, f"Unknown country code: {country}")
    return match.alpha_3


def _parse_label(label: str) -> dict[str, Any]:
    """Split a SpeciesNet label 'id;class;order;family;genus;species;common name'."""
    _, cls, order, family, genus, species, common = (label.split(";") + [""] * 7)[:7]
    return {
        "common_name": common,
        "taxonomy": {
            "class": cls,
            "order": order,
            "family": family,
            "genus": genus,
            "species": species,
        },
    }


def _normalize(prediction: dict[str, Any]) -> dict[str, Any]:
    if "prediction" not in prediction:
        raise HTTPException(
            422, f"SpeciesNet could not process the image: {prediction.get('failures')}"
        )
    label = _parse_label(prediction["prediction"])
    common = label["common_name"]
    if common == "no cv result":
        # SpeciesNet's marker for "could not classify".
        common = "unknown"
        label["taxonomy"] = {key: "" for key in label["taxonomy"]}
    kind = common if common in ("blank", "human", "vehicle", "unknown") else "animal"
    classifications = prediction.get("classifications", {})
    return {
        "kind": kind,
        "common_name": common,
        "score": prediction.get("prediction_score"),
        "source": prediction.get("prediction_source"),
        "taxonomy": label["taxonomy"],
        "detections": [
            {"label": d.get("label"), "confidence": d.get("conf"), "bbox": d.get("bbox")}
            for d in prediction.get("detections", [])
        ],
        "top_classifications": [
            {"common_name": _parse_label(c)["common_name"], "score": s}
            for c, s in zip(
                classifications.get("classes", [])[:5],
                classifications.get("scores", [])[:5],
            )
        ],
        "model_version": prediction.get("model_version"),
    }


def _check_token(authorization: str | None) -> None:
    token = os.environ.get("CTC_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(401, "Missing or invalid bearer token")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "model_loaded": _model is not None}


@app.post("/v1/classify")
async def classify(
    request: Request,
    country: str | None = Query(None, description="ISO 3166-1 alpha-2 or alpha-3"),
    admin1_region: str | None = Query(None, description="First-level region, e.g. a US state code"),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Classify one JPEG or PNG image sent as the raw request body."""
    _check_token(authorization)
    body = await request.body()
    if not body:
        raise HTTPException(400, "Send the image bytes as the request body")
    if len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large")
    country, admin1_region = _location(country, admin1_region)

    def run() -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.jpg"
            path.write_bytes(body)
            return _predict([path], country, admin1_region)[0]

    return await run_in_threadpool(run)


@app.post("/v1/classify_video")
async def classify_video(
    request: Request,
    country: str | None = Query(None, description="ISO 3166-1 alpha-2 or alpha-3"),
    admin1_region: str | None = Query(None, description="First-level region, e.g. a US state code"),
    frames: int = Query(6, ge=1, le=MAX_VIDEO_FRAMES, description="Frames to sample"),
    include_frame: bool = Query(True, description="Return the deciding frame as base64 JPEG"),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Classify an MP4 sent as the raw body by sampling evenly spaced frames.

    The answer is the most informative frame (animal > human > vehicle >
    unknown > blank, then highest score); every sampled frame is listed too.
    """
    _check_token(authorization)
    body = await request.body()
    if not body:
        raise HTTPException(400, "Send the video bytes as the request body")
    if len(body) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video too large")
    country, admin1_region = _location(country, admin1_region)

    def run() -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(body)
            stills = _extract_frames(video, Path(tmp), frames)
            results = _predict([path for path, _ in stills], country, admin1_region)
            per_frame = [
                {"time": seconds, **result} for (_, seconds), result in zip(stills, results)
            ]
            best_index = max(
                range(len(per_frame)),
                key=lambda i: (
                    _KIND_RANK.get(per_frame[i]["kind"], 0),
                    # A precise label only counts when the model was fairly sure.
                    _specificity(per_frame[i]) if (per_frame[i]["score"] or 0) >= 0.6 else 0,
                    per_frame[i]["score"] or 0,
                ),
            )
            best = per_frame[best_index]
            answer: dict[str, Any] = {
                **best,
                "frame_index": best_index,
                "frames": [
                    {
                        "time": f["time"],
                        "kind": f["kind"],
                        "common_name": f["common_name"],
                        "score": f["score"],
                    }
                    for f in per_frame
                ],
                "labels": sorted(
                    {f["common_name"] for f in per_frame if f["kind"] in ("animal", "human", "vehicle")}
                ),
            }
            if include_frame:
                answer["frame_jpeg"] = base64.b64encode(stills[best_index][0].read_bytes()).decode()
            return answer

    return await run_in_threadpool(run)


def _specificity(result: dict[str, Any]) -> int:
    """How precise a label is: a species beats "animal" or an order name."""
    taxonomy = result.get("taxonomy") or {}
    for rank, level in enumerate(("species", "genus", "family", "order", "class")):
        if taxonomy.get(level):
            return 5 - rank
    return 0


def _location(country: str | None, admin1_region: str | None) -> tuple[str | None, str | None]:
    return (
        _iso3(country or os.environ.get("CTC_COUNTRY")),
        admin1_region or os.environ.get("CTC_ADMIN1_REGION"),
    )


def _predict(paths: list[Path], country: str | None, admin1_region: str | None) -> list[dict[str, Any]]:
    """Run SpeciesNet on image files; results come back in input order."""
    instances = []
    for path in paths:
        instance: dict[str, Any] = {"filepath": str(path)}
        if country:
            instance["country"] = country
        if admin1_region:
            instance["admin1_region"] = admin1_region
        instances.append(instance)
    with _model_lock:
        result = _get_model().predict(
            instances_dict={"instances": instances},
            run_mode="single_thread",
            progress_bars=False,
        )
    by_path = {p["filepath"]: p for p in result["predictions"]}
    return [_normalize(by_path[str(path)]) for path in paths]


def _extract_frames(video: Path, folder: Path, count: int) -> list[tuple[Path, float]]:
    """Save `count` evenly spaced frames as JPEGs; return (path, seconds) pairs."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise HTTPException(422, "Could not read the video")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        if total <= 0:
            raise HTTPException(422, "The video has no frames")
        count = min(count, total)
        # Skip the very first and last frames: they are often dark or blurred.
        positions = [int(total * (i + 0.5) / count) for i in range(count)]
        stills = []
        for number, position in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                continue
            path = folder / f"frame{number:02d}.jpg"
            # Pillow writes the JPEG: SpeciesNet's cv2 build has a reduced imwrite.
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, quality=92)
            stills.append((path, round(position / fps, 2) if fps else 0.0))
        if not stills:
            raise HTTPException(422, "Could not decode any video frames")
        return stills
    finally:
        capture.release()


def main() -> None:
    import uvicorn

    # Load the model before accepting requests so the first photo is fast.
    _get_model()
    uvicorn.run(
        app,
        host=os.environ.get("CTC_HOST", "0.0.0.0"),
        port=int(os.environ.get("CTC_PORT", "8765")),
        log_level="info",
    )
