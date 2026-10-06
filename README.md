# Camera trap classifier

A small local HTTP service that classifies trail-camera / camera-trap photos and
short clips with [Google SpeciesNet](https://github.com/google/cameratrapai).
Everything runs on your own machine; no images leave your network.

It is the optional local model for the
[Trailwatch](https://github.com/corgicommander/ha-trailwatch) Home Assistant
integration, but any program can use its HTTP API.

## Run

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). About 1 GB of memory while
running; a photo takes about a second on a recent computer.

```bash
git clone https://github.com/corgicommander/camera-trap-classifier
cd camera-trap-classifier
uv sync
CTC_COUNTRY=US CTC_ADMIN1_REGION=CA uv run camera-trap-classifier
```

The first start downloads the SpeciesNet model (about 500 MB). Check it with
`curl http://localhost:8765/health`, then enter `http://<this computer's IP>:8765`
as the classifier URL in Trailwatch. Use a fixed IP address for that computer.

| Variable | Default | Meaning |
|---|---|---|
| `CTC_HOST` | `0.0.0.0` | Address to listen on |
| `CTC_PORT` | `8765` | Port |
| `CTC_COUNTRY` | none | Default ISO country (alpha-2 or alpha-3); improves species accuracy |
| `CTC_ADMIN1_REGION` | none | Default first-level region, such as a US state code |
| `CTC_TOKEN` | none | If set, requests must send `Authorization: Bearer <token>` |
| `CTC_MODEL` | SpeciesNet default | SpeciesNet model name |

## API

`POST /v1/classify` with the image bytes as the body (optional query
parameters `country`, `admin1_region`):

```json
{
  "kind": "animal",
  "common_name": "eastern fox squirrel",
  "score": 0.996,
  "source": "classifier",
  "taxonomy": {"class": "mammalia", "order": "rodentia", "family": "sciuridae", "genus": "sciurus", "species": "niger"},
  "detections": [{"label": "animal", "confidence": 0.97, "bbox": [0.5, 0.55, 0.28, 0.4]}],
  "top_classifications": [{"common_name": "eastern fox squirrel", "score": 0.996}],
  "model_version": "4.0.3a"
}
```

`kind` is one of `blank`, `animal`, `human`, `vehicle`, or `unknown` when SpeciesNet could not classify the image. `GET /health` reports status.

`POST /v1/classify_video` with MP4 bytes as the body classifies a clip by sampling
evenly spaced frames (query `frames`, default 6, max 16). The answer has the same
fields as a photo result, taken from the most informative frame (animal > human >
vehicle > unknown > blank, then the most specific label, then highest score), plus `frame_index`, `frames` (time,
kind, name and score per sampled frame), `labels` (everything seen) and
`frame_jpeg`, the deciding frame as base64 JPEG (`include_frame=false` omits it).

## Start automatically

### macOS

`launchd/local.camera-trap-classifier.plist` is an example LaunchAgent; copy it to
`~/Library/LaunchAgents/`, adjust paths, and load it with
`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.camera-trap-classifier.plist`.

### Linux (systemd)

```ini
# /etc/systemd/system/camera-trap-classifier.service
[Unit]
Description=Camera trap classifier
After=network-online.target

[Service]
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/camera-trap-classifier
Environment=CTC_COUNTRY=US CTC_ADMIN1_REGION=CA
ExecStart=/home/YOUR_USER/camera-trap-classifier/.venv/bin/camera-trap-classifier
Restart=always

[Install]
WantedBy=multi-user.target
```

Then `sudo systemctl enable --now camera-trap-classifier`.

## Tested with

Only tested on macOS (Apple Silicon M2 Pro, 16 GB) with Python 3.11 and SpeciesNet 5,
serving one Home Assistant instance with two cameras in the United States. Linux,
Windows and Docker have not been tried yet.

## How this was made

Written by **[Claude Code](https://claude.com/claude-code)** (Anthropic's AI coding
agent) with the repository owner, who tested it on the setup above.

## License

MIT for this code (see [LICENSE](LICENSE)). SpeciesNet and its model weights are
Google's, under the Apache 2.0 license; see the
[SpeciesNet repository](https://github.com/google/cameratrapai).
