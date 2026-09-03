# PRD: Free Model Discovery and Arena-Inspired Routing

> **⚠ This is a proposal. None of it is implemented.**
>
> Nothing described in this document exists in Croft today. It is kept in the
> repository as a design record, not as documentation — do not read it as a
> description of how Croft works, and do not file bugs against it. For what
> Croft actually does, see the [README](../README.md).
>
> The shipped model catalog is the static `models_manifest.json`, documented
> under [Adding custom models](../README.md#adding-custom-models).

Status: Proposal — not implemented, not scheduled
Owner: Croft
Last updated: 2026-08-04

## 1. Summary

Croft already has a static model manifest, hardware detection,
recommendations, resumable Hugging Face downloads, local inference through GGUF,
and OpenAI-compatible endpoint support. The next product step is a dynamic model
catalog that can discover freely available online models, normalize their
metadata, rank them for the user's machine, and make them installable or usable
from the existing web UI.

Arena.ai should influence this feature in two ways:

1. Use Arena-style comparison data as a quality/ranking signal.
2. Add a lightweight "battle" and task-router layer later, so the local system
   can learn which installed/free models work best for the user's tasks.

Arena should not be treated as the primary source for downloadable model files.
Downloadable local models should come from sources such as Hugging Face and, for
runtime convenience, Ollama-compatible registries. Free hosted API models should
come from providers with explicit pricing metadata such as OpenRouter. Arena API
can be added as an optional OpenAI-compatible gateway if the user has an Arena
API key.

## 2. Current Local System Assessment

Relevant existing files:

- `models_manifest.json`: static catalog used for recommendation and download.
- `installer/recommend.py`: loads the manifest, estimates memory, ranks models.
- `installer/downloader.py`: supports resumable Hugging Face snapshots and direct
  HTTP downloads.
- `installer/main.py`: CLI flow for detect, recommend, install, pull, status,
  and uninstall.
- `webui/app.py`: FastAPI routes for models, recommendations, status, chat,
  image, video, and settings.
- `webui/inference.py`: lists installed chat models, detects local
  OpenAI-compatible endpoints, supports embedded GGUF through llama-cpp-python.
- `webui/static/index.html`: current single-file UI with model picker and
  endpoint settings.

Current limitation:

- Model discovery is manual and static.
- `models_manifest.json` only contains a curated starter list.
- There is no online catalog sync, metadata normalization, free/license filter,
  quality ranking, install candidate review, or UI for browsing new models.

## 3. Definition of "Free"

"Free" must be explicit because model availability is messy.

Supported statuses:

- `free_download`: public model weights can be downloaded at no charge.
- `free_download_gated`: weights are free but require license acceptance or a
  token, such as gated Hugging Face repositories.
- `free_api`: hosted inference has zero-dollar pricing at discovery time.
- `free_noncommercial`: free to download/use but license restricts commercial
  use.
- `trial_or_quota`: free only under a temporary quota, promo, or account limit.
- `paid`: paid-only usage or paid license.
- `unknown`: missing price/license data.

MVP filters should default to:

- Include: `free_download`, `free_api`.
- Show with warning: `free_download_gated`, `free_noncommercial`.
- Exclude by default: `trial_or_quota`, `paid`, `unknown`.

The product must say "all discoverable free models from supported sources," not
"all free models on the internet." A complete internet-wide model list is not
reliable or legally safe.

## 4. Research Notes: Arena.ai

Arena.ai is the product built around the former LMArena ecosystem. Its useful
patterns for this project are:

- Public comparison experience: users compare model outputs in battle mode and
  votes feed model rankings.
- Leaderboards: ranked models across modalities and task types, useful as a
  quality signal.
- Model picker behavior: users can select specific models or use a router.
- Router behavior: task and preference based model selection is a good pattern
  for future local routing.
- Dataset availability: Arena publishes leaderboard data on Hugging Face, which
  can be used to enrich model quality metadata when licenses allow.
- API compatibility: Arena's developer docs describe an OpenAI-compatible API
  surface, so it can fit your existing `webui/inference.py` endpoint model.

Important product interpretation:

- Arena tells you which models users prefer.
- Hugging Face/Ollama tell you which local models can be downloaded.
- OpenRouter-like APIs tell you which hosted models currently have free pricing.

## 5. Goals

1. Discover free local-download models from supported online catalogs.
2. Discover free hosted API models from supported model APIs.
3. Normalize model metadata into one local catalog.
4. Filter by license, gated status, modality, model format, size, and hardware
   compatibility.
5. Rank by runnable fit first, then quality/popularity/freshness.
6. Let users approve and install models using the existing downloader.
7. Add UI and API endpoints for browsing and installing discovered models.
8. Preserve the current curated manifest and installer behavior.

## 6. Non-Goals

- Crawling the entire internet for model files.
- Downloading gated or license-restricted models without explicit user consent.
- Automatically trusting unofficial mirrors for safety-sensitive use.
- Replacing the current static manifest.
- Benchmarking every model locally in MVP.
- Guaranteeing commercial usability for every "free" model.

## 7. Users and Use Cases

Primary user: a local AI user who wants to run strong free models without
manually researching repos, file formats, hardware needs, and license terms.

Admin/developer user: the person maintaining the local app and deciding which
sources are allowed.

Core use cases:

1. As a user, I can open a model catalog and see free text/image/video models
   that my machine can run.
2. As a user, I can filter to "downloadable only," "free API only," "commercial
   use allowed," "not gated," "GGUF only," or "fits my hardware."
3. As a user, I can see why a model is recommended: size, format, license,
   source, expected memory, disk cost, and quality score.
4. As a user, I can approve a model download and watch install progress.
5. As a user, I can add a free hosted API model through an OpenAI-compatible
   endpoint without downloading weights.
6. As an admin, I can refresh the catalog and review new or suspicious sources.

## 8. Data Sources

### 8.1 Hugging Face Hub

Purpose:

- Primary source for downloadable free local models.

What to fetch:

- Model id, author/org, tags, pipeline tag, license, gated/private status,
  downloads, likes, last modified, card metadata, siblings/files, file sizes,
  safetensors metadata where available.

How to use:

- Use `huggingface_hub.HfApi.list_models(...)` for search.
- Use `model_info(..., files_metadata=True)` for file inspection.
- Keep query sets focused by modality and format.

Initial query targets:

- Text GGUF: `library:gguf`, `text-generation`, `text-generation-inference`,
  file suffix `.gguf`.
- Text safetensors: `text-generation`, `.safetensors`, not gated by default.
- Image diffusion: `diffusers`, `text-to-image`, `image-to-image`,
  `safetensors`, `model_index.json`.
- Video diffusion: `text-to-video`, `image-to-video`, `diffusers`.
- Embeddings later: `sentence-transformers`, `feature-extraction`.

MVP scope:

- Prioritize GGUF text models and Diffusers image models because your app
  already supports those execution paths best.

### 8.2 Arena Leaderboard Dataset

Purpose:

- Quality/ranking enrichment, not download.

What to fetch:

- Model names, Arena score/rank, modality, vote counts, license metadata where
  available, benchmark subset.

How to use:

- Load the public leaderboard dataset.
- Normalize names and map to Hugging Face/OpenRouter/Ollama entries.
- Store as optional quality metadata.

Do not:

- Treat Arena model names as installable without a source URL and license data.

### 8.3 OpenRouter

Purpose:

- Discover free hosted API models.

What to fetch:

- Model id, name, context length, architecture/modality, supported parameters,
  pricing fields, endpoint/provider metadata if exposed.

Free filter:

- A model is `free_api` only when prompt, completion, image, request, and other
  relevant pricing fields are zero or explicitly free.
- Also treat `:free` model ids as candidates, but still verify pricing fields.

How to use:

- Add OpenRouter as an optional OpenAI-compatible provider.
- Store `api_base=https://openrouter.ai/api/v1`.
- Require user-provided key where needed.

### 8.4 Ollama

Purpose:

- Optional local runtime and convenience install path.

MVP approach:

- Detect local Ollama through the already configured endpoint
  `http://127.0.0.1:11434/v1`.
- Discover locally installed Ollama models through its local API.
- For online discovery, keep an allowlisted registry file at first rather than
  scraping the public website.

Later:

- Add `ollama pull <name>` as a runtime action for known safe model names.

### 8.5 Arena API

Purpose:

- Optional hosted provider and router, if the user has an Arena API key.

How to use:

- Treat Arena as another OpenAI-compatible endpoint.
- Use its model listing endpoint when authenticated.
- Do not assume Arena models are free unless API response or product policy
  gives pricing/free status.

## 9. Catalog Data Model

Store discovered data in a local SQLite database or JSON cache. SQLite is
recommended because the app already uses SQLite through `webui/chatstore.py`
and catalog queries need filtering/sorting.

Table: `model_catalog`

Required fields:

- `id`: local stable id, for example `hf:Qwen/Qwen2.5-7B-Instruct-GGUF`.
- `canonical_name`: normalized family/name.
- `display_name`: user-facing name.
- `source`: `huggingface`, `arena`, `openrouter`, `ollama`, `manual`.
- `source_url`: canonical URL.
- `provider_model_id`: provider-specific model id.
- `availability`: one of the free statuses in section 3.
- `modality`: `text`, `image`, `video`, `embedding`, `audio`, `multimodal`.
- `tasks`: JSON array.
- `format`: `gguf`, `gptq`, `safetensors`, `diffusers`, `api`, `ollama`.
- `license`: SPDX-ish or source label.
- `commercial_use`: `yes`, `no`, `unknown`.
- `gated`: boolean.
- `requires_token`: boolean.
- `official_source`: boolean.
- `trust_level`: `official`, `known_mirror`, `community`, `unknown`.
- `params`: integer nullable.
- `quantization`: nullable string.
- `context_length`: nullable integer.
- `size_bytes`: nullable integer.
- `recommended_min_vram_gb`: nullable float.
- `recommended_min_ram_gb`: nullable float.
- `download_url`: nullable string.
- `hf_repo`: nullable string.
- `hf_revision`: nullable string, ideally commit sha when approved.
- `allow_patterns`: JSON array nullable.
- `api_base`: nullable string.
- `api_pricing`: JSON object nullable.
- `arena_rank`: nullable integer.
- `arena_score`: nullable float.
- `arena_votes`: nullable integer.
- `downloads`: nullable integer.
- `likes`: nullable integer.
- `last_modified`: nullable timestamp.
- `discovered_at`: timestamp.
- `updated_at`: timestamp.

Table: `model_aliases`

- `catalog_id`
- `alias`
- `source`

Table: `model_install_candidates`

- `catalog_id`
- `status`: `new`, `approved`, `rejected`, `installed`, `failed`
- `review_notes`
- `approved_at`
- `installed_model_id`

## 10. Normalization Rules

Name normalization:

- Lowercase.
- Remove provider prefixes such as `openai/`, `meta-llama/`, `qwen/` for alias
  matching, but keep original ids.
- Strip quantization suffixes into `quantization`, for example `Q4_K_M`,
  `Q5_K_M`, `IQ4_XS`.
- Normalize common aliases:
  - `llama-3.1`, `llama3.1`, `llama 3.1`
  - `qwen2.5`, `qwen-2.5`
  - `mistral-nemo`, `mistral nemo`

Deduplication:

- Prefer official Hugging Face repos over mirrors for metadata.
- Prefer quantized GGUF mirrors only for runnable GGUF files when the official
  repo does not publish GGUF.
- Keep API entries separate from downloadable entries, but connect them through
  aliases.

Trust classification:

- `official`: org or author matches the model owner.
- `known_mirror`: trusted quantization publisher or curated allowlist.
- `community`: public repo with enough downloads/likes but not verified.
- `unknown`: insufficient metadata.

MVP allowlist:

- Official orgs and well-known publishers should live in
  `catalog_sources/trusted_publishers.json`.
- Any `unknown` source requires manual approval before install.

## 11. Compatibility and Ranking

The existing recommendation formula in `installer/recommend.py` should remain
the source of truth for hardware fit.

Ranking order for installable models:

1. Fits current hardware.
2. Supported by current runtime without new major dependencies.
3. Availability is `free_download`.
4. Non-gated before gated.
5. Commercial-use allowed before non-commercial/unknown.
6. Arena score/rank if mapped.
7. Downloads/likes/popularity.
8. Freshness.
9. Smaller disk size for similar quality.

Suggested score:

```text
catalog_score =
  1000 if fits_hardware else 0
+ 300 if supported_format else 0
+ 200 if availability == free_download else 0
+ 100 if not gated else 0
+ 100 if commercial_use == yes else 0
+ normalized_arena_score * 2
+ log10(downloads + 1) * 20
+ recency_score
- disk_penalty
```

Ranking order for hosted free API models:

1. Verified zero pricing.
2. Endpoint is configured and reachable.
3. Context length.
4. Arena score/rank if mapped.
5. Provider reliability/fallback metadata.
6. Modality/task match.

## 12. User Experience

Add a new "Catalog" view to the web UI.

Primary UI controls:

- Source filter: Hugging Face, OpenRouter, Arena, Ollama, Manual.
- Availability filter: Downloadable, Free API, Gated, Non-commercial.
- Modality tabs: Text, Image, Video, Embeddings.
- Runtime filter: GGUF, Diffusers, API, Ollama.
- Hardware filter: Fits this machine, Needs GPU, CPU-friendly.
- License filter: Commercial allowed, Non-commercial, Unknown.
- Sort: Recommended, Arena rank, Popularity, Newest, Smallest.

Model row/card fields:

- Name
- Source
- Format
- Size
- License
- Gated/free badge
- Fits/not fits badge
- Estimated memory
- Arena rank/score if mapped
- Actions: Details, Approve, Install, Use API, Reject

Model details panel:

- Source URLs
- License warning
- File list/selected download pattern
- Hardware estimate
- Install command preview
- Trust level
- Last synced date

Install flow:

1. User clicks Install.
2. System shows license/source/size/token warning.
3. User confirms.
4. Backend creates install job.
5. Downloader streams progress.
6. Installed model appears in existing model picker.

## 13. API Requirements

New FastAPI routes:

- `GET /api/catalog`
  - Query: `source`, `availability`, `modality`, `format`, `fits`, `license`,
    `sort`, `limit`, `cursor`.
  - Returns normalized catalog rows with fit and rank fields.

- `POST /api/catalog/sync`
  - Body: `sources`, `limit`, `refresh`, `include_gated`.
  - Starts background sync job.
  - MVP can run synchronously for small limits.

- `GET /api/catalog/sync/{job_id}`
  - Returns progress, counts, warnings, failures.

- `GET /api/catalog/{catalog_id}`
  - Returns full normalized metadata.

- `POST /api/catalog/{catalog_id}/approve`
  - Marks a model as approved for install/use.

- `POST /api/catalog/{catalog_id}/install`
  - Creates an install job using existing downloader logic.

- `POST /api/catalog/{catalog_id}/reject`
  - Hides/rejects model from install candidates.

- `POST /api/providers`
  - Saves OpenAI-compatible provider settings for free API models.

CLI commands:

- `python -m installer.main catalog sync --source huggingface --kind text --limit 500`
- `python -m installer.main catalog search --free --fits --kind text`
- `python -m installer.main catalog approve <catalog_id>`
- `python -m installer.main catalog install <catalog_id>`
- `python -m installer.main catalog export-manifest`

## 14. Implementation Plan

### Phase 1: Catalog foundation

Files to add:

- `installer/catalog.py`
- `installer/catalog_store.py`
- `installer/catalog_sources/__init__.py`
- `installer/catalog_sources/huggingface.py`
- `installer/catalog_sources/openrouter.py`
- `installer/catalog_sources/arena.py`
- `installer/catalog_sources/ollama.py`
- `installer/catalog_sources/trusted_publishers.json`
- `tests/test_catalog.py`

Tasks:

1. Create SQLite-backed catalog store under `models/catalog.db`.
2. Define `DiscoveredModel` dataclass.
3. Add normalization helpers.
4. Add free/license classifier.
5. Add hardware fit enrichment by calling `recommend.recommend(...)` logic or a
   shared helper.
6. Add CLI `catalog sync/search/export-manifest`.
7. Keep existing `models_manifest.json` untouched.

Acceptance criteria:

- Running `catalog sync --source huggingface --kind text --limit 50 --dry-run`
  prints candidates without writing.
- Running without `--dry-run` stores candidates in `models/catalog.db`.
- `catalog search --free --fits --kind text` returns only free candidates that
  can run locally.
- Tests cover license/free classification, deduplication, and ranking.

### Phase 2: Install candidate workflow

Files to modify:

- `installer/main.py`
- `installer/downloader.py`
- `installer/recommend.py`
- `models_manifest.json` only if adding curated examples.

Tasks:

1. Convert approved catalog rows into install specs.
2. For Hugging Face GGUF, choose a single best file by quant preference instead
   of downloading every GGUF shard/variant.
3. Add `allow_patterns` support to `cmd_pull`.
4. Store installed metadata in `models/installed.json` exactly as current web UI
   expects.
5. Add install progress hooks for later UI streaming.

Acceptance criteria:

- Installing a discovered GGUF model creates a model directory and updates
  `models/installed.json`.
- The model appears in `GET /api/models`.
- Existing static manifest installs still work.

### Phase 3: Catalog web UI

Files to modify:

- `webui/app.py`
- `webui/static/index.html`

Tasks:

1. Add `/api/catalog` and sync/install routes.
2. Add a Catalog tab/view.
3. Add filters, sort, detail panel, approve/install actions.
4. Add progress UI for installs.
5. Add explicit license/gated warnings.

Acceptance criteria:

- User can discover, filter, approve, install, and then select a model in chat.
- Gated models cannot be installed without explicit confirmation and token.
- Free API models can be configured as endpoint-backed models.

### Phase 4: Arena-inspired ranking and battle mode

Files to add/modify:

- `webui/battle.py`
- `webui/chatstore.py`
- `webui/app.py`
- `webui/static/index.html`

Tasks:

1. Import Arena leaderboard data as quality metadata.
2. Add pairwise local battle mode for installed/API models.
3. Store user votes locally.
4. Add a simple task router:
   - coding -> best coding-rated installed/free endpoint model
   - creative -> best creative/user-voted model
   - precise -> best factual/user-voted model
5. Keep routing explainable in UI.

Acceptance criteria:

- User can compare two models blind and vote.
- Local votes influence default model selection per task.
- Router always shows which model was selected and why.

## 15. Hugging Face Sync Algorithm

Pseudo-flow:

```python
def sync_huggingface(kind: str, limit: int):
    queries = build_queries(kind)
    for query in queries:
        for model in hf_api.list_models(**query, limit=limit):
            info = hf_api.model_info(model.modelId, files_metadata=True)
            discovered = normalize_hf_model(info)
            discovered.availability = classify_free_status(info)
            discovered.format = infer_format(info.siblings, info.tags)
            discovered.size_bytes = estimate_size(info.siblings, discovered.format)
            discovered.allow_patterns = choose_allow_patterns(info.siblings, discovered)
            discovered.trust_level = classify_trust(info)
            discovered.fit = estimate_fit(discovered, hardware_report)
            catalog_store.upsert(discovered)
```

GGUF file selection:

1. Prefer non-split `Q4_K_M`.
2. Else prefer first shard of split `Q4_K_M`.
3. Else `Q5_K_M`.
4. Else `Q4_0`.
5. Else smallest supported quant above a minimum quality threshold.

Diffusers selection:

- Require `model_index.json`.
- Download only required pipeline files.
- Prefer fp16 variants where available.

## 16. OpenRouter/Free API Sync Algorithm

Pseudo-flow:

```python
def sync_openrouter():
    models = get_json("https://openrouter.ai/api/v1/models")
    for row in models["data"]:
        pricing = row.get("pricing", {})
        is_free = all_zero_pricing(pricing)
        discovered = normalize_openrouter_model(row)
        discovered.availability = "free_api" if is_free else "paid"
        discovered.format = "api"
        discovered.api_base = "https://openrouter.ai/api/v1"
        catalog_store.upsert(discovered)
```

Rules:

- Store free API models separately from local-download models.
- Do not mark model as locally installable.
- Require user API key before use.
- Re-check pricing during sync and before first use.

## 17. Arena Leaderboard Enrichment Algorithm

Pseudo-flow:

```python
def sync_arena_leaderboard():
    dataset = load_dataset("lmarena-ai/leaderboard-dataset", split="latest")
    for row in dataset:
        arena_model = normalize_arena_row(row)
        matches = catalog_store.find_alias_matches(arena_model.name)
        for match in matches:
            catalog_store.update_quality(
                match.id,
                arena_rank=arena_model.rank,
                arena_score=arena_model.score,
                arena_votes=arena_model.votes,
            )
```

Rules:

- Use fuzzy matching conservatively.
- Store unmatched Arena names for manual alias review.
- Do not create install candidates from Arena rows alone.

## 18. Security, Legal, and Safety Requirements

- Never auto-download gated models without explicit user approval.
- Never store API keys in `models_manifest.json`.
- Store secrets through environment variables or a local ignored settings file.
- Show license and commercial-use warnings before install.
- Pin Hugging Face revisions for approved installs where possible.
- Avoid executing arbitrary code from model repositories.
- For Hugging Face downloads, prefer file allow patterns over full snapshots.
- Keep source URLs visible in the UI.
- Maintain a reject/blocklist for suspicious repos.
- Do not claim a model is commercially usable unless the license is explicit.

## 19. Telemetry and Privacy

Default:

- No external telemetry.
- Catalog sync only contacts selected sources.
- Local battle votes stay local.

Optional later:

- Export local benchmark/battle results as JSON.

## 20. Success Metrics

MVP success:

- Finds at least 100 free downloadable text-model candidates from Hugging Face.
- Correctly filters to models supported by the user's hardware.
- Installs at least one discovered GGUF model end-to-end.
- Adds at least one free API model as an endpoint-backed model.
- No regressions to current static manifest installs.

Product success:

- User can go from "I need a free coding model" to installed/usable in under
  five minutes, excluding download time.
- Catalog results explain every recommendation.
- Gated/non-commercial/unknown-license models are never silently mixed with
  permissive free models.

## 21. Test Plan

Unit tests:

- License classifier.
- Free pricing classifier.
- Hugging Face file format detection.
- GGUF quant selection.
- Alias normalization.
- Deduplication.
- Ranking.
- Hardware fit enrichment.

Integration tests:

- Catalog sync with mocked Hugging Face responses.
- Catalog sync with mocked OpenRouter responses.
- Install from catalog row to `models/installed.json`.
- Existing `recommend` behavior remains compatible.
- Web API catalog filtering.

Manual QA:

- CPU-only host.
- Apple Silicon host.
- CUDA host.
- Gated model without token.
- Gated model with token.
- Free API model with missing key.
- Free API model with key.

## 22. Rollout Plan

Milestone 1: CLI-only discovery

- Add catalog store and Hugging Face sync.
- Add `catalog search`.
- Add tests.

Milestone 2: Install from discovered candidates

- Add approval/install command.
- Add allow-pattern downloads.
- Update installed-state writing.

Milestone 3: Web catalog

- Add FastAPI routes.
- Add Catalog UI.
- Add install progress.

Milestone 4: Free API providers

- Add OpenRouter sync.
- Add provider settings UI.
- Add pricing revalidation.

Milestone 5: Arena enrichment and local battle mode

- Add Arena leaderboard import.
- Add local model comparison.
- Add task router.

## 23. Open Questions

1. Should the first implementation prioritize only text models, or include image
   models in MVP?
2. Should non-commercial models appear by default with warnings, or only after
   enabling a filter?
3. Should API-backed free models be allowed in an app whose README currently
   emphasizes "Everything runs locally"?
4. Which trusted Hugging Face publishers should be in the initial allowlist?
5. Should the catalog database be resettable from the manager UI?

## 24. Recommended MVP

Build the smallest useful version:

1. Hugging Face text GGUF discovery.
2. Free/gated/license/trust classification.
3. Hardware-fit ranking using the existing recommendation logic.
4. CLI `catalog sync/search/install`.
5. Web `/api/catalog` read-only endpoint.
6. Then add UI install actions.

This is the best starting point because it turns the current static manifest
into a living model catalog while staying inside the runtimes your app already
supports.

## 25. Source References

- Arena: https://arena.ai/
- Arena help center: https://help.arena.ai/
- Arena leaderboard dataset on Hugging Face:
  https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset
- Arena API docs:
  https://portal.api.preview.arena.ai/docs/api-reference
- Hugging Face Hub Python search/list models:
  https://huggingface.co/docs/huggingface_hub/guides/search
- Hugging Face Hub API reference:
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
- OpenRouter model API:
  https://openrouter.ai/docs/api-reference/list-available-models
- Ollama API:
  https://docs.ollama.com/api
