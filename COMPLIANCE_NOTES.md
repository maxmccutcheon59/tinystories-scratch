# Compliance Notes — tinystories-scratch

**Status:** Research / educational ML code. Not a consumer product, SaaS, or data-collecting service.

## Data inventory

| Data | Collected by this repo? | Storage | Notes |
|------|-------------------------|---------|-------|
| End-user PII | **No** | — | No accounts, analytics, or telemetry |
| Synthetic micro-stories | Bundled constants in `data.py` | In source | Author-written; not TinyStories data |
| TinyStories texts (optional) | Downloaded on demand via HuggingFace `datasets` | Local `./data/hf_cache/` (gitignored) | See license below |
| Model checkpoints | Produced locally by `scripts/train.py` | Local `./checkpoints/` (gitignored) | Do not commit |

## Dataset license notes

- **roneneldan/TinyStories** (Hugging Face): metadata license **CDLA-Sharing-1.0**. Synthetic children’s stories generated with GPT-3.5/GPT-4, described in Eldan & Li (2023). This project may **download and train on** the dataset locally; it does **not** redistribute TinyStories files inside the git tree. Users should retain attribution and consult [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) for share-alike obligations if they redistribute derived datasets.
- **Synthetic stories in this repo:** original short strings for offline demos; MIT with the rest of the code.

Users must verify current terms for their jurisdiction and use case.

## Legal / regulatory checklist (flags)

| Item | Applies? | Action |
|------|----------|--------|
| Privacy Policy / ToS | No (no user data collection) | N/A |
| GDPR / US state privacy | No personal data processed | N/A — re-review if productized |
| COPPA / minors | Training text is child-*themed* fiction, not data about real children | No PII; still avoid deploying as a product for kids without review |
| Payments / PCI | No | N/A |
| HIPAA / health | No | N/A |
| AI disclosure | Research reimplementation | Cite Eldan & Li 2023; do not claim paper results |
| Export controls | Standard open-source ML | Human review if redistributing under sanction regimes |
| IP / third-party code | MIT original code + PyTorch (+ optional `datasets`) | Keep LICENSE; do not vendor TinyStories corpora |
| PII in training data | TinyStories is synthetic; no intentional PII collection | Do not add real user text without review |

## Security controls in this repo

- `.gitignore` excludes `.env*`, keys, checkpoints, datasets, HF caches
- CI template: `pytest`, `pip-audit`, `gitleaks`
- `SECURITY.md` with contact MaxMcCutcheon1@outlook.com
- No network server, auth, or PII handling
- Documented risk: `torch.load` of untrusted checkpoints

## Items needing human / lawyer review before productization

- If wrapping this as a public generative API: Privacy Policy, ToS, abuse monitoring, and model-output disclosure (especially child-facing UX).
- If redistributing TinyStories-derived corpora: CDLA-Sharing-1.0 share-alike / attribution review.
- If training on non-public or personal text: lawful basis, retention, and dataset licensing review.

## Authorization

Security scanning in CI targets **this** repository only (owned by maxmccutcheon59).

## CI workflow placement

The OAuth token used to publish this repo has scopes `gist`, `read:org`, `repo` but **not** `workflow`.
GitHub rejects both `git push` and Contents API writes to `.github/workflows/*` without that scope (HTTP 404).
The workflow YAML therefore ships at `ci/github-workflows/ci.yml` with enable instructions in `ci/README.md`.
This is a tooling limitation, not an intentional weakening of CI controls.
