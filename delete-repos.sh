#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
OWNER="talismanic"          # <- change me
REPOS=( "travel-planner-app" "OpenOmni_mark" "GUI-Agents-Evaluation-Suite-Collection-Test" "MCP-Universe-Research-0030" "ModelHub-X" "comment-auto-bot-28" "comment-auto-bot" "stanford_alpaca" "BLIP" "salesforce-blip-issues" "huggingface-diffusers-issues" "MCP-Universe-Research" "ModelHub" "auto-comment-bot2" "auto-comment-bot-x" "auto-issue-close" "ci-extensive-challenge" "Kimi-VL" "DeepSeek-VL2" "microsoft-air-no-label-issues" "microsoft-air-car-issues" "qwen-agent-close-wip-issues" "google-generative-ai-issues" "facebook-react-issues" "BigCodeLLM-FT-Proj" "ai-code-reviewer" "video-llm-evaluation-harness" "llm-training-toolkit" "Aria" "Aria-UI" "LVB-Dev" "Resource_Papers_Dev" "Qwen3-VL" "ScreenSpot-Pro-Performance-Dev" "openomni_emnlp2024_best_demo_mark" "WizardLM" "LAVIS")       # <- change me
DRY_RUN=false                          # set to false to actually delete

# Auth: export a token with permission to delete repos
# For classic PAT: needs "delete_repo" scope
# For fine-grained PAT: enable "Administration" (delete repo) on the target repos
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN env var first}"

confirm() {
  read -r -p "About to delete ${#REPOS[@]} repos under '$OWNER'. Continue? [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy]$ ]]
}

if ! $DRY_RUN; then
  confirm || { echo "Aborted."; exit 1; }
else
  echo "[DRY RUN] No repositories will be deleted."
fi

for repo in "${REPOS[@]}"; do
  url="https://api.github.com/repos/${OWNER}/${repo}"
  if $DRY_RUN; then
    echo "[DRY RUN] Would DELETE $url"
    continue
  fi

  http_code="$(
    curl -sS -o /dev/null -w "%{http_code}" \
      -X DELETE \
      -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "$url"
  )"

  case "$http_code" in
    204) echo "✅ Deleted: ${OWNER}/${repo}";;
    404) echo "❓ Not found or no access: ${OWNER}/${repo}";;
    403) echo "⛔ Forbidden (check token scopes/permissions): ${OWNER}/${repo}";;
    *)   echo "⚠️ Unexpected HTTP $http_code for ${OWNER}/${repo}";;
  esac
done
