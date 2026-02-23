#!/usr/bin/env python3
"""
linear_sync.py — Bootstrap a Linear project for Cognitive Bias Codex.

Usage:
    LINEAR_API_KEY=lin_api_... python linear_sync.py

Get your API key at: https://linear.app/settings/api
"""

import json
import os
import sys
import httpx

LINEAR_API = "https://api.linear.app/graphql"


def gql(query: str, variables: dict = None, token: str = "") -> dict:
    resp = httpx.post(
        LINEAR_API,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": token, "Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


# ── Issue definitions ────────────────────────────────────────────────────────
# (title, description, priority 0=no 1=urgent 2=high 3=medium 4=low, label)
ISSUES = [
    # UX / Design
    (
        "First-run onboarding flow",
        "Show a welcome overlay on first visit explaining the 3 core features (Library, Scanner, HF connect). Store dismissal in localStorage so it only shows once.",
        2, "UX",
    ),
    (
        "Bias of the day — featured card in Library",
        "Show a highlighted random concept at the top of the Library tab each session. Optionally cache to localStorage per day.",
        3, "UX",
    ),
    (
        "Scan tips — show examples of what to point camera at",
        "Add a horizontal scrollable chip row in the Scan tab showing: Ads, Social posts, Product packaging, News headlines, Menus. Tapping a chip could pre-fill a text prompt.",
        3, "UX",
    ),
    (
        "Inline AI-connect prompt when no backend is configured",
        "If no AI backend is active, show a clear amber banner inside the Scan tab (not buried in Settings) with a direct 'Connect Hugging Face — Free' CTA.",
        2, "UX",
    ),
    (
        "Simplify Settings tab language",
        "Remove technical jargon (ANTHROPIC_API_KEY, server-side, etc.) from the user-facing Settings tab. Replace with plain-language explanations of what each tier provides.",
        3, "UX",
    ),
    (
        "Share scan results",
        "Add a Share button to scan results that generates a shareable card/URL with the detected biases and scene description.",
        3, "Feature",
    ),
    (
        "Scan history — local-only log of past scans",
        "Store the last 20 scan results in localStorage. Add a History view in the Settings or Library tab to review past detections.",
        4, "Feature",
    ),
    # Features
    (
        "Concept detail page / deep-dive modal",
        "Tapping a concept card opens a full modal with: description, Wikipedia summary, related concepts, and external link. Currently the card just shows truncated text.",
        2, "Feature",
    ),
    (
        "Related concepts — show similar biases after scan",
        "After a scan, below the detected biases, show 2-3 related concepts from the library that are conceptually similar.",
        3, "Feature",
    ),
    (
        "Export library to CSV / PDF",
        "Allow users to download the full dataset (or current search results) as CSV or a formatted PDF reference sheet.",
        4, "Feature",
    ),
    (
        "Push notifications — bias of the day (PWA)",
        "Implement a PWA service worker and notification permission flow so users can opt in to a daily bias/fallacy notification.",
        4, "Feature",
    ),
    (
        "Add more mental models (target: 100)",
        "Current dataset has 20 mental models. Research and add 80 more well-known frameworks (First Principles, Inversion, Circle of Competence, etc.).",
        3, "Content",
    ),
    # AI / Backend
    (
        "Upgrade scanner to real image vision via HF multimodal models",
        "Replace the text-only HF fallback with a multimodal model (e.g. Idefics, LLaVA via HF Inference) so the HF backend actually processes the camera image.",
        1, "AI",
    ),
    (
        "Improve JSON parsing robustness from LLM responses",
        "LLMs sometimes return malformed JSON. Add a retry + JSON-repair fallback so a single bad response doesn't kill the whole scan.",
        2, "AI",
    ),
    (
        "Streaming scan results",
        "Stream the AI response token-by-token so users see partial results appearing instead of a blank spinner for 10-45 seconds.",
        3, "AI",
    ),
    # DevOps / Infrastructure
    (
        "Set up CI — lint + test on every push",
        "Add a GitHub Actions workflow that runs: ruff (lint), mypy (types), pytest (unit tests) on every PR and push to main.",
        2, "DevOps",
    ),
    (
        "Add unit tests for core search and AI backend logic",
        "Write pytest tests for: _load_data(), search_concepts(), _match_concept(), _strip_json(), validate_hf_token(). Target 80% coverage.",
        2, "DevOps",
    ),
    (
        "Docker image size optimization",
        "Current image installs all Python deps including large packages. Multi-stage build or slim base could reduce image size significantly.",
        4, "DevOps",
    ),
    (
        "Add Linear sync to project workflow",
        "Use linear_sync.py to keep the Linear backlog in sync with development priorities. Run on new feature additions.",
        3, "DevOps",
    ),
    # Integration
    (
        "Linear project integration",
        "Connect the GitHub repo to this Linear project. Enable auto-close of issues when PRs are merged using branch naming conventions.",
        3, "Integration",
    ),
]


def main():
    token = os.getenv("LINEAR_API_KEY", "").strip()
    if not token:
        print("ERROR: Set LINEAR_API_KEY environment variable.")
        print("Get yours at: https://linear.app/settings/api")
        sys.exit(1)

    print("Connecting to Linear...")

    # 1. Get viewer + teams
    me = gql("{ viewer { id name } teams { nodes { id name } } }", token=token)
    viewer = me["viewer"]
    teams = me["teams"]["nodes"]
    if not teams:
        print("No teams found. Create a team in Linear first.")
        sys.exit(1)

    team = teams[0]
    print(f"Logged in as: {viewer['name']}")
    print(f"Using team: {team['name']} ({team['id']})")

    # 2. Create or find project
    projects = gql(
        """query($teamId: String!) {
          team(id: $teamId) { projects { nodes { id name } } }
        }""",
        {"teamId": team["id"]},
        token=token,
    )
    existing = projects["team"]["projects"]["nodes"]
    project = next((p for p in existing if "Cognitive Bias" in p["name"]), None)

    if not project:
        print("Creating Linear project: Cognitive Bias Codex...")
        result = gql(
            """mutation($name: String!, $teamIds: [String!]!) {
              projectCreate(input: { name: $name, teamIds: $teamIds }) {
                project { id name }
              }
            }""",
            {"name": "Cognitive Bias Codex", "teamIds": [team["id"]]},
            token=token,
        )
        project = result["projectCreate"]["project"]
        print(f"  Created project: {project['name']}")
    else:
        print(f"Found existing project: {project['name']}")

    # 3. Get workflow states
    states = gql(
        """query($teamId: String!) {
          team(id: $teamId) { states { nodes { id name type } } }
        }""",
        {"teamId": team["id"]},
        token=token,
    )["team"]["states"]["nodes"]
    backlog_state = next(
        (s for s in states if s["type"] in ("triage", "backlog") or "backlog" in s["name"].lower()),
        states[0] if states else None,
    )

    # 4. Get/create labels
    label_data = gql(
        """query($teamId: String!) {
          team(id: $teamId) { labels { nodes { id name } } }
        }""",
        {"teamId": team["id"]},
        token=token,
    )["team"]["labels"]["nodes"]
    label_map = {l["name"]: l["id"] for l in label_data}

    label_colors = {
        "UX": "#6ea0ff",
        "Feature": "#10b981",
        "Content": "#f59e0b",
        "AI": "#c084fc",
        "DevOps": "#94a3b8",
        "Integration": "#f97316",
    }

    for name, color in label_colors.items():
        if name not in label_map:
            result = gql(
                """mutation($teamId: String!, $name: String!, $color: String!) {
                  issueLabelCreate(input: { teamId: $teamId, name: $name, color: $color }) {
                    issueLabel { id name }
                  }
                }""",
                {"teamId": team["id"], "name": name, "color": color},
                token=token,
            )
            label_map[name] = result["issueLabelCreate"]["issueLabel"]["id"]
            print(f"  Created label: {name}")

    # 5. Create issues
    print(f"\nCreating {len(ISSUES)} issues...")
    created = 0
    for title, description, priority, label in ISSUES:
        variables = {
            "teamId": team["id"],
            "title": title,
            "description": description,
            "priority": priority,
            "projectId": project["id"],
            "labelIds": [label_map[label]] if label in label_map else [],
        }
        if backlog_state:
            variables["stateId"] = backlog_state["id"]

        gql(
            """mutation(
              $teamId: String!, $title: String!, $description: String,
              $priority: Int, $projectId: String, $labelIds: [String!], $stateId: String
            ) {
              issueCreate(input: {
                teamId: $teamId, title: $title, description: $description,
                priority: $priority, projectId: $projectId, labelIds: $labelIds,
                stateId: $stateId
              }) {
                issue { id identifier title }
              }
            }""",
            variables,
            token=token,
        )
        print(f"  [{label}] {title}")
        created += 1

    print(f"\nDone. {created} issues created in '{project['name']}'.")
    print(f"View your project: https://linear.app")


if __name__ == "__main__":
    main()
