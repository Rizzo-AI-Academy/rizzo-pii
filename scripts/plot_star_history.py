"""Star-history chart, generated locally.

star-history.com no longer works for this repo: GitHub restricted the stargazer API,
and restoring the service would mean installing a third-party GitHub App on the org.
For a privacy project that trade is not worth a decorative chart, so we read the
stargazers ourselves with a personal token and commit the rendered PNG.

Writes two files (the README picks one via prefers-color-scheme):
    docs/star_history_light.png
    docs/star_history_dark.png

Usage:
    python scripts/plot_star_history.py                 # repo taken from git remote
    python scripts/plot_star_history.py --repo owner/name

The token comes from GH_TOKEN / GITHUB_TOKEN, in the environment or in .env.
It only needs public read access: stargazers of a public repo are public data.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs"
API = "https://api.github.com"
PER_PAGE = 100

# light theme, then dark. bg=None keeps the PNG transparent-free but flat;
# GitHub renders the README on #ffffff / #0d1117, so we match those.
THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", grid="#d8dee4", line="#2f7d4f", fill="#2f7d4f22"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", line="#4ade80", fill="#4ade8022"),
}


def load_token() -> str:
    """GH_TOKEN / GITHUB_TOKEN from the environment, falling back to .env."""
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key].strip()

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(GH_TOKEN|GITHUB_TOKEN)\s*=\s*(.+)\s*$", line)
            if m:
                return m.group(2).strip().strip("'\"")

    sys.exit(
        "No GitHub token found. Set GH_TOKEN in the environment or in .env "
        "(a token with public read access is enough)."
    )


def default_repo() -> str:
    """owner/name from the git remote, so the script needs no argument."""
    try:
        url = subprocess.run(
            ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("Cannot read the git remote — pass --repo owner/name.")

    m = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?$", url)
    if not m:
        sys.exit(f"Remote is not a GitHub URL: {url} — pass --repo owner/name.")
    return f"{m['owner']}/{m['name']}"


def get(url: str, token: str) -> tuple[object, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={
            # star+json is what turns each stargazer into {starred_at, user}
            "Accept": "application/vnd.github.star+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rizzo-pii-star-history",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        sys.exit(f"GitHub API {exc.code} on {url}\n{body}")


def fetch_stars(repo: str, token: str) -> list[datetime]:
    """Every star timestamp, oldest first."""
    stars: list[datetime] = []
    page = 1
    while True:
        data, headers = get(f"{API}/repos/{repo}/stargazers?per_page={PER_PAGE}&page={page}", token)
        if not isinstance(data, list):
            sys.exit(f"Unexpected response for {repo}: {data}")
        for item in data:
            when = item.get("starred_at")
            if when:
                stars.append(datetime.strptime(when, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
        print(f"  page {page}: {len(data)} stargazers (total {len(stars)})")
        # GitHub caps pagination at 400 pages; the Link header is the real stop signal
        if len(data) < PER_PAGE or 'rel="next"' not in headers.get("Link", ""):
            break
        page += 1
    stars.sort()
    return stars


def render(stars: list[datetime], repo: str, theme: str, out: Path) -> None:
    c = THEMES[theme]
    xs = list(stars)
    ys = list(range(1, len(stars) + 1))

    # extend the line to today, otherwise a quiet month looks like the chart ended
    now = datetime.now(timezone.utc)
    if xs and now > xs[-1]:
        xs.append(now)
        ys.append(ys[-1])

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])

    ax.plot(xs, ys, color=c["line"], linewidth=2.2, solid_capstyle="round")
    ax.fill_between(xs, ys, color=c["fill"])
    if stars:
        ax.plot([stars[-1]], [len(stars)], "o", color=c["line"], markersize=5)

    ax.set_title(f"{repo} — stars over time", color=c["fg"], fontsize=12, pad=12)
    ax.set_ylabel("stars", color=c["fg"], fontsize=10)
    ax.grid(True, color=c["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=c["fg"], labelsize=9)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(c["grid"])
    ax.set_ylim(bottom=0)
    # a young repo spans weeks, an old one years: a fixed "%b %Y" would print the
    # same month four times. Let matplotlib pick the tick unit, then label it.
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.text(
        0.99, 0.02, f"generated locally · {now:%Y-%m-%d}",
        transform=ax.transAxes, ha="right", va="bottom",
        color=c["fg"], alpha=0.45, fontsize=7,
    )

    fig.tight_layout()
    # write via a buffer so a crash mid-render cannot leave a truncated PNG committed
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=c["bg"])
    plt.close(fig)
    out.write_bytes(buf.getvalue())
    print(f"  wrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="owner/name (default: the origin remote)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    repo = args.repo or default_repo()
    token = load_token()

    print(f"Fetching stargazers of {repo}…")
    stars = fetch_stars(repo, token)
    if not stars:
        sys.exit(f"{repo} has no stars yet — nothing to plot.")
    print(f"{len(stars)} stars, from {stars[0]:%Y-%m-%d} to {stars[-1]:%Y-%m-%d}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        render(stars, repo, theme, args.out_dir / f"star_history_{theme}.png")


if __name__ == "__main__":
    main()
