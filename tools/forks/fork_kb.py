#!/usr/bin/env python3
"""Incremental knowledge base of the commits and READMEs of every direct fork of Niko1221/Strata.

Every run lists all direct forks, then queries GitHub again only for forks whose `pushed_at` changed since the last
successful scan (new forks included).  Everything goes into one SQLite file, so the knowledge base can be queried
again at any time, and each run also writes a CSV with only the commits first seen in that run.

Layout under KB_DIR (default /home/dan/clone-of-d/strata-forks-kb):
  forks.sqlite                tables runs, forks, commits, readmes (see SCHEMA)
  readmes/<blob_sha>.md       README contents, stored once per distinct content (most forks share upstream's)
  runs/<run_id>-new-commits.csv
  runs/<run_id>-errors.csv

Usage:  GITHUB_TOKEN=$(gh auth token) python3 tools/forks/fork_kb.py
        FULL=1 ...   rescan every fork even if pushed_at did not change
        LIMIT=50 ... scan only the first N forks that need scanning (smoke test)
Example query (no sqlite3 CLI needed):
  python3 -c "import sqlite3; db=sqlite3.connect('KB_DIR/forks.sqlite'); [print(*r) for r in db.execute(
    'select repo, date, substr(message,1,80) from commits where first_seen_run=(select max(id) from runs)
     order by date desc')]"
"""
import base64
import csv
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

UPSTREAM = os.getenv("UPSTREAM", "Niko1221/Strata")
KB_DIR = Path(os.getenv("KB_DIR", "/home/dan/clone-of-d/strata-forks-kb"))
WORKERS = int(os.getenv("WORKERS", "5"))
FULL = os.getenv("FULL", "0") == "1"
LIMIT = int(os.getenv("LIMIT", "0"))

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    sys.exit("Setează GITHUB_TOKEN înainte de rulare, de ex.: GITHUB_TOKEN=$(gh auth token) python3 fork_kb.py")

BASE_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "strata-fork-kb",
}
PER_PAGE = 100

SCHEMA = """
create table if not exists runs (
  id integer primary key, started_at text, finished_at text, upstream_branch text,
  forks_total integer, forks_scanned integer, new_commits integer, errors integer);
create table if not exists forks (
  repo text primary key, owner text, branch text, archived integer, created_at text,
  pushed_at text,            -- as last listed
  scanned_pushed_at text,    -- pushed_at at the last successful scan; rescan when it differs
  compare_status text, ahead_by integer, behind_by integer,
  first_seen_run integer, last_scanned_run integer, last_error text);
create table if not exists commits (
  sha text, repo text, date text, author_login text, author_name text, message text, commit_url text,
  first_seen_run integer,
  last_seen_run integer,     -- last scan that still showed it ahead of upstream (older = force-pushed away or merged)
  upstream_history integer default 0,  -- 1 = old upstream history seen in a no-common-ancestor fork, not fork work
  primary key (repo, sha));
create table if not exists readmes (
  repo text primary key, blob_sha text, name text, html_url text, local_path text, updated_run integer);
create index if not exists commits_first_seen on commits(first_seen_run);
create index if not exists commits_date on commits(date);
"""

tls = threading.local()


def session():
    if not hasattr(tls, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        tls.session = s
    return tls.session


class NotFound(RuntimeError):
    pass


def get_json(url, params=None):
    for attempt in range(10):
        try:
            response = session().get(url, params=params, timeout=(15, 90))
        except requests.RequestException as exc:
            time.sleep(min(60, 2 ** attempt))
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            raise NotFound(f"404 on {url}: {response.text[:200]}")
        if response.status_code in (403, 429):
            retry_after = response.headers.get("Retry-After")
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            if retry_after:
                delay = int(retry_after) + 1
            elif remaining == "0" and reset:
                delay = max(1, int(reset) - int(time.time()) + 2)
            elif response.status_code == 403 and "rate limit" not in response.text.lower():
                raise RuntimeError(f"HTTP 403 for {url}: {response.text[:300]}")
            else:
                delay = min(60, 2 ** attempt)
            print(f"GitHub rate limit/throttle; sleeping {delay}s", file=sys.stderr)
            time.sleep(delay)
            continue
        if response.status_code >= 500:
            time.sleep(min(60, 2 ** attempt))
            continue
        raise RuntimeError(f"HTTP {response.status_code} for {url}: {response.text[:300]}")
    raise RuntimeError(f"Retries exhausted for {url}")


def get_all_direct_forks():
    forks, page = [], 1
    while True:
        payload = get_json(f"{BASE_URL}/repos/{UPSTREAM}/forks",
                           {"per_page": PER_PAGE, "page": page, "sort": "newest"})
        if not payload:
            break
        for repo in payload:
            forks.append({
                "repo": repo["full_name"],
                "owner": repo["owner"]["login"],
                "branch": repo.get("default_branch"),
                "archived": int(bool(repo.get("archived"))),
                "created_at": repo.get("created_at"),
                "pushed_at": repo.get("pushed_at"),
            })
        print(f"Fork page {page}: {len(payload)} repos", file=sys.stderr)
        if len(payload) < PER_PAGE:
            break
        page += 1
    return forks


def fetch_readme(repo):
    """Returns (meta, raw bytes) or (None, None) when the fork has no README."""
    try:
        data = get_json(f"{BASE_URL}/repos/{repo}/readme")
    except NotFound:
        return None, None
    content, encoding = data.get("content"), data.get("encoding")
    if content and encoding == "base64":
        raw = base64.b64decode(content)
    elif data.get("download_url"):
        resp = session().get(data["download_url"], timeout=(15, 90))
        resp.raise_for_status()
        raw = resp.content
    else:
        raise RuntimeError("README găsit, dar fără conținut detectabil")
    return {"blob_sha": data.get("sha"), "name": Path(data.get("name") or "README.md").name,
            "html_url": data.get("html_url", "")}, raw


def scan_fork(fork, upstream_branch):
    """Network only (no DB access): commits ahead of upstream plus the README."""
    if not fork["branch"]:
        raise RuntimeError("Forkul nu are branch implicit")
    compare_url = f"{BASE_URL}/repos/{UPSTREAM}/compare/{upstream_branch}...{fork['owner']}:{fork['branch']}"
    commits, page, info = [], 1, {}
    while True:
        try:
            payload = get_json(compare_url, {"per_page": PER_PAGE, "page": page})
        except NotFound as exc:
            if "No common ancestor" not in str(exc):
                raise
            # Upstream history was rewritten after this fork was made, so compare cannot work.  Fall back to the
            # commits on the fork's branch dated after the fork was created: work done in the fork itself.
            commits = commits_since_fork(fork)
            info = {"compare_status": "no-common-ancestor", "ahead_by": len(commits), "behind_by": None}
            break
        info = {"compare_status": payload.get("status"), "ahead_by": payload.get("ahead_by"),
                "behind_by": payload.get("behind_by")}
        batch = payload.get("commits", [])
        commits += [commit_row(item) for item in batch]
        if len(batch) < PER_PAGE:
            break
        page += 1
    readme_meta, readme_raw, readme_error = None, None, ""
    try:
        readme_meta, readme_raw = fetch_readme(fork["repo"])
    except Exception as exc:  # README failure must not lose the commits
        readme_error = f"README: {exc}"
    return info, commits, readme_meta, readme_raw, readme_error


def commit_row(item):
    commit = item.get("commit") or {}
    author = commit.get("author") or {}
    committer = commit.get("committer") or {}
    return {
        "sha": item.get("sha"),
        "date": committer.get("date") or author.get("date"),
        "author_login": (item.get("author") or {}).get("login"),
        "author_name": author.get("name"),
        "message": commit.get("message", ""),
        "commit_url": item.get("html_url"),
    }


def commits_since_fork(fork):
    if not fork["created_at"] or (fork["pushed_at"] or "") <= fork["created_at"]:
        return []  # nothing pushed to the fork since it was created
    rows, page = [], 1
    while True:
        batch = get_json(f"{BASE_URL}/repos/{fork['repo']}/commits",
                         {"sha": fork["branch"], "since": fork["created_at"], "per_page": PER_PAGE, "page": page})
        rows += [commit_row(item) for item in batch]
        if len(batch) < PER_PAGE:
            return rows
        page += 1


def main():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    (KB_DIR / "readmes").mkdir(exist_ok=True)
    (KB_DIR / "runs").mkdir(exist_ok=True)
    db = sqlite3.connect(KB_DIR / "forks.sqlite")
    db.executescript(SCHEMA)
    if "upstream_history" not in [r[1] for r in db.execute("pragma table_info(commits)")]:
        db.execute("alter table commits add column upstream_history integer default 0")

    started = datetime.now(timezone.utc)
    upstream_branch = get_json(f"{BASE_URL}/repos/{UPSTREAM}")["default_branch"]
    run_id = db.execute("insert into runs (started_at, upstream_branch) values (?, ?)",
                        (started.isoformat(timespec="seconds"), upstream_branch)).lastrowid
    db.commit()
    tag = f"{run_id:04d}-{started:%Y%m%d-%H%M}"

    forks = get_all_direct_forks()
    known = {r[0]: r[1] for r in db.execute("select repo, scanned_pushed_at from forks")}
    for f in forks:
        db.execute("""insert into forks (repo, owner, branch, archived, created_at, pushed_at, first_seen_run)
                      values (:repo, :owner, :branch, :archived, :created_at, :pushed_at, :run)
                      on conflict(repo) do update set owner=excluded.owner, branch=excluded.branch,
                        archived=excluded.archived, pushed_at=excluded.pushed_at""", {**f, "run": run_id})
    db.commit()

    todo = [f for f in forks if FULL or known.get(f["repo"]) != f["pushed_at"]]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"Upstream branch: {upstream_branch}; forks: {len(forks)}; to scan: {len(todo)}", file=sys.stderr)

    new_rows, errors = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan_fork, f, upstream_branch): f for f in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            fork = futures[future]
            repo = fork["repo"]
            try:
                info, commits, readme_meta, readme_raw, readme_error = future.result()
            except Exception as exc:
                errors.append({"repo": repo, "error": str(exc)})
                db.execute("update forks set last_error=? where repo=?", (str(exc), repo))
                db.commit()
                print(f"[{index}/{len(todo)}] ERROR {repo}: {exc}", file=sys.stderr)
                continue
            fresh = 0
            for c in commits:
                cur = db.execute("""insert into commits (sha, repo, date, author_login, author_name, message,
                                      commit_url, first_seen_run, last_seen_run)
                                    values (:sha, :repo, :date, :author_login, :author_name, :message,
                                      :commit_url, :run, :run)
                                    on conflict(repo, sha) do update set last_seen_run=excluded.last_seen_run
                                    returning first_seen_run""", {**c, "repo": repo, "run": run_id})
                if cur.fetchone()[0] == run_id:
                    fresh += 1
                    new_rows.append({"repo": repo, "repo_url": f"https://github.com/{repo}",
                                     "fork_branch": fork["branch"], **info, **c})
            if readme_meta:
                path = KB_DIR / "readmes" / f"{readme_meta['blob_sha']}.md"
                if not path.exists():
                    path.write_bytes(readme_raw)
                db.execute("""insert into readmes (repo, blob_sha, name, html_url, local_path, updated_run)
                              values (?, ?, ?, ?, ?, ?)
                              on conflict(repo) do update set name=excluded.name, html_url=excluded.html_url,
                                local_path=excluded.local_path,
                                updated_run=case when readmes.blob_sha = excluded.blob_sha
                                                 then readmes.updated_run else excluded.updated_run end,
                                blob_sha=excluded.blob_sha""",
                           (repo, readme_meta["blob_sha"], readme_meta["name"], readme_meta["html_url"],
                            str(path), run_id))
            if readme_error:
                errors.append({"repo": repo, "error": readme_error})
            db.execute("""update forks set scanned_pushed_at=?, compare_status=?, ahead_by=?, behind_by=?,
                          last_scanned_run=?, last_error=? where repo=?""",
                       (fork["pushed_at"], info["compare_status"], info["ahead_by"], info["behind_by"],
                        run_id, readme_error or None, repo))
            db.commit()
            print(f"[{index}/{len(todo)}] {repo}: {len(commits)} ahead, {fresh} new", file=sys.stderr)

    # Forks made before upstream rewrote its history have no common ancestor with it, so their commits come from the
    # date fallback and include the old upstream commits they synced.  Those are shared by many such forks (or are
    # the upstream owner's own); mark them so they do not count as fork work.  Recomputed every run.
    owner = UPSTREAM.split("/")[0]
    db.execute("""update commits set upstream_history = (author_login is ? or sha in (
                    select c.sha from commits c join forks f on f.repo = c.repo
                    where f.compare_status = 'no-common-ancestor' group by c.sha having count(distinct c.repo) >= 2))
                  where repo in (select repo from forks where compare_status = 'no-common-ancestor')""", (owner,))
    db.execute("""update forks set ahead_by = (select count(*) from commits c where c.repo = forks.repo
                    and c.upstream_history = 0 and c.last_seen_run = forks.last_scanned_run)
                  where compare_status = 'no-common-ancestor'""")
    db.commit()
    history = {(r[0], r[1]) for r in db.execute("select repo, sha from commits where upstream_history = 1")}
    new_rows = [r for r in new_rows if (r["repo"], r["sha"]) not in history]

    fields = ["repo", "repo_url", "fork_branch", "compare_status", "ahead_by", "behind_by",
              "date", "sha", "commit_url", "author_login", "author_name", "message"]
    new_csv = KB_DIR / "runs" / f"{tag}-new-commits.csv"
    with open(new_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(new_rows, key=lambda r: r["date"] or "", reverse=True))
    err_csv = KB_DIR / "runs" / f"{tag}-errors.csv"
    with open(err_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["repo", "error"])
        w.writeheader()
        w.writerows(errors)

    db.execute("""update runs set finished_at=?, forks_total=?, forks_scanned=?, new_commits=?, errors=?
                  where id=?""", (datetime.now(timezone.utc).isoformat(timespec="seconds"), len(forks),
                                  len(todo), len(new_rows), len(errors), run_id))
    db.commit()
    db.close()
    print(f"\nRun {run_id}: {len(todo)} forks scanned, {len(new_rows)} new commits, {len(errors)} errors")
    print(f"Knowledge base: {KB_DIR / 'forks.sqlite'}")
    print(f"New commits:    {new_csv}")
    print(f"Errors:         {err_csv}")


if __name__ == "__main__":
    main()
