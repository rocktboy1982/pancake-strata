"""plan_page.py - build plan.html (the filterable plan board) from MOD_PLAN.md and TEST_LEDGER.md.

Run after every change to MOD_PLAN.md or TEST_LEDGER.md and commit plan.html in the same commit:
    python3 tools/pancake/plan_page.py
Template: tools/pancake/plan_page_template.html. Output: plan.html at the repository root (AGENTS.md documentation
rules allow it as the one generated view). Stdlib only; the page loads marked 12.0.2 from cdnjs to render Markdown.

Plan item status comes from the plan section an item sits in (section 3 done, section 4 decisions by id, section 5
queued with item 1 and the night queue in progress, section 6 code to write unless the row starts with CODED / PORTED /
DONE (queued) or PARTLY CODED (in progress), 7 fix needed, 8 parked, 9 dropped). Test verdicts come from
TEST_LEDGER.md rows as written.
"""
import datetime
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(HERE))
SEP = re.compile(r'(?<!\\)\|')
DECISION_STATUS = {'D1': 'done', 'D2': 'done', 'D9': 'notstarted', 'D10': 'progress'}


def cells(line):
    return [c.strip() for c in SEP.split(line.strip())[1:-1]]


def sections(md):
    out, cur, buf = {}, None, []
    for line in md.split('\n'):
        m = re.match(r'^## (\d+)\. (.*)', line)
        if m:
            if cur:
                out[cur] = buf
            cur, buf = int(m.group(1)), []
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = buf
    return out


def table_rows(lines):
    rows, hdr = [], False
    for i, line in enumerate(lines):
        if line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|\s*:?-', lines[i + 1]):
            hdr = True
            continue
        if line.startswith('|') and hdr and not re.match(r'^\|\s*:?-', line):
            rows.append(cells(line))
        elif not line.startswith('|'):
            hdr = False
    return rows


def numbered(lines):
    items, cur = [], None
    for line in lines:
        m = re.match(r'^(\d+)\. (.*)', line)
        if m:
            if cur:
                items.append(cur)
            cur = [int(m.group(1)), m.group(2)]
        elif cur and line.startswith('  ') and line.strip():
            cur[1] += ' ' + line.strip()
        elif cur and not line.strip():
            items.append(cur)
            cur = None
    if cur:
        items.append(cur)
    return items


def bullets(lines, stop=None):
    items, cur = [], None
    for line in lines:
        if stop and stop(line):
            break
        if line.startswith('- '):
            if cur:
                items.append(cur)
            cur = line[2:]
        elif cur is not None and line.startswith('  ') and line.strip():
            cur += ' ' + line.strip()
        elif cur is not None and not line.strip():
            items.append(cur)
            cur = None
    if cur:
        items.append(cur)
    return items


def defect_status(b):
    """MOD_PLAN section 7 bullet -> status: its own marker (FIXED / RESOLVED / CLOSED / CODED, a code read or a
    measurement at the start) is done; an explicit OPEN, an inconclusive try or a cosmetic gap still needs a fix."""
    head = b[:80]
    if re.search(r'\b(FIXED|RESOLVED|CLOSED|CODED)\b', head) or 'code read' in head or re.search(r'measured 20\d\d', head):
        return 'done'
    if re.search(r'\bOPEN\b', b) or 'TRIED' in b or 'cosmetic' in b:
        return 'fix'
    if 'CLOSED' in b:
        return 'done'
    if 'not applicable' in b:
        return 'parked'
    return 'fix'


def short(s, n=120):
    return s[:n] + ('...' if len(s) > n else '')


def plan_items(plan):
    S = sections(plan)
    items = []

    def add(sec, status, title, body, ref):
        items.append({'sec': sec, 'status': status, 'title': title, 'body': body, 'ref': ref})

    for r in table_rows(S.get(3, [])):
        if len(r) >= 2:
            add('Daily preset', 'done', r[0], ' | '.join(r[1:]), 'MOD_PLAN 3')
    for r in table_rows(S.get(4, [])):
        if r and re.match(r'^D\d+$', r[0]):
            body = (r[2] if len(r) > 2 else '') + ('<br><b>Next:</b> ' + r[3] if len(r) > 3 else '')
            add('Decisions', DECISION_STATUS.get(r[0], 'waiting'), r[0] + ' ' + r[1], body, 'MOD_PLAN 4')
    for n, txt in numbered(S.get(5, [])):
        m = re.match(r'\*\*(.+?)\*\*[: ]*(.*)', txt)
        title, body = (m.group(1), m.group(2)) if m else (short(txt, 80), txt)
        add('GPU queue', 'progress' if n == 1 or 'NIGHT QUEUE' in title else 'queued', '%d. %s' % (n, title), body, 'MOD_PLAN 5')
    for r in table_rows(S.get(6, [])):
        if r and re.match(r'^\d+$', r[0]):
            body = r[1] + '<br><b>Expected:</b> ' + (r[2] if len(r) > 2 else '') + (' (' + r[3] + ')' if len(r) > 3 else '')
            # a row's own marker wins: "PARTLY CODED" in progress, "CODED" / "PORTED" / "DONE" queued for measurement
            # "DROPPED" (measured and dropped, upper case) wins over the coded markers
            st = ('dropped' if 'DROPPED' in r[1] else
                  'progress' if r[1].startswith(('PARTLY CODED', 'IN PROGRESS')) else
                  'queued' if re.match(r'^(CODED|PORTED|DONE)[ :]', r[1]) or 'PORTED 2026' in r[1][:200] else 'notstarted')
            add('Code to write', st, '#%s %s' % (r[0], short(r[1], 110)), body, 'MOD_PLAN 6')
    for b in bullets(S.get(7, []), stop=lambda l: l.startswith('Plan claims corrected')):
        add('Defects', defect_status(b), short(b), b, 'MOD_PLAN 7')
    for b in bullets(S.get(8, [])):
        add('Parked', 'parked', short(b), b, 'MOD_PLAN 8')
    for b in bullets(S.get(9, [])):
        add('Dropped (do not repeat)', 'dropped', b.split(':')[0], b, 'MOD_PLAN 9')
    return items


def ledger_rows(ledger):
    tests, area = [], 'other'
    for line in ledger.split('\n'):
        m = re.match(r'^## (.+?)(?: \(\d+\))?$', line)
        if m:
            area = m.group(1) if not m.group(1).lower().startswith('added') else 'added 2026-11-15'
            continue
        if line.startswith('| T-'):
            c = cells(line)
            if len(c) >= 10:
                tests.append({'id': c[0], 'date': c[1], 'item': c[2], 'test': c[3], 'hw': c[4], 'tier': c[5],
                              'result': c[6], 'exact': c[7], 'verdict': c[8], 'archive': c[9], 'area': area})
    return tests


def main():
    plan = open(os.path.join(REPO, 'MOD_PLAN.md'), encoding='utf-8').read()
    ledger = open(os.path.join(REPO, 'TEST_LEDGER.md'), encoding='utf-8').read()
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short=9', 'HEAD'], capture_output=True, text=True).stdout.strip()
    data = {'items': plan_items(plan), 'tests': ledger_rows(ledger), 'commit': commit,
            'date': datetime.date.today().isoformat()}
    page = open(os.path.join(HERE, 'plan_page_template.html'), encoding='utf-8').read()
    page = (page.replace('__DATA__', json.dumps(data, ensure_ascii=True).replace('</', '<\\/'))
                .replace('__MARKDOWN__', plan.replace('</script', '<\\/script'))
                .replace('__COMMIT__', commit).replace('__DATE__', data['date']))
    open(os.path.join(REPO, 'plan.html'), 'w', encoding='utf-8', newline='\n').write(page)
    print('plan.html: %d plan items %s, %d tests' % (len(data['items']), dict(Counter(i['status'] for i in data['items'])), len(data['tests'])))


if __name__ == '__main__':
    main()
