"""End-to-end check of the web app: every subject, both downloads, seeds and bad input.

Run from the project folder:   python tests/smoke_test.py
"""
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as A  # noqa: E402

client = A.app.test_client()
problems = []


def expect(ok, message):
    if not ok:
        problems.append(message)
        print("FAIL", message)


today = date.today().isoformat()
for key, recipe in sorted(A.RECIPES.items()):
    for cols in (1, 6, 99):
        url = f"/?theme={key}&rows=200&cols={cols}&seed=42&d={today}"
        r = client.get(url)
        expect(r.status_code == 200 and b"What the columns mean" in r.data, f"page {url} -> {r.status_code}")
    args = f"theme={key}&rows=200&cols=8&seed=42&d={today}"
    csv, xlsx = client.get(f"/download/csv?{args}"), client.get(f"/download/xlsx?{args}")
    expect(csv.status_code == 200 and xlsx.status_code == 200, f"download status {key}")
    if csv.status_code == 200 and xlsx.status_code == 200:
        expect(len(pd.read_csv(io.BytesIO(csv.data))) == 200, f"csv rows {key}")
        expect(len(pd.read_excel(io.BytesIO(xlsx.data))) == 200, f"xlsx rows {key}")
        expect("seed42" in csv.headers["Content-Disposition"], f"filename {key}")
        expect(csv.data == client.get(f"/download/csv?{args}").data, f"csv reproducible {key}")

# a table with no seed is redirected to an address that fixes seed and date
r = client.get("/?theme=retail_sales&rows=5&cols=4")
expect(r.status_code == 302 and "seed=" in r.headers["Location"] and "d=" in r.headers["Location"], "seed redirect")
a = client.get(f"/?theme=retail_sales&rows=20&cols=6&seed=7&d={today}").data
b = client.get(f"/?theme=retail_sales&rows=20&cols=6&seed=7&d={today}").data
c = client.get(f"/?theme=retail_sales&rows=20&cols=6&seed=8&d={today}").data
expect(a == b, "same seed identical")
expect(a != c, "different seed differs")

# bad input never crashes
for url in ["/?theme=nope", "/?theme=retail_sales&rows=abc&cols=&seed=x&d=junk",
            "/?theme=retail_sales&rows=99999&cols=99999&seed=-5&d=1999-01-01", "/download/csv?theme=retail_sales",
            "/download/pdf?theme=retail_sales&seed=1", "/surprise", "/healthz", "/"]:
    r = client.get(url, follow_redirects=True)
    expect(r.status_code in (200, 400, 404), f"bad input {url} -> {r.status_code}")
expect(client.get("/download/csv?theme=retail_sales").status_code == 400, "download without seed is 400")
expect(client.get("/download/pdf?theme=retail_sales&seed=1").status_code == 404, "unknown format is 404")

# ---- messy data ----
for key, recipe in sorted(A.RECIPES.items()):
    base = f"theme={key}&rows=50&cols=6&seed=42&d={today}"
    page = client.get(f"/?{base}&messy=1")
    expect(page.status_code == 200 and b"problems planted" in page.data, f"messy page {key}")
    clean_page = client.get(f"/?{base}")
    expect(b"problems planted" not in clean_page.data, f"clean page {key} shows messy note")
    a = client.get(f"/download/csv?{base}&messy=1")
    b = client.get(f"/download/csv?{base}&messy=1")
    c = client.get(f"/download/csv?{base}")
    expect(a.status_code == 200 and a.data == b.data and a.data != c.data, f"messy csv {key}")
    expect("_messy.csv" in a.headers.get("Content-Disposition", ""), f"messy filename {key}")
    k = client.get(f"/download/key?{base}")
    expect(k.status_code == 200 and k.data.startswith(b"row,column,problem"), f"answer key {key}")
    x = client.get(f"/download/xlsx?{base}&messy=1")
    expect(x.status_code == 200 and x.data[:2] == b"PK", f"messy xlsx {key}")

print("max columns:", A.MAX_COLS, "| problems:", len(problems))
sys.exit(1 if problems else 0)
