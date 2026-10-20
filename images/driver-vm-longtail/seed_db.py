#!/usr/bin/env python3
"""Seed the long-tail demo database (baked at image build; replaced
honestly by a manifest mount at runtime when the owner brings data).

One table, a couple of rows — enough UI surface for record/replay."""
import sqlite3

conn = sqlite3.connect("/home/chrome/demo")
cur = conn.cursor()
cur.execute(
    "CREATE TABLE IF NOT EXISTS notes"
    " (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0)"
)
cur.execute("SELECT COUNT(*) FROM notes")
if cur.fetchone()[0] == 0:
    cur.executemany(
        "INSERT INTO notes (name, n) VALUES (?, ?)",
        [("first baked row", 1), ("second baked row", 2)],
    )
conn.commit()
conn.close()
print("demo db seeded: /home/chrome/demo (table: notes)")
