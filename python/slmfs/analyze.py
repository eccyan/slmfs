"""SLMFS Analyze — statistical dashboard of the Poincaré disk state.

Connects to the SQLite checkpoint database in read-only mode and prints
brain-wave metrics: node population, spatial distribution, Fisher-Rao
certainty, cohomology frictions, and an ASCII disk map.
"""

import math
import sqlite3
import sys
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open SQLite in read-only mode."""
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _print_header(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def _ascii_disk(nodes: list[tuple[float, float]], size: int = 21):
    """Render a size x size ASCII scatter plot of the Poincaré disk."""
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    center = size // 2
    radius = center

    # Draw disk boundary
    for i in range(size):
        for j in range(size):
            dx = (j - center) / radius
            dy = (i - center) / radius
            r = math.sqrt(dx * dx + dy * dy)
            if abs(r - 1.0) < 0.08:
                grid[i][j] = '.'

    # Plot nodes
    for px, py in nodes:
        col = int(center + px * radius)
        row = int(center - py * radius)  # flip y
        if 0 <= row < size and 0 <= col < size:
            grid[row][col] = '*'

    # Mark center
    grid[center][center] = '+'

    print()
    for row in grid:
        print("    " + ''.join(row))
    print()


def main():
    db_path = Path.home() / ".slmfs" / "memory.db"

    # Allow override via --db-path=...
    for arg in sys.argv[1:]:
        if arg.startswith("--db-path="):
            db_path = Path(arg.split("=", 1)[1])

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Is the engine running and has it checkpointed at least once?")
        sys.exit(1)

    conn = _connect(db_path)
    cur = conn.cursor()

    # --- Node Population ---
    _print_header("Node Population")

    cur.execute("SELECT COUNT(*) FROM memory_nodes WHERE status = 0")
    active_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM memory_nodes WHERE status != 0")
    archived_count = cur.fetchone()[0]
    total = active_count + archived_count

    print(f"  Total nodes:    {total}")
    print(f"  Active:         {active_count}")
    print(f"  Archived:       {archived_count}")

    # --- Spatial Distribution ---
    _print_header("Spatial Distribution (Langevin Drift)")

    cur.execute("SELECT pos_x, pos_y FROM memory_nodes WHERE status = 0")
    active_nodes = cur.fetchall()

    working = 0   # r < 0.3
    drifting = 0  # 0.3 <= r < 0.8
    nearing = 0   # 0.8 <= r < 0.95
    beyond = 0    # r >= 0.95

    radii = []
    for px, py in active_nodes:
        r = math.sqrt(px * px + py * py)
        radii.append(r)
        if r < 0.3:
            working += 1
        elif r < 0.8:
            drifting += 1
        elif r < 0.95:
            nearing += 1
        else:
            beyond += 1

    if radii:
        avg_r = sum(radii) / len(radii)
        max_r = max(radii)
    else:
        avg_r = max_r = 0.0

    print(f"  Working Memory  (r < 0.30):   {working:4d}  nodes")
    print(f"  Drifting        (r < 0.80):   {drifting:4d}  nodes")
    print(f"  Nearing Archive (r < 0.95):   {nearing:4d}  nodes")
    print(f"  Beyond boundary (r >= 0.95):  {beyond:4d}  nodes")
    print(f"  Avg radius: {avg_r:.4f}   Max radius: {max_r:.4f}")

    # --- Certainty / Access (Fisher-Rao) ---
    _print_header("Certainty / Access (Fisher-Rao)")

    cur.execute(
        "SELECT AVG(access_count), MAX(access_count), AVG(sigma) "
        "FROM memory_nodes WHERE status = 0"
    )
    row = cur.fetchone()
    avg_access = row[0] or 0.0
    max_access = row[1] or 0
    avg_sigma = row[2] or 0.0

    print(f"  Avg access count:  {avg_access:.2f}")
    print(f"  Max access count:  {max_access}")
    print(f"  Avg sigma:         {avg_sigma:.4f}")
    if avg_sigma > 0:
        print(f"  Confidence:        {'HIGH' if avg_sigma < 0.5 else 'MEDIUM' if avg_sigma < 1.0 else 'LOW'}")

    # --- Cohomology Frictions ---
    _print_header("Cohomology Frictions")

    cur.execute(
        "SELECT COUNT(*) FROM memory_nodes WHERE annotation IS NOT NULL AND annotation != ''"
    )
    friction_count = cur.fetchone()[0]
    print(f"  Nodes with annotations: {friction_count}")
    if active_count > 0:
        pct = (friction_count / active_count) * 100
        print(f"  Friction ratio:         {pct:.1f}% of active nodes")

    # --- ASCII Disk Map ---
    _print_header("Poincare Disk Map (Active Nodes)")
    if active_nodes:
        _ascii_disk(active_nodes)
    else:
        print("  (no active nodes to display)")

    # --- Legend ---
    print("  Legend: + = origin, * = node, . = disk boundary")
    print()

    conn.close()


if __name__ == "__main__":
    main()
