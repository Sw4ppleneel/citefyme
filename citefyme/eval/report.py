def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {
        c: max(len(c), *(len(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
              for r in rows))
        for c in columns
    }
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(
            (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c])
            for c in columns
        ))
