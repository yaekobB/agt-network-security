"""scripts/00_sanity_check.py

Sanity check for AGT project environment.
Run this AFTER installing requirements.

Expected:
- Prints Python version and package versions
- Ends with 'Setup OK ✅'
"""

import sys

def main() -> None:
    print("=== AGT Project Sanity Check ===")
    print(f"Python: {sys.version.split()[0]}")

    # Import key dependencies
    import networkx as nx
    import numpy as np
    import pandas as pd
    import matplotlib

    print(f"networkx: {nx.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")

    # Quick tiny graph test
    G = nx.path_graph(5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4

    print("Setup OK ✅")

if __name__ == "__main__":
    main()
