# E-Cut
Based on G-Cut, I add more metrics, such as for angle and radius.
The graph cut is no long based on topological trees, but a highly
interconnected network.

## Usage
```python

from ecut import ECut
from ecut.swc_handler import parse_swc, write_swc

# NOTE: the node numbering of this tree should be SORTED, and starts from ZERO.
tree = parse_swc('filepath.swc')
e = ECut(tree, [0, 100])
e.run()
```