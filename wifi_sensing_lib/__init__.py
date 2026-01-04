import os
import sys

# Ensure vendorized dependencies are in path
# Assuming they are siblings to wifi_sensing_lib in the repo root
# OR we assume the user installs them? 
# The prompt implies we fuse them.
# Let's add the paths relative to this file.

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WIPICAP_PATH = os.path.join(_ROOT, 'WiPiCap', 'Python')
_RFCRATE_PATH = os.path.join(_ROOT, 'RF_CRATE')

if _WIPICAP_PATH not in sys.path:
    sys.path.append(_WIPICAP_PATH)
if _RFCRATE_PATH not in sys.path:
    # RF_CRATE likely has relative imports inside it, so adding its root is good.
    sys.path.append(_RFCRATE_PATH)
