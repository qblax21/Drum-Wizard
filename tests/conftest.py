import os
import tempfile

# The server keeps jobs under a data directory chosen at import time: never let tests write to the real one.
os.environ["DRUMWIZARD_DATA"] = tempfile.mkdtemp(prefix="drumwizard-test-")
