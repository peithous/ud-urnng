"""Contains functions useful for the test suite."""
import sys

from absl import flags

FLAGS = flags.FLAGS

# pytest doesn't run the test as a main, so it doesn't parse the flags.
# If flags are required in tests, this will ensure that they are manually
# parsed and the desired flag exists.
def ensure_flag(flagname: str) -> None:
    """Ensures that a flag, e.g. `test_tmpdir`, is present without app.run()."""
    try:
        getattr(FLAGS, flagname)
    except flags.UnparsedFlagAccessError:
        FLAGS(sys.argv)
    finally:
        assert getattr(FLAGS, flagname)
