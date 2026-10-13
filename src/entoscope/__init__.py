import logging
from importlib.metadata import version

__version__ = version("entoscope")

# entoscope owns its output instead of deferring to the root logger: dependencies configure root as a
# side effect (transformer_lens logs through the module-level ``logging.info``, which runs
# ``basicConfig``), and that would reformat entoscope's messages behind the user's back.
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("[entoscope] %(levelname)s %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
_logger.addHandler(_handler)
