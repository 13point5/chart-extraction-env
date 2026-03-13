import verifiers as vf

from chart_extraction_env.environment import load_environment as _load_environment


def load_environment(**kwargs) -> vf.Environment:
    """Load the chart extraction environment."""
    return _load_environment(**kwargs)
