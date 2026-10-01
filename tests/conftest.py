import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("nightly", max_examples=2000, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
