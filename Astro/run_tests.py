"""Small dependency-free smoke test runner for the integrated pipeline."""

from tests.test_pipeline import test_period_recovery
from submission.validate_submission import validate
from pathlib import Path


def main():
    test_period_recovery()
    path = Path("_test_submission.csv")
    path.write_text("star_id,prediction,confidence,period,depth_ppm,duration_hours\nS,0,0.2,,,\n")
    try:
        validate(path, ["S"])
    finally:
        path.unlink(missing_ok=True)
    print("PASS detector")
    print("PASS submission validator")


if __name__ == "__main__":
    main()
