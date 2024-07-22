from pathlib import Path


log_path = Path(__file__).parent / "logs"
log_path.mkdir(parents=True, exist_ok=True)
