from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.model import save_model, train_and_select_model


def main() -> None:
    payload = train_and_select_model(BASE_DIR)
    model_path = BASE_DIR / "artifacts" / "best_model.pkl"
    save_model(payload, model_path)

    print("Model evaluation results")
    print("=" * 24)
    for item in payload["results"]:
        print(f"{item['name']}: {item['accuracy']:.4f}")
    print()
    print(f"Best model: {payload['best_model_name']} ({payload['best_accuracy']:.4f})")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
