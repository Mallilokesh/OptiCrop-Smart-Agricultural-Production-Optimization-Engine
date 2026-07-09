from __future__ import annotations


FEATURE_FIELDS = [
    {"name": "N", "label": "Nitrogen (N)", "min": 0, "max": 160, "step": "1"},
    {"name": "P", "label": "Phosphorous (P)", "min": 0, "max": 160, "step": "1"},
    {"name": "K", "label": "Potassium (K)", "min": 0, "max": 220, "step": "1"},
    {"name": "temperature", "label": "Temperature (C)", "min": 0, "max": 50, "step": "0.1"},
    {"name": "humidity", "label": "Humidity (%)", "min": 0, "max": 100, "step": "0.1"},
    {"name": "ph", "label": "pH Level", "min": 0, "max": 14, "step": "0.1"},
    {"name": "rainfall", "label": "Rainfall (mm)", "min": 0, "max": 350, "step": "0.1"},
]


def validate_form(form_data) -> tuple[dict[str, float], dict[str, str]]:
    values = {}
    errors = {}

    for field in FEATURE_FIELDS:
        name = field["name"]
        raw_value = str(form_data.get(name, "")).strip()
        if not raw_value:
            errors[name] = "This value is required."
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors[name] = "Enter a valid number."
            continue

        if value < field["min"] or value > field["max"]:
            errors[name] = f"Enter a value from {field['min']} to {field['max']}."
            continue

        values[name] = value

    return values, errors
