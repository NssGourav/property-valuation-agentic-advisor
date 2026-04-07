from dataclasses import dataclass, field

# Physical limits
HARD_BOUNDS = {
    "area": (300, 25000),
    "bedrooms": (1, 10),
    "bathrooms": (1, 6),
    "stories": (1, 5),
    "parking": (0, 5),
}

# Training data distribution limits 
SOFT_BOUNDS = {
    "area": (1650, 16200),
    "bedrooms": (1, 5),
    "bathrooms": (1, 4),
    "stories": (1, 4),
    "parking": (0, 3),
}

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

class PropertyInputValidator:
    def validate(self, inputs: dict) -> ValidationResult:
        result = ValidationResult()
        
        # Check hard bounds
        for feat, (lo, hi) in HARD_BOUNDS.items():
            val = inputs.get(feat)
            if val is None:
                result.add_error(f"Missing field: {feat}")
                continue
            if not (lo <= val <= hi):
                result.add_error(f"{feat.capitalize()} ({val}) is out of range [{lo}-{hi}]")

        # Check soft bounds
        for feat, (lo, hi) in SOFT_BOUNDS.items():
            val = inputs.get(feat)
            if val and not (lo <= val <= hi):
                result.add_warning(f"{feat.capitalize()} ({val}) is unusual for this dataset")

        # Cross-field logic
        area = inputs.get("area", 0)
        beds = inputs.get("bedrooms", 1)
        baths = inputs.get("bathrooms", 1)
        stories = inputs.get("stories", 1)

        if beds > 0:
            sqft_per_bed = area / beds
            if sqft_per_bed < 200:
                result.add_error(f"Area per bedroom ({sqft_per_bed:.0f}) is too low")
            elif sqft_per_bed < 350:
                result.add_warning("Property seems unusually cramped")

            if baths / beds > 2.0:
                result.add_warning("High bathroom to bedroom ratio")
            elif baths > beds:
                result.add_warning("More bathrooms than bedrooms")

        if stories >= 3 and area < 1500:
            result.add_warning("Multiple stories with small area - please verify")

        if beds >= 5 and area < 2000:
            result.add_error("Too many bedrooms for this area")

        return result
