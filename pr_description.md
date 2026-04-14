# feat: Phase 3 Input Validation Layer (Reject Noisy Inputs)

This Pull Request introduces a dedicated validation layer to the Intelligent Property Valuation system. It ensures that only physically plausible and logically consistent property data is used for price prediction, enhancing the reliability of the advisor.

### What Has Been Changed?

*   **New Validation Engine (`validator.py`):** Created a standalone `PropertyInputValidator` that performs three tiers of checks:
    *   **Hard Bounds:** Blocks inputs that are physically impossible (e.g., area < 300 sq ft or > 25,000 sq ft).
    *   **Soft Bounds:** Issues warnings for inputs that are outside the typical distribution of the training data (e.g., area > 16,200 sq ft).
    *   **Cross-Field Logic:** Enforces common-sense relationships between features:
        *   **Cramped Housing:** Errors if Area-per-bedroom is < 200 sq ft; warns if < 350 sq ft.
        *   **Bath/Bed Ratios:** Warns if bathrooms exceed bedrooms or the ratio is > 2.0.
        *   **Anomaly Detection:** Flags 3+ story buildings with very small footprints.
        *   **Plausibility Check:** Blocks configurations like 5+ bedrooms in less than 2,000 sq ft.
*   **UI Integration (`app.py`):** The Streamlit frontend now intercepts inputs before they reach the model:
    *   **Red Errors:** Completely block the "Predict Price" action until resolved.
    *   **Orange Warnings:** Surface non-blocking alerts to the user, indicating the model might be less accurate for unusual inputs.

### Why is this PR Useful?

*   **Data Integrity:** Prevents the ML model from generating "garbage-in-garbage-out" results based on typos or nonsensical inputs.
*   **User Guidance:** Educates the user on why certain inputs might be unusual or physically impossible in the context of the dataset.
*   **Robustness:** Adds a critical "defense-in-depth" layer between the raw user input and the predictive heart of the system.

### Future Work
*   Add PDF generation tools for exporting valuation reports with validation summaries.