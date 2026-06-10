-- Add feature_value to shap_patient_level so Power BI can display the source
-- value next to its SHAP contribution without an extra join. Stored as VARCHAR
-- to accommodate both numeric (e.g. "63") and ENUM (e.g. "asymptomatic") values.

ALTER TABLE shap_patient_level
    ADD COLUMN feature_value VARCHAR(64) NULL AFTER feature_name;
