
-- Heart Disease Project — MySQL schema
-- Database: heart_disease_project
-- Author: project setup
--
-- Design notes:
--   * patient_id is a surrogate key (the UCI source has no patient identifier).
--     It is assigned at raw-load time as the row's 1-based position in the
--     processed.cleveland.data file, and used as the join key everywhere else.
--   * Raw table stores values verbatim, including '?' for missing — so the
--     numeric columns that can be missing (`ca`, `thal`) are VARCHAR there.
--   * Cleaned table stores human-readable category labels (ENUMs) so that
--     Power BI dashboards are readable without an extra mapping layer.
--     One-hot encoding is applied only in Python, in memory, never persisted.
--   * Model-output tables FK to model_runs so every metric / prediction / SHAP
--     value is traceable to a specific training run (run_id, git_sha, seed,
--     hyperparameters). Health analytics standard: no overwrites, full lineage.
--   * SHAP per-patient values use LONG format (one row per
--     patient × feature × run) to support Power BI drill-down.
--   * Every table has `created_at` for auditability.
-- ============================================================================

USE heart_disease_project;


-- 1. raw_heart_disease — verbatim copy of processed.cleveland.data

DROP TABLE IF EXISTS shap_patient_level;
DROP TABLE IF EXISTS shap_global_importance;
DROP TABLE IF EXISTS model_predictions;
DROP TABLE IF EXISTS model_training_results;
DROP TABLE IF EXISTS model_runs;
DROP TABLE IF EXISTS data_quality_summary;
DROP TABLE IF EXISTS cleaned_patient_records;
DROP TABLE IF EXISTS raw_heart_disease;

CREATE TABLE raw_heart_disease (
    patient_id    INT UNSIGNED   NOT NULL PRIMARY KEY COMMENT 'Surrogate key: 1-based row position in source file',
    age           DECIMAL(4,1)   NOT NULL,
    sex           DECIMAL(2,1)   NOT NULL COMMENT '1.0 = male, 0.0 = female',
    cp            DECIMAL(2,1)   NOT NULL COMMENT 'Chest pain type 1.0–4.0',
    trestbps      DECIMAL(5,1)   NOT NULL COMMENT 'Resting BP in mm Hg',
    chol          DECIMAL(5,1)   NOT NULL COMMENT 'Serum cholesterol mg/dl',
    fbs           DECIMAL(2,1)   NOT NULL COMMENT 'Fasting blood sugar > 120 mg/dl: 1.0 / 0.0',
    restecg       DECIMAL(2,1)   NOT NULL COMMENT 'Resting ECG 0.0 / 1.0 / 2.0',
    thalach       DECIMAL(5,1)   NOT NULL COMMENT 'Max heart rate achieved',
    exang         DECIMAL(2,1)   NOT NULL COMMENT 'Exercise-induced angina 1.0 / 0.0',
    oldpeak       DECIMAL(3,1)   NOT NULL COMMENT 'ST depression',
    slope         DECIMAL(2,1)   NOT NULL COMMENT 'Slope of peak exercise ST 1.0–3.0',
    ca            VARCHAR(8)     NOT NULL COMMENT 'Number of major vessels 0–3, or ? if missing',
    thal          VARCHAR(8)     NOT NULL COMMENT '3.0 / 6.0 / 7.0, or ? if missing',
    num           TINYINT UNSIGNED NOT NULL COMMENT 'Diagnosis severity 0–4 (target before binarization)',
    created_at    TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;


-- 2. cleaned_patient_records — imputed, human-readable, model-ready
--    (excluding the one-hot expansion which is done in Python)

CREATE TABLE cleaned_patient_records (
    patient_id        INT UNSIGNED NOT NULL PRIMARY KEY,
    age               TINYINT UNSIGNED NOT NULL,
    sex               ENUM('male', 'female') NOT NULL,
    cp                ENUM('typical_angina', 'atypical_angina', 'non_anginal_pain', 'asymptomatic') NOT NULL,
    trestbps          SMALLINT UNSIGNED NOT NULL COMMENT 'Resting BP mm Hg',
    chol              SMALLINT UNSIGNED NOT NULL COMMENT 'Cholesterol mg/dl',
    fbs               TINYINT UNSIGNED NOT NULL COMMENT '0 or 1',
    restecg           ENUM('normal', 'st_t_abnormality', 'lv_hypertrophy') NOT NULL,
    thalach           SMALLINT UNSIGNED NOT NULL COMMENT 'Max heart rate',
    exang             TINYINT UNSIGNED NOT NULL COMMENT '0 or 1',
    oldpeak           DECIMAL(3,1) NOT NULL,
    slope             ENUM('upsloping', 'flat', 'downsloping') NOT NULL,
    ca                TINYINT UNSIGNED NOT NULL COMMENT '0–3, median-imputed if originally missing',
    thal              ENUM('normal', 'fixed_defect', 'reversable_defect') NOT NULL COMMENT 'Mode-imputed if originally missing',
    num               TINYINT UNSIGNED NOT NULL COMMENT 'Original severity 0–4',
    target_binary     TINYINT UNSIGNED NOT NULL COMMENT '0 = no disease (num=0), 1 = disease (num>=1)',
    ca_was_imputed    TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Audit flag: 1 if ca was imputed for this row',
    thal_was_imputed  TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Audit flag: 1 if thal was imputed for this row',
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES raw_heart_disease(patient_id) ON DELETE CASCADE
) ENGINE=InnoDB;


-- 3. data_quality_summary — one row per (column, check_type, run_timestamp)

CREATE TABLE data_quality_summary (
    check_id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_timestamp     TIMESTAMP NOT NULL COMMENT 'Identifies a DQ batch (all checks from one run share this)',
    column_name       VARCHAR(64) NOT NULL,
    check_type        VARCHAR(64) NOT NULL COMMENT 'e.g. missing_count, out_of_range, unexpected_category, duplicate_rows',
    check_value       DECIMAL(15,4) NULL COMMENT 'Numeric result of the check',
    check_detail      TEXT NULL,
    passed            TINYINT UNSIGNED NOT NULL COMMENT '1 = passed, 0 = failed',
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dq_run (run_timestamp),
    INDEX idx_dq_col (column_name)
) ENGINE=InnoDB;


-- 4. model_runs — one row per training run; everything below FKs to this

CREATE TABLE model_runs (
    model_run_id      INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_timestamp     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_type        VARCHAR(64) NOT NULL COMMENT 'e.g. logistic_regression, random_forest, gradient_boosting',
    random_seed       INT NOT NULL,
    git_sha           VARCHAR(40) NULL COMMENT 'Commit SHA at training time, for reproducibility',
    hyperparameters   JSON NULL,
    cv_folds          TINYINT UNSIGNED NULL COMMENT 'Number of stratified CV folds',
    notes             TEXT NULL,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;


-- 5. model_training_results — metrics in LONG format
--    fold = NULL for aggregate (mean / OOF) metrics, otherwise per-fold

CREATE TABLE model_training_results (
    result_id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    model_run_id      INT UNSIGNED NOT NULL,
    fold              TINYINT UNSIGNED NULL COMMENT 'NULL = aggregate across folds',
    metric_name       VARCHAR(64) NOT NULL COMMENT 'accuracy, roc_auc, pr_auc, precision, recall, f1, brier_score, etc.',
    metric_value      DECIMAL(10,6) NOT NULL,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    INDEX idx_mtr_run (model_run_id, metric_name)
) ENGINE=InnoDB;


-- 6. model_predictions — one row per (model_run, patient) held-out prediction

CREATE TABLE model_predictions (
    prediction_id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    model_run_id           INT UNSIGNED NOT NULL,
    patient_id             INT UNSIGNED NOT NULL,
    fold                   TINYINT UNSIGNED NOT NULL COMMENT 'Which CV fold held out this patient',
    predicted_class        TINYINT UNSIGNED NOT NULL,
    predicted_probability  DECIMAL(7,6) NOT NULL COMMENT 'P(target_binary = 1)',
    true_label             TINYINT UNSIGNED NOT NULL,
    created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    FOREIGN KEY (patient_id)   REFERENCES cleaned_patient_records(patient_id) ON DELETE CASCADE,
    UNIQUE KEY uq_pred (model_run_id, patient_id),
    INDEX idx_pred_run (model_run_id)
) ENGINE=InnoDB;


-- 7. shap_global_importance — one row per (model_run, feature)

CREATE TABLE shap_global_importance (
    importance_id     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    model_run_id      INT UNSIGNED NOT NULL,
    feature_name      VARCHAR(64) NOT NULL,
    mean_abs_shap     DECIMAL(10,6) NOT NULL,
    rank_position     SMALLINT UNSIGNED NOT NULL COMMENT '1 = most important',
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    UNIQUE KEY uq_global (model_run_id, feature_name)
) ENGINE=InnoDB;


-- 8. shap_patient_level — LONG format, one row per (run, patient, feature)

CREATE TABLE shap_patient_level (
    shap_id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    model_run_id      INT UNSIGNED NOT NULL,
    patient_id        INT UNSIGNED NOT NULL,
    feature_name      VARCHAR(64) NOT NULL,
    shap_value        DECIMAL(10,6) NOT NULL COMMENT 'Signed SHAP contribution toward P(target=1)',
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    FOREIGN KEY (patient_id)   REFERENCES cleaned_patient_records(patient_id) ON DELETE CASCADE,
    INDEX idx_shap_patient (patient_id, model_run_id),
    INDEX idx_shap_run     (model_run_id)
) ENGINE=InnoDB;


-- VIEWS for Power BI
DROP VIEW IF EXISTS vw_dashboard_patient_summary;
DROP VIEW IF EXISTS vw_dashboard_model_comparison;
DROP VIEW IF EXISTS vw_dashboard_shap_summary;

-- ---- 9a. Per-patient summary: features + the most recent model's prediction
CREATE VIEW vw_dashboard_patient_summary AS
SELECT
    c.patient_id,
    c.age,
    c.sex,
    c.cp,
    c.trestbps,
    c.chol,
    c.fbs,
    c.restecg,
    c.thalach,
    c.exang,
    c.oldpeak,
    c.slope,
    c.ca,
    c.thal,
    c.target_binary           AS true_label,
    p.predicted_class,
    p.predicted_probability,
    p.fold                    AS prediction_fold,
    mr.model_run_id           AS latest_model_run_id,
    mr.model_type             AS latest_model_type,
    mr.run_timestamp          AS latest_run_timestamp
FROM cleaned_patient_records c
LEFT JOIN (
    SELECT model_run_id, model_type, run_timestamp
    FROM model_runs
    ORDER BY run_timestamp DESC
    LIMIT 1
) mr ON 1=1
LEFT JOIN model_predictions p
       ON p.patient_id = c.patient_id
      AND p.model_run_id = mr.model_run_id;

-- ---- 9b. Aggregated metric comparison across runs
CREATE VIEW vw_dashboard_model_comparison AS
SELECT
    mr.model_run_id,
    mr.model_type,
    mr.random_seed,
    mr.run_timestamp,
    mtr.metric_name,
    mtr.metric_value
FROM model_runs mr
JOIN model_training_results mtr
  ON mtr.model_run_id = mr.model_run_id
WHERE mtr.fold IS NULL;   -- aggregate metrics only

-- ---- 9c. Top SHAP features for the most recent run
CREATE VIEW vw_dashboard_shap_summary AS
SELECT
    sgi.model_run_id,
    mr.model_type,
    mr.run_timestamp,
    sgi.feature_name,
    sgi.mean_abs_shap,
    sgi.rank_position
FROM shap_global_importance sgi
JOIN model_runs mr ON mr.model_run_id = sgi.model_run_id
WHERE sgi.model_run_id = (
    SELECT model_run_id
    FROM model_runs
    ORDER BY run_timestamp DESC
    LIMIT 1
);
