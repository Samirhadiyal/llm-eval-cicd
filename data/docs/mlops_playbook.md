# MLOps Production Engineering Playbook

## Section 1: Feature Store & Data Ingestion
- **Ingestion Cadence**: Batch features are updated every 6 hours via Airflow pipelines. Online features are streamed real-time via Kafka into Redis.
- **Validation Thresholds**: Data drift is evaluated using Kolmogorov-Smirnov (KS) tests. An alert is triggered if the p-value drops below 0.05 over a 24-hour rolling window.
- **Schema Enforcement**: Missing values in critical numerical features (e.g., `user_age`, `transaction_amount`) are imputed using the median of the trailing 7 days. If missingness exceeds 15% in a batch, the pipeline halts immediately.

## Section 2: Model Training & Evaluation
- **Retraining Triggers**: Automated model retraining is executed when the production F1-score drops below 0.82 or when statistical data drift affects >20% of input features.
- **Evaluation Criteria**: Candidate models must outperform the baseline model by at least 2.5% accuracy on the holdout evaluation dataset and maintain a p95 latency under 120 milliseconds.
- **Champion-Challenger Strategy**: New models undergo shadow deployment for 7 days, receiving 100% of production traffic in mirror mode before being promoted to active serving via a 10% canary rollout.

## Section 3: Monitoring & Incident Response
- **Alert Escalation**: Severity-1 incidents (P95 latency > 500ms or error rate > 2%) trigger an automated rollback to the previous model artifact within 90 seconds.
- **Resource Constraints**: Model serving nodes must operate under 80% RAM utilization and keep GPU VRAM footprint below 14GB per instance.