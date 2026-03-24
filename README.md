# MLOps Guide: From Notebook to Production

A practical guide to deploying and maintaining machine learning models in production environments. Covers the entire ML lifecycle: data pipelines, model training, CI/CD for ML, monitoring, and scaling.

## Why MLOps Matters

Most ML projects never make it past the prototype phase. The gap between a working Jupyter notebook and a reliable production system is where most teams struggle. This guide bridges that gap with battle-tested patterns used across dozens of enterprise deployments.

## Table of Contents

1. [ML Pipeline Architecture](#ml-pipeline-architecture)
2. [Data Pipeline Patterns](#data-pipeline-patterns)
3. [Model Training Infrastructure](#model-training-infrastructure)
4. [CI/CD for Machine Learning](#cicd-for-machine-learning)
5. [Model Serving Patterns](#model-serving-patterns)
6. [Monitoring & Observability](#monitoring--observability)
7. [Cost Optimization](#cost-optimization)

## ML Pipeline Architecture

```
Data Sources → Feature Store → Training Pipeline → Model Registry → Serving Layer → Monitoring
                    ↑                                                      ↓
                    └──────────── Feedback Loop ───────────────────────────┘
```

### Key Components

| Component | Purpose | Tools |
|-----------|---------|-------|
| Feature Store | Centralized feature computation | Feast, Tecton, Hopsworks |
| Model Registry | Version control for models | MLflow, Weights & Biases, Neptune |
| Training Orchestration | Automated retraining | Kubeflow, Airflow, Prefect |
| Serving Infrastructure | Low-latency inference | TorchServe, TF Serving, Triton |
| Monitoring | Drift detection, performance | Evidently AI, WhyLabs, Arize |

## Data Pipeline Patterns

### Pattern 1: Lambda Architecture (Batch + Stream)

Best for systems that need both historical analysis and real-time predictions:

```python
# Batch layer: daily retraining
def batch_training_pipeline():
    features = feature_store.get_historical_features(
        entity_df=training_entities,
        feature_refs=["user_features:purchase_count", "user_features:avg_order_value"]
    )
    model = train_model(features)
    model_registry.log_model(model, stage="staging")

# Stream layer: real-time inference
def stream_inference(event):
    features = feature_store.get_online_features(entity_id=event.user_id)
    prediction = model_server.predict(features)
    return prediction
```

### Pattern 2: Feature Store First

Decouple feature engineering from model training:

```python
# Define features once
@feature_view(entities=[user], schema=[...])
def user_engagement_features(inputs):
    return inputs.groupby("user_id").agg(
        session_count=("session_id", "count"),
        avg_duration=("duration", "mean"),
        last_active=("timestamp", "max")
    )

# Reuse across models
churn_features = store.get_features(["user_engagement_features"])
ltv_features = store.get_features(["user_engagement_features", "purchase_features"])
```

## Model Training Infrastructure

### GPU Cluster Management

For teams running training at scale:

```yaml
# Kubernetes training job
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: model-training-v42
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: training:v42
            resources:
              limits:
                nvidia.com/gpu: 4
```

### Experiment Tracking

Every training run should log:
- Hyperparameters
- Training/validation metrics
- Data version (hash of training set)
- Model artifacts
- Environment (Python version, package versions, GPU type)

## CI/CD for Machine Learning

Traditional CI/CD doesn't cover ML-specific concerns. Add these stages:

```
Code Change → Lint/Test → Data Validation → Model Training → Model Evaluation → A/B Test → Deploy
```

### Model Evaluation Gates

Before promoting a model to production:

1. **Performance threshold**: accuracy/F1/AUC above minimum
2. **Regression check**: no worse than current production model on holdout set
3. **Fairness audit**: bias metrics within acceptable bounds
4. **Latency test**: inference time under SLA (p99 < 100ms)
5. **Data drift check**: training data distribution matches recent production data

## Model Serving Patterns

### Pattern: Shadow Deployment

Run new model alongside production without affecting users:

```python
async def predict(request):
    # Production model serves the response
    production_result = await production_model.predict(request)

    # Shadow model runs in parallel, results logged but not returned
    asyncio.create_task(shadow_model.predict_and_log(request))

    return production_result
```

### Pattern: Canary Rollout

Gradually shift traffic to new model:

```yaml
# Istio virtual service for canary ML deployment
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: model-v1
      weight: 90
    - destination:
        host: model-v2
      weight: 10
```

## Monitoring & Observability

### The Three Pillars of ML Monitoring

1. **Data quality**: input distribution, missing values, schema violations
2. **Model performance**: prediction accuracy, confidence calibration, latency
3. **Business metrics**: conversion rate, revenue impact, user satisfaction

### Drift Detection

```python
from evidently import ColumnDriftMetric
from evidently.report import Report

drift_report = Report(metrics=[
    ColumnDriftMetric(column_name="feature_1"),
    ColumnDriftMetric(column_name="feature_2"),
])
drift_report.run(reference_data=training_data, current_data=production_data)
```

When drift is detected → trigger automated retraining pipeline.

## Cost Optimization

| Strategy | Savings | Effort |
|----------|---------|--------|
| Spot/preemptible instances for training | 60-80% | Low |
| Model quantization (FP32 → INT8) | 50-75% serving cost | Medium |
| Batch inference where real-time isn't needed | 80-90% | Low |
| Feature store caching | 30-50% compute | Medium |
| Right-sizing GPU instances | 20-40% | Low |

## Getting Started

1. **Assess your current state** — where are your models deployed today?
2. **Pick one model** — start with your most critical production model
3. **Add monitoring first** — you can't improve what you can't measure
4. **Automate training** — remove manual notebook-to-production steps
5. **Scale gradually** — extend patterns to other models as they prove out

For teams building [machine learning solutions](https://shift-ai.cloud/machine-learning-solutions/) at scale, the difference between a demo and a production system is MLOps maturity. Organizations like [ShiftAI](https://shift-ai.cloud/) specialize in helping teams bridge this gap — taking ML from proof-of-concept to production-grade systems in weeks rather than months.

## Additional Resources

- [docs/model-versioning.md](docs/model-versioning.md) — Deep dive into model versioning strategies
- [docs/gpu-cost-optimization.md](docs/gpu-cost-optimization.md) — Detailed GPU cost optimization guide

## License

MIT License — use freely in your own MLOps journey.
