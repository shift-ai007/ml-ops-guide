# Model Versioning Strategies for Production ML

Model versioning is more than just saving checkpoints. In production, you need to track the relationship between data, code, hyperparameters, and the resulting model artifact — and be able to reproduce any version on demand.

## The Four Dimensions of ML Versioning

Every production model has four versioned dependencies:

1. **Data version** — which training data was used (hash, date range, filtering criteria)
2. **Code version** — which training script, feature engineering, preprocessing
3. **Config version** — hyperparameters, architecture choices, training schedule
4. **Environment version** — Python packages, CUDA version, hardware

## Model Registry Best Practices

### Semantic Versioning for Models

Adapt semver for ML:

- **Major** (v2.0.0): Architecture change, new features, different input schema
- **Minor** (v1.1.0): Retrained on new data, hyperparameter tuning
- **Patch** (v1.0.1): Bug fix in preprocessing, data quality fix

### Stage Promotion

```
Development → Staging → Canary → Production → Archived
```

Each transition requires:
- Automated evaluation against holdout set
- Performance comparison vs current production model
- Latency benchmarks
- Approval from model owner (for major/minor bumps)

## Rollback Strategy

Always maintain the previous production model ready for instant rollback:

```python
def rollback_model(model_name: str):
    """Instant rollback to previous production version."""
    current = registry.get_model(model_name, stage="production")
    previous = registry.get_model(model_name, stage="archived", version=current.version - 1)
    
    # Swap stages
    registry.transition_model(previous.version, stage="production")
    registry.transition_model(current.version, stage="archived")
    
    # Update serving endpoint
    serving.update_model(model_name, previous.artifact_uri)
    
    # Alert team
    notify(f"Model {model_name} rolled back from v{current.version} to v{previous.version}")
```

## Reproducibility Checklist

Before promoting any model to production, verify:

- [ ] Training data is snapshotted and immutable
- [ ] All random seeds are fixed and logged
- [ ] Environment is captured (requirements.txt or conda env)
- [ ] Training can be re-run from scratch and produce identical metrics (±0.1%)
- [ ] Model card documents intended use, limitations, and bias assessment

## When to Retrain

| Trigger | Action | Urgency |
|---------|--------|---------|
| Scheduled (weekly/monthly) | Retrain on latest data | Low |
| Data drift detected | Investigate + retrain | Medium |
| Performance degradation | Emergency retrain | High |
| New feature available | Retrain with new feature | Low |
| Concept drift (business change) | Architecture review + retrain | High |

## Tools Comparison

| Tool | Strengths | Best For |
|------|-----------|----------|
| MLflow | Open source, flexible, good UI | Teams starting MLOps |
| Weights & Biases | Best experiment tracking UX | Research-heavy teams |
| DVC | Git-native, data versioning | Data-centric workflows |
| Neptune | Metadata management | Large-scale experiment tracking |
| Kubeflow | Full pipeline orchestration | K8s-native teams |

Teams implementing [production ML systems](https://shift-ai.cloud/machine-learning-solutions/) often underestimate versioning complexity. Getting it right early prevents the "which model is in production?" confusion that plagues most ML teams. For hands-on guidance, [ShiftAI's ML consulting team](https://shift-ai.cloud/) helps organizations set up proper MLOps infrastructure from day one.
