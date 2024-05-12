# FocusLearn

FocusLearn is a fully-interpretable modular neural network capable of matching or surpassing the predictive performance of deep networks trained on multivariate time series. In FocusLearn, a recurrent neural network learns the temporal dependencies in the data, while a multi-headed attention layer learns to weight selected features while also suppressing redundant features. Modular neural networks are then trained in parallel and independently, one for each selected feature. This modular approach allows the user to inspect how features influence outcomes in the exact same way as with additive models. Experimental results show that this new approach outperforms additive models in both regression and classification of time series tasks, achieving predictive performance that is comparable to state-of-the-art, non-interpretable deep networks applied to time series.

Pre-print: 

```bibtex
@article{su2023modular,
  title={Modular Neural Networks for Time Series Forecasting: Interpretability and Feature Selection using Attention},
  author={Su, Qiqi and Kloukinas, Christos and d'Garcez, Artur},
  journal={arXiv preprint arXiv:2311.16834},
  year={2023}
}
```