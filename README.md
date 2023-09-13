
<div align="left">
  <img src="https://howso.com/assets/images/Howso_Readme.svg">
</div>

The Howso Engine&trade; is a natively and fully explainable ML engine and toolbox, 
serving as an alternative to black box AI. Its core features give users data 
exploration and machine learning capabilities through the creation and use
of Trainees that help users store, explore, and analyze the relationships in
their data. Howso&trade; leverages an instance-based learning approach with
strong ties to the [k-nearest neighbors
algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) and
[information theory](https://en.wikipedia.org/wiki/Information_theory) to scale
for real world applications.

At the core of Howso is the concept of a Trainee, a collection of cases that
comprise knowledge. In traditional ML, this is typically referred to as a model,
but a Trainee may additionally include metadata, parameters, details of feature
attributes, with data lineage and provenance. Unlike traditional ML, Trainees
are designed to be versatile, a single model that after training a dataset can
do the following without the need to retrain:
- Perform **classification** on any target feature using any set of
  input features
- Perform **regression** on any target feature using any set of input features
- Perform **anomaly detection** based on any set of features
- Measure **feature importance** for predicting any target feature
- **Synthesize** data that maintains the same feature relationships of the
  original data while maintaining privacy

Furthermore, Trainees are auditable, debuggable, and editable.
- **Debuggable**: Every prediction of a Trainee can be drilled down to
  investigate which cases from the training data were used to make
  the prediction.
- **Auditable**: Trainees manage metadata about themselves including: when data
  is trained, when training data is edited, when data is removed, etc.
- **Editable**: Specific cases of training data can be removed, edited, and
  emphasized (through case weighting) without the need to retrain.

## Resources
- [Documentation](https://docs.howso.com)
- [Howso Engine Recipes (sample notebooks)](https://github.com/howsoai/howso-engine-recipes)
- [Howso Playground](https://playground.howso.com)

## General Overview
This Repo provides the Python interface with
[Howso Engine](https://github.com/howsoai/howso-engine) that exposes the Howso
Engine functionality. The Client objects directly interface with the engine API
endpoints while the Trainee objects provides the python functionality for
general users. Client functions may be called by the user but for most workflows
the Trainee functionality is sufficient. Each Trainee represents an individual
Machine Learning object or model that can perform functions like training and
predicting, while a client may manage the API interface for multiple Trainees.


## Supported Platforms

Compatible with Python versions: 3.8, 3.9, 3.10, and 3.11

#### Operating Systems

| OS      | x86_64 | arm64 |
|---------|--------|-------|
| Windows | Yes    | No    |
| Linux   | Yes    | Yes   |
| MacOS   | Yes    | Yes   |


## Install

To install the current release:
```bash
pip install howso-engine
```

You can verify your installation is working by running the following command in
your python environment terminal:

```bash
verify_howso_install
```

See the Howso Engine
[Install Guide](https://docs.howso.com/getting_started/installing.html) for
additional help and troubleshooting information.
## Usage

The Howso Engine is designed to support users in the pursuit of many different
machine learning tasks using Python.

Below is a very high-level set of steps recommended for using the Howso Engine:

1. Define the feature attributes of the data (Feature types, bounds, etc.)
2. Create a Trainee and set the feature attributes
3. Train the Trainee with the data
4. Call Analyze on the Trainee to find optimal hyperparameters
5. Explore your data!

Once the Trainee has been given feature attributes, trained, and analyzed, then
the Trainee is ready to be used for all supported machine learning tasks. At
this point one could start making predictions on unseen data, investigate the
most noisy features, find the most anomalous training cases, and much more.

Please see the [User Guide](https://docs.howso.com/user_guide/index.html) for
basic workflows as well as additional information about:
- Anomaly detection
- Classification
- Regression
- Time-series forecasting
- Feature importance analysis
- Reinforcement learning
- Data synthesis
- Prediction auditing
- Measuring model performance (global or conditional)
- Bias mitigation
- Trainee editing
- ID-based privacy

There is also a set of basic [Jupyter notebooks](https://jupyter.org/) to run
that provides a
[complete set of examples](https://docs.howso.com/examples/index.html) of how
to use Howso Engine.

## License

[License](LICENSE.txt)

## Contributing

[Contributing](CONTRIBUTING.md)
