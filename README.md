# Predicting Survival Time of Ball Bearings in the Presence of Censoring

This repository is the official implementation of [Predicting Survival Time of Ball Bearings in the Presence of Censoring](https://arxiv.org/abs/2309.07188), AAAI Fall Symposium 2023 on Survival Prediction.

The proposed method is implemented based on scikit-survival, auton-survival and others.

<p align="center"><img src="https://github.com/thecml/ball-bearing-survival/img/pipeline.png" width="95%" height="95%">

In this paper, we propose a novel approach for predictive maintenance of ball bearings using survival analysis.
    
License
--------
To view the license for this work, visit https://github.com/thecml/ball-bearing-survival/blob/main/LICENSE

Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/UE-BNNSurv/blob/main/requirements.txt).

Code was tested in virtual environment with `Python 3.9`.

Training
--------
- Download the data from: https://mega.nz/file/UEEyjBSB#y91n1D9pEcL0AXQXOHp8Fd9227SVOty39UaM7Zua2_8
- Unzip the data inside the repository to create /data.
- Please refer to `config.py` for configuration.
- Run the training code:

```
# Hyperparameter tuning
python run_cross_validation.py

# Inference
python run_inference.py
``
