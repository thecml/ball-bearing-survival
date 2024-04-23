# Predicting Survival Time of Ball Bearings in the Presence of Censoring

This repository is the official implementation of [Predicting Survival Time of Ball Bearings in the Presence of Censoring](https://arxiv.org/abs/2309.07188), AAAI Fall Symposium 2023 on Survival Prediction.

The proposed method is implemented based on scikit-survival, auton-survival and others.

<p align="left"><img src="https://github.com/thecml/ball-bearing-survival/blob/main/img/pipeline.png" width="50%" height="50%">

In this paper, we propose a novel approach for predictive maintenance of ball bearings using survival analysis.
    
License
--------
To view the license for this work, visit https://github.com/thecml/ball-bearing-survival/blob/main/LICENSE

Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/ball-bearing-survival/blob/main/requirements.txt).

Install auton-survival manually from Git:
```
pip install git+https://github.com/autonlab/auton-survival.git
```

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
```

Citation
--------
```
@misc{lillelund_predicting_2023,
      title={Predicting Survival Time of Ball Bearings in the Presence of Censoring}, 
      author={Christian Marius Lillelund and Fernando Pannullo and Morten Opprud Jakobsen and Christian Fischer Pedersen},
      howpublished = {Accepted at AAAI Fall Symposium 2023},
      year={2023},
      eprint={2309.07188},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```
