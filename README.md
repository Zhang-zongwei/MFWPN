# A machine learning model for hub-height short-term wind speed prediction

*2025.01*

## Abstract

Accurate short-term wind speed prediction is crucial for maintaining the safe, stable, and efficient operation of wind power systems. We propose a multivariate meteorological data fusion wind prediction network (MFWPN) to study fine-grid vector wind speed prediction, taking Northeast China as an example.

![Graphabstract](data/graphabstract.jpg)


## Installation

```
conda create -n wpn python=3.8
conda activate wpn
git clone https://github.com/Zhang-zongwei/MF-WPN.git
cd MF-WPN
pip install -r requirements.txt
```

## Overview

- `data/:` contains a test set for the northeast region of the manuscript, which can be downloaded via the link .
- `openstl/models/mfwpn.py:` contains the network architecture of this MF-WPN.
- `openstl/modules/:` contains partial modules of the MF-WPN.
- `utils/:` contains data processing files and loss calculations..
- `chkfile/:` contains weights for predicting 24-hour wind speeds in the Northeast region, which can be downloaded via the link.
- `result/:` contains predicted wind speed results and evaluation methods.
- `config.py：`  training configs for the MF-WPN.
- `main.py:` Train the MF-WPN.
- `test.py` Test the MF-WPN.

## Data preparation
The data used in this study and its processing have been described in detail in the manuscript. To facilitate the testing, we have prepared the [MFWPN weights](https://drive.google.com/file/d/1YrJP1sCWUcsHcYdNL_sWFbkuS4WfaeJf/view?usp=sharing) , [test dataset](train_val_dataset](https://drive.google.com/drive/folders/1ppxlPq2PABTpUfXTWfQZ3ZDCXvmXfuqk?usp=sharing) and [train_val_dataset](https://drive.google.com/drive/folders/1ppxlPq2PABTpUfXTWfQZ3ZDCXvmXfuqk?usp=sharing).

## Train
After the data is ready, use the following commands to start training the model:
```
python main.py
```

## Test
We provide the test model weights and test dataset, which can be tested using the following commands after downloading:
```
python test.py
```

Note that the predictions are obtained as npy files containing u, v variables, which need to be converted to wind speed results using: 
```
cd result
python uv_to_wind.py
```
After that, we can obtain the wind speed prediction evaluation result by:
```
python evaluate.py
```
## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL),[Restormer](https://github.com/swz30/Restormer),[CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse). We sincerely appreciate for their contributions.

If you have any questions or suggestions, please do not hesitate to contact us at [zhangzongwei@stu.hit.edu.cn].
