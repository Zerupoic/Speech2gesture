# Speech2gesture
code written in pytorch for speech2gesture model from paper "Learning Individual Styles of Conversational Gestures"

Parts of the code is based on "[Learning Individual Styles of Conversational Gesture](https://github.com/amirbar/speech2gesture)"

And I copied some of the code from "[*Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional Mixture Approach*](https://github.com/chahuja/mix-stage)"

### Environment

- pytorch
- tqdm
- librosa
- joblib
- transformers
- scikit_learn



### Data Preparation

Processed data in [PATS dataset](https://chahuja.com/pats/) is used to train and test this model.

``` python
# This piece of code is used in many files in this model to read data from PATS dataset
common_kwargs = dict(path2data = PATS_PATH,
                     speaker = [SPEAKER],
                     modalities = ['pose/data', 'audio/log_mel_512'],
                     fs_new = [15, 15],
                     batch_size = 4,
                     window_hop = 5)

dataloader = Data(**common_kwargs)
```



### Training

Run `train.py`. Before training, please modify `SPEAKER` and `PATS_PATH` in `train.py` to the correct value.

### Testing

Run `test.py`. Before testing, please modify `SPEAKER`, `PATS_PATH` and `MODEL_PATH_G` in `train.py` to the correct value.

### Rendering

Run `generate_video.py`. Before rendering, please modify `SPEAKER`, `PATS_PATH` and `MODEL_PATH_G` in `train.py` to the correct value.









