# HiFiGAN

Packages:

```bash
cd HW4
pip install -r requirements.txt
python3 scripts/get_test.py
```

`python3 scripts/get_pretrained.py` for pretrained model.

## Training

From the beginning:
```
python3 train.py -c src/configs/config_name.json
```

From checkpoint:

```
python3 train.py -r path/to/saved/checkpoint.pth
```

## Synthesizing

```bash
python3 synthesis.py -c path/to/saved/config -p path/to/model/checkpoint
```

Best model:

```bash
python3 synthesis.py -c saved/models/pretrained/train/config.json\
        -p saved/models/pretrained/train/model_best.pth
```

Check `results` directory.

## wandb

https://wandb.ai/mathalex/hifigan_project/overview

Here you also can hear the audios.
