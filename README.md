# music-transformer

The [Music Transformer](https://arxiv.org/abs/1809.04281), or Transformer Decoder with Relative Self-Attention, is a deep learning sequence model designed to generate music. It builds upon the Transformer architecture to consider the relative distances between different elements of the sequence rather than / along with their absolute positions in the sequence. I explored my interest in AI-generated music through this project and learned quite a bit about current research in the field of AI in terms of both algorithms and architectures. This repository contains Python scripts to preprocess MIDI data, train a pre-LayerNorm Music Transformer using PyTorch, as well as to generate MIDI files with a trained (or if you're brave, untrained) Music Transformer.

While the data preprocessing and generation functionality require MIDI files and the event vocabulary described in [Oore et al. 2018](https://arxiv.org/pdf/1808.03715.pdf) or vocabulary.py, anyone should be able to use the train.py script to train their own Relative Attention Transformer on any dataset, provided correct specification of hyperparameters, and provided they have properly preprocessed their data into a single PyTorch tensor. Do create an issue if something does not work as expected.

## 📋 Table of Contents
- [Key Dependencies](#key-dependencies)
- [Setting up](#setting-up)
- [Preprocess MIDI Data](#preprocess-midi-data)
- [Train a Music Transformer](#train-a-music-transformer)
  - [Training Scripts Overview](#training-scripts-overview)
- [Generate Music](#generate-music)
- [Analysis Tools](#analysis-tools)
- [Acknowledgements](#acknowledgements)
- PyTorch 2.1.0
- Mido 1.2.9

## 🚀 Setting up
Clone the git repository, cd into it if necessary, and install the requirements. Then you're ready to preprocess MIDI files, as well as train and generate music with a Music Transformer.

```shell
git clone https://github.com/spectraldoy/music-transformer
cd ./music-transformer
pip install -r requirements.txt
```

## Preprocess MIDI Data
Most sequence models require a general upper limit on the length of the sequences being modeled, it being too computationally or memory expensive to handle longer sequences. So, suppose you have a directory of MIDI files at .../datapath/ (for instance, any of the folders in the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)), and would like to convert these files into an event vocabulary that can be trained on, cut these sequences to be less than or equal to an approximate maximum length, lth, and store this processed data in a single PyTorch tensor (for use with torch.utils.data.TensorDataset) at .../processed_data.pt. Running the preprocessing.py script as follows:

```shell
python preprocessing.py .../datapath/ .../processed_data.pt lth
```

will translate the MIDI files to the event vocabulary laid out in vocabulary.py, tokenize it with functionality from tokenizer.py, cut the data to approximately the specified lth, augment the dataset by a default set of pitch transpositions and stretches in time, and finally, store the sequences as a single concatenated PyTorch tensor at .../processed_data.pt. The cutting is done by randomly generating a number from 0 to lth, randomly sampling a window of that length from the sequence, and padding with pad_tokens to the maximum sequence length in the data. Pitch transpositions and factors of time stretching can also be specified when running the script from the shell (for details, run python preprocessing.py -h).

**NOTE:** This script will not work properly for multi-track MIDI files, and any other instruments will automatically be converted to piano (the reason for this is that I worked only with single-track piano MIDI for this project).

## Train a Music Transformer
Being highly space complex, as well as requiring inordinate amounts of time to train on both GPUs as well as TPUs, the Music Transformer needs to be checkpointed while training. I implemented a deliberate and slightly unwieldy checkpointing mechanism in the MusicTransformerTrainer class from train.py, to be able to checkpoint while training a Music Transformer. At its very simplest, given a path to a preprocessed dataset in the form of a PyTorch tensor, .../preprocessed_data.pt, and specifying a path at which to checkpoint the model, .../ckpt_path.pt, a path at which to save the model, .../save_path.pt, and the number of epochs for which to train the model this session, epochs, running the following:

```shell
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs
```

will split the data 80/20 into training and validation sets, train the model for the specified number of epochs on the given dataset, printing progress messages, and will checkpoint the optimizer state, learning rate schedule state, model weights, and hyperparameters if a KeyboardInterrupt is encountered, anytime a progress message is printed, and when the model finishes training for the specified number of epochs. Hyperparameters can also be specified when creating a new model, i.e., not loading from a checkpoint (for details on these, run python train.py -h). However, if the -l or --load-checkpoint flag is also entered:

```shell
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs -l
```

the latest checkpoint stored at .../ckpt_path.pt will be loaded, overloading any hyperparameters explicitly specified with the hyperparameters of the saved model, restoring the model, optimizer, and learning rate schedule states, and continuing training from there. Once training is completed, i.e., the model has been trained for the specified number of epochs, another checkpoint will be created, and the model's state_dict and hparams will be stored in a Python dictionary and saved at .../save_path.pt.

### Training Scripts Overview

This repository includes several training scripts for different experimental setups:

| Script | Description |
|--------|-------------|
| `train.py` | Standard training script for 200 epochs on pretrained models |
| `train_batch.py` | Baseline training for 100 epochs |
| `traincl_finale_20.py` | Curriculum learning training with 60% curriculum threshold |
| `traincl_finale_80.py` | Curriculum learning training with 80% curriculum threshold |
| `traincl_learning.py` | Curriculum learning training with 60% curriculum threshold and learning rate scheduling |

## 🎵 Generate Music
Given a trained Music Transformer's state_dict and hparams saved at .../save_path.pt, and specifying the path at which to save a generated MIDI file, .../gen_audio.mid, running the following:

```shell
python generate.py .../save_path.pt .../gen_audio.mid
```

will autoregressively greedy decode the outputs of the Music Transformer to generate a list of token_ids, convert those token_ids back to a MIDI file using functionality from tokenizer.py, and will save the output MIDI file at .../gen_audio.mid. Parameters for the MIDI generation can also be specified - 'argmax' or 'categorical' decode sampling, sampling temperature, the number of top_k samples to consider, and the approximate tempo of the generated audio (for more details, run python generate.py -h).

## 📊 Analysis Tools

The repository also includes several analysis scripts:

| Script | Purpose |
|--------|---------|
| `KLD.py` | Analyzing Kullback-Leibler Divergence |
| `plotKLD.py` | Plotting KLD and OA (Overall Accuracy) figures |
| `computed_difficulty.py` | Computing loss on all sequences and sorting them by difficulty |
| `scatter.py` | Plotting loss distribution and scatter plots for different models |

## Acknowledgements

I trained most of my models on Western classical music from [the MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro). Some models were also trained using subsets of the [ADL Piano MIDI Dataset](https://github.com/lucasnfe/adl-piano-midi/).
