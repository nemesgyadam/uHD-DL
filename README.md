# uHD EEG Machine Learning Models: A Spectacular Repository

Welcome to the *uHD EEG Machine Learning Models Repository*, where we aim to train and evaluate state-of-the-art machine learning models on **Ultra-High Density Electroencephalography (uHD EEG) data** to decode individual finger movements.

## Dataset & Inspiration

The dataset used in this repository is available through the following URL:\
[Dataset](https://osf.io/4dwjt/?view_only=d23acfd50655427fbaae381a17cbfbcc)

This incredible repository is inspired by the groundbreaking research article:\
[Individual Finger Movement Decoding using a novel Ultra-High Density EEG-based BCI system](https://www.researchgate.net/publication/364419537_Individual_finger_movement_decoding_using_a_novel_ultra-high-density_electroencephalography-based_brain-computer_interface_system)

## Unveil the Secrets of the Dataset

Begin your journey with the `inspect_data.ipynb` script. It will guide you through the basics of the dataset and provide essential insights.

## Feature Extraction: Unlock the Potential

The key to successful machine learning lies in feature extraction. The `extract_features.ipynb` script will extract features for a single subject and store the data in a convenient h5 file.

## Training Binary Finger Classifications: Power at Your Fingertips

### Support Vector Machines (SVM) Models

- Conquer SVM models for each finger pair for a single subject:\
  `train_binary_SVM.ipynb`
- Master SVM models for each subject and store results in CSV:\
  `train_binary_SVM.py`

### Deep Learning Models: Dive into the Depths

- Train Deep Learning models for each finger pair for a single subject:\
  `train_binary_DL.ipynb`
- Harness the power of DL models for each subject and store results in CSV:\
  `train_binary_DL.py`

#### Make plots
- Process output CSVs and save visually stunning plots to file:\
  `visualize_results.ipynb`

## Saliency Plot: Visualize the Importance

Create breathtaking saliency plots based on the backpropagated value of a neural network:\
`saliency_plot_2classes.py`

## Training Classification for Single Subjects: Five Classes of Excellence

- `train_DL_singleSubject_5classes.ipynb`
- `train_DL_singleSubject_5classes_RNN.ipynb`

## Training Neural Network on All Subjects: The Ultimate Power

- `train_DL_all_subject.ipynb`
- `train_DL_all_subject_individual_first_layer.ipynb`
- `tain_DL_all_subject_individual_cross_attention.ipynb`
