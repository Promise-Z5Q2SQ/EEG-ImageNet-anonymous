# EEG-ImageNet-Dataset

This is the official repository for the paper "**EEG-ImageNet: An Electroencephalogram Dataset and Benchmarks with Image Visual Stimuli of Multi-Granularity Labels**".

<img width="776" alt="image" src="https://github.com/user-attachments/assets/55ac9916-e6ff-4f27-afbe-21a5d8206df2">

**Figure 1**: Schematic Diagram of the Data File Storage Structure. 

The dataset is available for download through the provided cloud storage(detailed information will be released after review). 

The EEG-ImageNet dataset contains a total of 63,850 + 24,000 EEG-image pairs from 16 participants, with a total of 22 sessions (i.e., 6 participants took part in two sessions each). 
Each EEG data sample has a size of (n_channels, $f_s \cdot T$), where n_channels is the number of EEG electrodes, which is 62 in our dataset; $f_s$ is the sampling frequency of the device, which is 1000 Hz in our dataset; and T is the time window size, which in our dataset is the duration of the image stimulus presentation, i.e., 0.5 seconds.
Due to ImageNet's copyright restrictions, our dataset only provides the file index of each image in ImageNet and the wnid of its category corresponding to each EEG segment.

<img width="766" alt="procedure" src="https://github.com/user-attachments/assets/992c8861-6339-4204-b8c0-519b52f65f79" />

**Figure 2**: The overall procedure of our dataset construction and benchmark design. The experimental paradigm involves four stages: S1: Category Presentation (displaying the category label), S2: Fixation (500~ms), S3: Image Presentation (each image displayed for 500~ms), and S4: an optional random test to verify participant engagement. Data flow is indicated by blue arrows, while collected data is highlighted in gray. The stimuli images are sourced from ImageNet21k, with EEG signals aligned to image indices, granularity levels, and labels. The benchmarks (image reconstruction and object classification) are designed to evaluate coarse and fine granularity classification tasks.

Our experiment consists of two stages. 
The first stage follows the same setup as [Spampinato et al., 2017], with N=50, where images from each category are presented consecutively, as shown in Figure 3(A). 
16 participants took part in this stage. 
However, as [Li et al., 2020] pointed out, the experimental results under the paradigm of [Spampinato et al., 2017] may be influenced by temporal effects when using shuffled training and test sets. 
Therefore, we conducted a second stage of the experiment, with N=30/20 and random shuffling, as shown in Figure 3(B). 6 participants participated in this phase.
Ultimately, the dataset we construct includes the EEG signals of participants exposed to each image visual stimulus in each valid session, along with the corresponding category's wnid and the image's index in ImageNet21k.

<img width="683" alt="3" src="https://github.com/user-attachments/assets/85ab3ee8-c39b-4ba0-9710-e18e2ce9f5ad" />

**Figure 3**: Comparison of different experimental paradigms. The same colors represent the same categories. Paradigms A and B are adopted in our study. Paradigm C (Random) shuffles all images and has been previously adopted in some psychological research. However, we do not apply Paradigm C in this study as this paradigm can lead to semantic overlap encoded between adjacent contents.

**Table 1**: The average results of all participants in the object classification task of Stage 1. * indicates the use of time-domain features, otherwise the use of frequency-domain features. † indicates that the difference compared to the best-performing model is significant with p-value < 0.05.

| **Model**         | **Acc (all)**    | **Acc (coarse)** | **Acc (fine)**   | **F1 (all)**     | **F1 (coarse)**  | **F1 (fine)**    |
|-------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| **Classic model** |                  |                  |                  |                  |                  |                  |
| Ridge             | 0.286±0.074†     | 0.394±0.081†     | 0.583±0.074†     | 0.261±0.070†     | 0.373±0.082†     | 0.610±0.121†     |
| KNN               | 0.304±0.086†     | 0.401±0.097†     | 0.696±0.068†     | 0.286±0.081†     | 0.380±0.096†     | 0.717±0.132†     |
| RandomForest      | 0.349±0.087†     | 0.454±0.105†     | 0.729±0.072†     | 0.323±0.083†     | 0.425±0.099†     | 0.723±0.092†     |
| SVM               | **0.392±0.086†** | **0.506±0.099†** | **0.778±0.054†** | **0.378±0.083†** | **0.486±0.105†** | **0.770±0.054†** |
| **Deep model**    |                  |                  |                  |                  |                  |                  |
| MLP               | 0.404±0.103†     | **0.534±0.115**  | **0.816±0.054**  | 0.397±0.100†     | **0.523±0.108**  | **0.819±0.053**  |
| EEGNet*           | 0.260±0.098†     | 0.303±0.108†     | 0.365±0.095†     | 0.251±0.095†     | 0.291±0.098†     | 0.374±0.102†     |
| LSTM              | 0.356±0.082†     | 0.459±0.101†     | 0.745±0.068†     | 0.347±0.076†     | 0.437±0.092†     | 0.729±0.058†     |
| Transformer       | 0.367±0.085†     | 0.460±0.108†     | 0.750±0.070†     | 0.353±0.079†     | 0.451±0.096†     | 0.761±0.075†     |
| RGNN              | **0.405±0.095**  | 0.470±0.092†     | 0.706±0.073†     | **0.401±0.098**  | 0.455±0.087†     | 0.723±0.079†     |

**Table 2**: The average results of all participants in the object classification task of Stage 2. * indicates the use of time-domain features, the use of frequency-domain features. † indicates that the difference compared to the best-performing model is significant with p-value < 0.05.

| **Model**        | **Acc (all)**       | **Acc (coarse)**    | **Acc (fine)**      | **F1 (all)**        | **F1 (coarse)**     | **F1 (fine)**       |
|-------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| **Classic model** |                     |                     |                     |                     |                     |                     |
| Ridge             | 0.182±0.053†       | 0.253±0.074†        | 0.431±0.108†        | 0.178±0.052†        | 0.243±0.075†        | 0.438±0.107†        |
| KNN               | 0.220±0.081†       | 0.310±0.113†        | 0.574±0.119†        | 0.211±0.083†        | 0.299±0.105†        | 0.565±0.134†        |
| RandomForest      | 0.268±0.101†       | 0.358±0.129†        | 0.609±0.136†        | 0.259±0.098†        | 0.341±0.117†        | 0.596±0.139†        |
| SVM               | 0.281±0.090†       | 0.368±0.107†        | 0.657±0.140†        | 0.271±0.084†        | 0.365±0.109†        | 0.648±0.134†        |
| **Deep model**    |                     |                     |                     |                     |                     |                     |
| MLP               | 0.297±0.093        | 0.395±0.110         | **0.718±0.149**     | 0.285±0.087         | **0.392±0.108**     | **0.710±0.140**     |
| EEGNet*           | 0.169±0.044†       | 0.244±0.095†        | 0.377±0.096†        | 0.160±0.041†        | 0.228±0.088†        | 0.372±0.096†        |
| RGNN              | **0.302±0.097**    | **0.401±0.105**     | 0.693±0.140†        | **0.297±0.100**     | 0.388±0.106         | 0.701±0.142†        |

<img width="776" alt="image" src="https://github.com/user-attachments/assets/026182bd-5b8d-4b84-aaca-a69ea7e2f0fa">

**Figure 3**: The image reconstruction results of a single participant (S8).


