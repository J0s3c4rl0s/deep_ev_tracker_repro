# Group 58 Reproducibility project: Data-driven feature tracking for event cameras

| Name | Contact | Student ID|
|----------|----------|----------|
| Jose   |     |     |
| Mitali   | m.s.patil@student.tudelft.nl    | 5934060    |
| Dean   |    |     |
| Nils   |    |    |

# Introduction

The paper "Data-Driven Feature Tracking for Event Cameras" by Nico Messikommer et al. addresses the advantages of event cameras, such as their high temporal resolution and resilience to motion blur, which make them ideal for low-latency and low-bandwidth feature tracking, especially in challenging scenarios. However, existing feature tracking methods for event cameras often require extensive parameter tuning, are sensitive to noise, and lack generalization to different scenarios. To overcome these shortcomings, the authors introduce the first data-driven feature tracker for event cameras. Leveraging low-latency events to track features detected in a grayscale frame, their approach achieves performance through a novel frame attention module, enabling information sharing across feature tracks. By transferring knowledge from synthetic to real data and employing a self-supervision strategy, their tracker outperforms existing methods in relative feature age, maintaining the lowest latency, highlighting significant advancements in event camera feature tracking. While our experiments are not supporting the findings of the authors and this reproduction is mainly focused on setting up the authors code, we outline how their method works, what we tried to reproduce as well as a division of tasks among the group.

## Introduction to the Authors method

### Introduction of Event Cameras:
Event cameras, also known as dynamic vision sensors (DVS), are sensors that detect changes in brightness (events) asynchronously, unlike traditional cameras that capture frames at fixed intervals.

### Challenges with Event Cameras:
Event cameras offer advantages like high temporal resolution and low latency, but traditional feature tracking methods designed for frame-based cameras struggle to adapt to the asynchronous nature of event data.

### Proposed Method:
Messikommer et al. propose a novel approach for feature tracking with event cameras that leverages the unique characteristics of event data. Their method is data-driven, meaning it learns directly from event data without relying on predefined features or handcrafted algorithms.

### Key Components of the Event Tracking Method:

#### Event Representation:
- Events are represented using a spatiotemporal feature representation capturing location, time, and polarity.
- This representation facilitates accurate feature tracking by providing a comprehensive understanding of each event's spatial and temporal context.

#### Feature Descriptor Learning:
- Deep neural networks learn feature descriptors directly from event data, enabling robust matching across frames.
- These descriptors encode unique characteristics of each feature, including spatial patterns, temporal dynamics, and brightness changes.

#### Temporal Consistency:
- Temporal relationships between consecutive events are leveraged to ensure feature tracking consistency over time.
- Incorporating temporal information enables the model to maintain accurate correspondences between features across frames, even in challenging scenarios.

#### Displacement Distance Computation:
- Spatial displacement of features between consecutive frames is quantified to determine feature motion and trajectory over time.
- Techniques like optical flow estimation or feature matching are employed to compute displacement distances accurately.

#### Reprojection:
- Features detected in one frame are projected onto subsequent frames to establish correspondences.
- This enables continuous tracking of features over time by associating them with their counterparts in consecutive frames.

### Network Architecture (Figure 2):

The network architecture of the proposed event tracker comprises two main components: input and feature displacement prediction, and the frame attention module for information fusion. Both components are essential for accurately tracking features in dynamic environments.

#### (a) Input and Feature Displacement Prediction:
- The event tracker receives input in the form of a reference patch \(P_0\) in a grayscale image \(I_0\) and an event patch \(P_j\) generated from an event stream \(E_j\) at timestep \(t_j\).
- Its primary objective is to predict the relative feature displacement \(\Delta \hat{f}_j\) between the reference patch and the event patch.
- Individual feature processing is handled by a feature network, which integrates a ConvLSTM layer with a state \(F\) to ensure temporal consistency.
- By leveraging a correlation map \(C_j\) derived from a template feature vector \(R_0\) of the template patch encoder and the feature map of the event patch, the feature network accurately predicts the displacement.

#### (b) Frame Attention Module for Information Fusion:
- Introducing a novel frame attention module significantly enhances tracking performance by sharing information across different feature tracks within an image.
- This module combines processed feature vectors for all tracks in the image using self-attention and a temporal state \(S\).
- Leveraging self-attention mechanisms enables the model to prioritize relevant features across different tracks, resulting in improved tracking accuracy.
- The temporal state \(S\) captures dependencies between feature tracks over time, facilitating the consideration of feature evolution.
- The fused information guides the computation of the final displacement \(\Delta \hat{f}_j\), ensuring consistent and accurate feature tracking across frames.

The  integration of individual feature processing and information fusion through the frame attention module empowers the proposed event tracker enables performance in tracking features within dynamic environments.

**Figure 2: Overview of the Event Tracker**

## Method

Feature tracking algorithms aim to track a given point in a reference frame in subsequent timesteps. They usually do this by extracting appearance information around the feature location in the reference frame, which is then matched and localized in subsequent ones. Following this pipeline, an image patch \(P_0\) in a grayscale frame for the given feature location at timestep \(t_0\) is extracted, and the feature is tracked using the asynchronous event stream. The event stream \(E_j = \{e_i\}_{i=1}^n\) between timesteps \(t_{j−1}\) and \(t_j\) consists of events \(e_i\), each encoding the pixel coordinate \(x_i\), timestamp with microsecond-level resolution \(\tau_i\), and polarity \(p_i \in \{-1, 1\}\) of the brightness change. Given the reference patch \(P_0\), the network predicts the relative feature displacement \(\Delta \hat{f}_j\) during \(t_{j−1}\) and \(t_j\) using the corresponding event stream \(E_j\) in the local neighborhood of the feature location at the previous timestep \(t_{j−1}\). The events inside the local window are converted to a dense event representation \(P_j\), specifically a maximal timestamp version of SBT where each pixel is assigned the timestamp of the most recent event. Once the network has localized the reference patch \(P_0\) inside the current event patch \(P_j\), the feature track is updated, and a new event patch \(P_{j+1}\) is extracted at the newly predicted feature location while keeping the reference patch \(P_0\). This procedure can then be iteratively repeated while accumulating the relative displacements to construct one continuous feature track. The overview of the method and the novel frame attention module are visualized in Fig. 2.

### Feature Network

To localize the template patch \(P_0\) inside the current event patch \(P_j\), the feature network first encodes both patches using separate encoders based on Feature Pyramid Networks. The resulting outputs are per-pixel feature maps for both patches that contain contextual information while keeping the spatial information. To explicitly compute the similarity measure between each pixel in the event patch and the template patch, a correlation map \(C_j\) is constructed based on the bottleneck feature vector \(R_0\) of the template patch encoder and the feature map of the event patch. Together with the correlation map \(C_j\), both feature maps are then given as input to a second feature encoder to refine the correlation map. This feature encoder consists of standard convolutions and a ConvLSTM block with a temporal cell state \(F_j\). The temporal information is crucial for predicting consistent feature tracks over time and integrating the motion information provided by the events. The output of the feature network is a single feature vector with spatial dimension \(1 \times 1\), processing each feature independently.

### Frame Attention Module

To share information between features in the same image, a novel frame attention module is introduced. Since points on a rigid body exhibit correlated motion in the image plane, there is a substantial benefit in sharing information between features across the image. The frame attention module takes the feature vectors of all patches at the current timestep \(t_j\) as input and computes the final displacement for each patch based on a self-attention weighted fusion of all feature vectors. Specifically, a state \(S\) is maintained for each feature across time to leverage the displacement prediction of the previous timesteps in the attention fusion. The temporal information facilitates the information-sharing of features with similar motion in the past, allowing the model to adaptively condition vulnerable feature tracks on similar feature tracks. Each input feature vector is individually first fused with the current state \(S_{j-1}\) using two linear layers with Leaky ReLU activations (MLP). All resulting fused features in an image are then used as key, query, and value pairs for a multi-head attention layer (MHA), which performs self-attention over each feature in an image. To facilitate training, a skip connection is introduced around the multi-head attention for each feature, adaptively weighted during training by a Layerscale layer (LS). The resulting feature vectors are then used in a simple gating layer to compute the updated state \(S_j\) based on the previous state \(S_{j-1}\), which is then processed by one linear layer to predict the final displacement.

### Supervision

#### Synthetic Supervision

The network is first trained on synthetic data from the Multiflow dataset, which contains frames, synthetically generated events, and ground truth pixel flow. A loss based on the L1 distance is directly applied for each prediction step \(j\) between the predicted and ground truth relative displacement. If the predicted feature tracks diverge beyond the template patch, such that the next feature location is not in the current search, the loss contribution is excluded. This truncated loss \(L_{rp}\) is formulated to avoid introducing noise in supervision. To reduce the gap between synthetic and real data, on-the-fly augmentation during training is applied, significantly increasing the motion distribution. Affine transformations are applied to the current event patch to obtain an augmented patch at each prediction step, enhancing the network's ability to learn geometrically robust representations.

#### Pose Supervision

A novel pose supervision loss is introduced based solely on ground truth poses of a calibrated camera to adapt the network to real events. Ground truth poses can be obtained for sparse timesteps using structure-from-motion algorithms or external motion capture systems. The supervision strategy relies on the triangulation of 3D points based on poses, applicable only in static scenes. For each predicted track, the corresponding 3D point is computed using the direct linear transform. The final pose supervision loss is constructed based on the predicted feature and the reprojected feature for each available camera pose at timestep \(t_j\), using a truncated loss to exclude loss contributions if the reprojected feature is outside of the event patch.


# Our reproduction goals

We managed to reproduce some of the results of the paper. Namely recreating the benchmark results for the fine-tuned EC dataset. For this we used the authors model checkpoint that had been fine tuned on the EC dataset as well as their already preprocessed evaluation dataset which they provided as well. This was not without its complications which will be explained in the following section. The results obtained are also explained in a later section

We also attempted to recreate the fine tuning itself, taking their model checkpoint on the synthetic dataset and fine tuning it on a (smaller) sample of the EC dataset and then evaluate it. This we did not manage to reproduce as we ran into a number of issues with the pre-processing pipeline and training script. 

# Reproducing our results

## Setting up dependencies 
Install the dependencies `pip install -r requirements.txt`. This wont work for torch, you need to supply the source for the dependency so 

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Adding large files
All large files used in this project are in this google drive file: [here](https://drive.google.com/file/d/1rDlUabCKunZ8L6IphaDkY0GUjzo6zgXW/view?usp=sharing). Simply download it and extract its contents in the root directory (i.e. just the loose files and directories, they will match the structure needed)

## Generating the inferences
1. Update all instances of <path_to_repo> in `configs.eval_real_defaults.yaml`
2. Run `python evaluate_real.py`
3. Results of inference should be in `correlation3_unscaled/timestamp/`

N.B. We provide our results so this step is not necessary.

## Benchmarking the results 
1. Move the results into `gt/network_pred/` (We provide our own results in the drive)
2. Run `python -m scripts.benchmark`
3. Results will be written to `out/benchmarking_results.csv`

## Issues Encountered 
1. During data preprocessing, some time query gives an error for being out of the pose data range. (can be fixed by removing the first entry: '0.000000' from the respective images.txt file)
2. When running train.py, the same query issue mentioned above is encountered. (ultimate issue while running train.py)
3. COLMAP instructions in the GITHUB readme of the original code were sometimes wrong , we had to refer the COLMAP documentation(we suggest the reader do the same)
4. Pre-processing steps like feature-extract were very slow and took upto an hour to run.
5. We also experienced dependency issues during the setup process. Not all dependencies listed in the requirements.txt file are functional; specifically, torch needs to be manually downloaded as per the official documentation. Additionally, certain parts of the code rely on deprecated functionality, necessitating the downgrading of dependency versions.
6. The preprocessed data for training EC is not provided. We had to download it [here](https://rpg.ifi.uzh.ch/davis_data.html) .
7. The instructions to import and export the model on COLMAP are unclear, especially the which image folder is to be imported (we imported images_corrected).
8. The intermediate files generated during the preprocessing stage are considerably large, with sizes upwards of 2 to 3 gigabytes per sequence.
9. The train.py script contained a bug that required rewriting. Our attempted fix involves ensuring that a method is called on the class itself rather than on an instance of the class, although there's a possibility that our correction might be incorrect.


## Team Contribution 
| Member | Work |
|----------|----------|
| Jose   |    | 
| Mitali    | Data pre-processing for POSE EC Dataset | 
| Dean   |    | 
| Nils    |     |
