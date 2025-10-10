Project Setup Instructions

Before running any scripts, complete the following steps:
1. Create a Python environment:
    Create the environment from the YAML file:  `conda env create -f environment.yaml`

2. Prepare the LIVECell dataset
    Navigate to the directory `livecell_dataset_prep` and follow the steps described in `instructions.txt`.
    This will guide you through downloading and organizing the dataset correctly.

3. Download pretrained models for benchmark
    1. Anchor-free model
        - go to https://github.com/sartorius-research/LIVECell/tree/main/model
        - Download the anchor_free model trained on the LIVECell dataset.
        - Place the downloaded `.pth` file inside `Cell_counter/benchmark_models`, in the same folder as its corresponding configuration file.
    2. LACSS model
        - Go to https://github.com/jiyuuchc/lacss
        - Download the weights for the small-2dL model trained on the LIVECell dataset.
        - Place the weights file inside `Cell_counter/benchmark_models` as well.
    3. Clone CenterMask2
        - Clone the CenterMask2 repository into our project directory: `git clone https://github.com/youngwanLEE/centermask2.git`

✅ After completing these steps, you will be ready to run the project scripts.
💡 If anything in these instructions is unclear, please feel free to contact me.