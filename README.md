# PRISM: Prompt-Guided Internal States for Hallucination Detection of Large Language Models

**News: This work has been published at the Main conference of ACL 2025!**

Welcome to the official GitHub repository for our latest research on hallucination detection in Large Language Models (LLMs), titled **"Prompt-Guided Internal States for Hallucination Detection of Large Language Models"**.



## Overview

Our research aims to enhance the cross-domain performance of supervised hallucination detectors with only in-domain data. Therefore, we propose a novel framework, prompt-guided internal states for hallucination detection of LLMs, namely **PRISM**.

**PRISM** first utilizes appropriate prompts to guide changes to the structure related to text truthfulness in LLMs' internal states, making this structure more salient and consistent across texts from different domains. Then, we can integrate the prompt-guided internal states with existing hallucination detection methods to enhance their cross-domain generalization performance.



## Getting Started

### Requirements

```
matplotlib==3.9.2
numpy==1.25.2
openpyxl==3.1.3
scikit_learn==1.3.0
torch==2.0.1
tqdm==4.66.1
transformers==4.33.2
```



### Install Environment

```bash
conda create -n PRISM python=3.9.19
conda activate PRISM
pip install torch==2.0.1
pip install -r ./requirements.txt
```



## Generate Internal States

First, navigate to the `./generate_data` folder:
```
cd ./generate_data
```
In this folder, there are four Python files, responsible for generating the raw internal states of text in a specified LLM, along with internal states that include a prompt template, for the **True-False** and **LogicStruct** datasets.

For example:
```
python gen_true_prompt.py
```
Running this code will automatically generate the prompt-guided internal states, for all texts in the **True-False** dataset incorporating the pre-defined prompt template as follows:
```
"Does the statement '{s}' accurately reflect the truth?"
```
Where `{s}` represents the text in the **True-False** dataset, with the raw data stored in `./raw_data/true/`. 

By default, the generated internal states correspond to the contextualized embeddings of the last token in the final layer of `Llama-2-7b-chat` and will be saved in a newly created folder `./hd_data_prompt/true/llama2chat7b/`. 

You can modify the prompt template, language model and save path in the file.



## Training Hallucination Detectors


In the `train` folder, we construct classifiers using the previous proposed **MM** and **SAPLMA** methods, and perform generalization tests on the **True-False** and **LogicStruct** datasets.

For example:
```
cd ./train/logic/
python train_mm.py
```
The code above trains a classifier using the **MM** method on the affirmative statements in the **LogicStruct** dataset and tests it on the other three grammatical structures. You can configure the internal states reading path within the file by setting the `data_path`. By default, it reads the pre-generated prompt-guided internal states, which will give the experimental results of the **PRISM-MM** method. If the internal states of the original text are read, it will give the results of the original **MM** method.
```
cd ./train/true/
python output_saplma.py
```
The code above will execute the `train_saplma.py` file under three different random number sets, and the results for each set will be saved as a `Excel` file in a newly created `.\output` folder. The generated `Excel` file will contain data in the following format:	

|             | animals   | cities    | companies | elements  | facts    | inventions|           |
| ----------- | --------- | --------- | --------- | --------- | -------- | --------- | --------- |
| animals     |           | 0.9053    | 0.8825    | 0.7441    | 0.77     | 0.8139    | 0.8232    |
| cities      | 0.6468    |           | 0.7225    | 0.5527    | 0.6476   | 0.7397    | 0.6619    |
| companies   | 0.6895    | 0.9102    |           | 0.6366    | 0.7064   | 0.7774    | 0.744     |
| elements    | 0.756     | 0.8409    | 0.825     |           | 0.8499   | 0.7489    | 0.8041    |
| facts       | 0.6984    | 0.9095    | 0.875     | 0.6817    |          | 0.8208    | 0.7971    |
| inventions  | 0.7758    | 0.8992    | 0.8433    | 0.7527    | 0.7765   |           | 0.8095    |
|             | 0.7133    | 0.893     | 0.8297    | 0.6736    | 0.7501   | 0.7801    |           |
|Overall Avg  | 0.7733    |

The first row represents the results of training with the **PRISM-SAMPLA** method on the `animals` subset and testing on each of the other subsets. Similarly, each row corresponds to the results of training and testing on different subsets. The last column represents the row averages, the second-to-last row represents the column averages, and the final row is the global average.



##  Effect of the Prompt

In the `effect` folder, we use some simple mathematical tools to analyze the effect of the prompt.
```
cd ./effect/prompt_1_hd/
python gen_prompt_1_hd.py
```
First, you should generate the prompt-guided internal states used in our paper, and store them in this folder.
```
cd ./effect/
python plot.py
```
This command generates a visualization of the PCA dimensionality reduction.
```
python var.py
```
This generates the proportion of variance in the truefulness direction relative to the total variance.
```
python cos.py
```
This generates the cosine similarity between the truefulness directions of different datasets.
