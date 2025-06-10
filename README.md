# Reflective Actions Decision Assistant

## 💬 Introduction
This project aims to let Vision-Language Models (VLMs) and Large Language Models (LLMs) act as an assistant that can understand **ambiguous** user commands and proactively perform reflective actions. We leverages the reasoning power of VLMs and LLMs to predict the best actions in real-world scenarios.

The **system prompt** guiding the model is shown below:

```text
You are a robot that selects reflective actions to assist users, based on the their ambiguous utterances.
A reflective action is one that has not been explicitly instructed but is useful to the user.  
Given the user’s situation, select all of the appropriate reflective actions that the robot should perform from the 40 action categories listed below.

All action categories can be executed in any situation.  
Also, please choose more than one category and do not select the same category more than once.

Your answer should list only the action category names and their assigned numbers, one per line.  
When you are done, output `[End of Answer]`.

Below is the list of 40 action categories:
----------
[1] Bring a banana  
[2] Bring the charging cable  
[3] Bring a cup  
[4] Bring ketchup  
[5] Bring the delivery package  
[6] Bring a plastic bottle  
[7] Bring the remote control  
[8] Bring the smartphone  
[9] Bring snacks  
[10] Bring a box of tissues  
[11] Put away the charging cable  
[12] Put away the cup  
[13] Put away ketchup  
[14] Put away the toy car  
[15] Put away the plastic bottle  
[16] Put away the remote control  
[17] Put away the smartphone  
[18] Put away snacks  
[19] Put away the box of tissues  
[20] Throw the trash into the trash can  
[21] Bring a can opener  
[22] Bring cooking sheet (parchment paper)  
[23] Bring a glass  
[24] Bring a grater  
[25] Bring kitchen paper (paper towels)  
[26] Bring a lemon  
[27] Bring olive oil  
[28] Bring a potato  
[29] Bring plastic wrap  
[30] Bring a thermos  
[31] Put the can opener on the shelf  
[32] Put the cooking sheet on the shelf  
[33] Put the glass on the shelf  
[34] Put the grater on the shelf  
[35] Put the kitchen paper on the shelf  
[36] Put the plastic bottle in the refrigerator  
[37] Put the plastic wrap on the shelf  
[38] Put the tupperware in the microwave  
[39] Put the tupperware in the refrigerator  
[40] Put the thermos on the shelf  
----------
```
## ✅ How to use

You can use various LLMs/VLMs to simulate a considerate robot. We can see which reflective actions will be taken by the considerate robot in daily-life scenarios. 

**`asagi_generate.py`**: Generate reflective actions with the Asagi VLM

**`gpt_generate.py`**: Generate reflective actions with a GPT-4o VLM

**`sarashina2-vision_generate.py`**: Generate reflective actions with the Sarashina VLM   

**`llm-jp-3-instruct_generate.py`**: Generate reflective actions with the LLM-jp-3-instruct LLM 

**`extract_revised_image.py`**: Extract scene images aligned with user utterances

**`model_eval.py`**: Evaluate the performance of model by compute top-k accuracy of predicted actions

## 🔥 Current Results
### Top-1 accuracy (%)

| Model | Anno. (%) | Anno.+Img. (%) |
|-------|-----------|----------------|
| GPT-4o | 72.4 | **76.6** |
| sarashina2-vision-14b | 51.3 | 43.4 |
| multi-hop (Yamasaki 2024) | 45.4 | — |
| GPT-4 | **74.1** | — |
| llm-jp-3-13b-instruct | 36.1 | — |

---

### Top-3 accuracy (%)

| Model | Anno. (%) | Anno.+Img. (%) |
|-------|-----------|----------------|
| GPT-4o | **81.7** | **83.1** |
| sarashina2-vision-14b | 56.6 | 48.2 |
| multi-hop (Yamasaki 2024) | 70.7 | — |
| GPT-4 | **81.7** | — |
| llm-jp-3-13b-instruct | 43.4 | — |

---

### Top-5 accuracy (%)

| Model | Anno. (%) | Anno.+Img. (%) |
|-------|-----------|----------------|
| GPT-4o | **83.9** | **85.6** |
| sarashina2-vision-14b | 60.6 | 52.4 |
| multi-hop (Yamasaki 2024) | 76.6 | — |
| GPT-4 | 83.4 | — |
| llm-jp-3-13b-instruct | 52.4 | — |

---

### Average inference time per sample (ms) – Top-1 setting

| Model | Anno. (ms) | Anno.+Img. (ms) |
|-------|------------|-----------------|
| GPT-4o | 730 | 3530 |
| sarashina2-vision-14b | 750 | 1530 |
| GPT-4 | 1555 | — |
| llm-jp-3-13b-instruct | 790 | — |

------------------------------------------------------------------
- **Unfine-tuned mid-scale VLMs struggle on downstream reasoning tasks.**  
  Without task-specific fine-tuning, many mid-scale VLMs tend to output generic image descriptions rather than follow the required response format. As a result, most VLMs are unusable on this dataset. This limitation is likely related to the short prompt lengths used during VLM training.

- **VLM text encoders can outperform size-matched LLMs in understanding image information represented in text.**  
  With annotation only, sarashina2-vision-14b outperforms the similarly sized llm-jp-3-13b-instruct, indicating that multimodal pre-training helps the text encoder capture visual cues embedded in the annotation.

- **Image utility depends on model scale.**  
  Supplying visual input boosts accuracy only for very large VLMs; for mid-scale models it even degrades performance.



## 🪨 Limitations
1. Many VLMs can not follow the output format given which makes it difficult to evaluate the models.
2. The information in images can not be fully utilized by VLMs, because the performance goes down when we add the \<image\> token in the prompt.


## 📢 Notification
Raw image data is not publicly available.

