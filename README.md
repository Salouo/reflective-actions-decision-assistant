# 💗 Reflective Actions Decision Assistant

## 💬 Introduction
This project aims to let Vision-Language Models (VLMs) and Large Language Models (LLMs) act as an assistant that can understand **ambiguous** user commands and proactively perform reflective actions. We leverages the reasoning power of VLMs and LLMs to predict the best actions in real-world scenarios.

The **system prompt** guiding the model is shown below:

```text
You are an assistant that, based on the user’s situation in the living room, comes up with thoughtful actions for a support robot to perform.  
A thoughtful action is one that has not been explicitly instructed but is useful to the user.  
Given the user’s situation, select all of the appropriate thoughtful actions that the robot should perform from the 40 action categories listed below.

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
## 🚀 How to use

You can use various LLMs/VLMs to simulate which action will be taken by considerate robot in the daily-life scenarios. 

**`asagi_generate.py`**: Generate reflective actions with the Asagi VLM

**`gpt_generate.py`**: Generate reflective actions with a GPT-4o VLM

**`sarashina2-vision_generate.py`**: Generate reflective actions with the Sarashina VLM   

**`llm-jp-3-instruct_generate.py`**: Generate reflective actions with the LLM-jp-3-instruct LLM 

**`extract_revised_image.py`**: Extract scene images aligned with user utterances

**`model_eval.py`**: Evaluate the performance of model by compute top-k accuracy of predicted actions

## 🪨 Limitations
1. Sometimes VLMs can not follow the output format given which makes it difficult to evaluate the models.
2. The number of actions are fixed, which means that the assistant cannot perform well at other situations
3. The information in images can not be fully utilized by VLMs, because the performance goes down when we add the \<image\> token in the prompt.


## 📢 Notification
Raw image data is not publicly available.

