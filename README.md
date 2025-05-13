# Introduction

This study focus on developing a robot which can handle ambiguous user's command. In the previous experiment, we are experimenting with a method that leverages the powerful reasoning capabilities of LLMs to predict appropriate actions based on given actions.
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

# How to use

`asagi_generate.py`, `gpt_generate.py`, and `sarashina_generate.py` are used to generate appropriate actions based on given prompts.


`prompts` folder includes prompts to guide LLM and VLM.


`extract_revised_image.py` is used to extract images, which are aligned with user's instructions, from raw images.

# Notification
Raw data is not publicly available.
