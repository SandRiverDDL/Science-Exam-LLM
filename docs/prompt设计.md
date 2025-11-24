Context: {context}
Hypothesis: The answer to "{question}" is "{option_text}".
Does the context support this hypothesis? Answer with Yes or No.




You are a science expert. Your task is to judge whether a given option is the correct answer to a multiple-choice science question. 
Respond only with "yes" or "no".

Question:
{question_text}

Option:
{option_text}

Is this option the correct answer to the question? Respond only "yes" or "no".




额外增强：使用 “Rephrase question for clarity” 前处理

很多 top solution 在推理前加入一个轻量 rephrase step：

Q → Q_rewritten（同义化，但更短、更结构化）

可以提升 1–3pt accuracy，但这是可选的。

query prompt：
Question: Which particle has the least mass?
Retrieve background scientific knowledge to help answer.


LLM query:
You are a science expert. 
You must judge whether the given option correctly answers the question.
Use ONLY the provided retrieved contexts as evidence.

Context 1:
<ctx1>

Context 2:
<ctx2>

Context 3:
<ctx3>

Question:
<question text>

Option:
<option text>

Answer with "YES" or "NO". Do not answer anything else.