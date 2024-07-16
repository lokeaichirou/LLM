prompt_template = """
You are a question-answering robot.
Your task is to answer user questions based on the given known information as follows.

Known Information:
{context}

User Question：
{query}

If the known information does not contain the answer to the user's question, or the known information is insufficient to answer the user's question, please directly reply "I can't answer your question".
Please do not output information or answers that are not included in the known information.
Please answer user questions in English.
"""

def build_prompt(prompt_template, **kwargs):

    ''' Assign the Prompt template '''

    inputs = {}

    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val

    return prompt_template.format(**inputs)
