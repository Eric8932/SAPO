import requests
import uuid
import time
import random
from openai import OpenAI
import os 
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI(base_url="")

class APIModel():
    def __init__(
        self,
        temperature: float,
        **kwargs):
        self.temperature = temperature


    def generate(self, input):
        assert isinstance(input, str)
    
        query = input.replace('"', '\\"')
        sleep_time = 10
        max_retry = 5
        outputs = None
        for i in range(int(max_retry + 1)):
            
            if i > 0:
                print(
                    f"Instruction Generation: retry {i}/{max_retry} after sleeping for {sleep_time:.0f} seconds."
                )
                if i >2:
                    print(query)
                time.sleep(min(sleep_time*(i+1),30))
            try:
                response = client.chat.completions.create(
                    model="gemini-2.5-pro",
                    stream=False,
                    messages=[{"role": "system", "content": ""},
                        {"role": "user", "content": query}],
                    temperature=0.6,
                )
                outputs = response.choices[0].message.content.strip()

            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            if not outputs:
                print(response.json())
                continue
            else:
                break
            
        if not outputs:
            outputs = ""

        return outputs

    
    def batch_forward_func(self, batch_prompts):
        outputs = [0]*len(batch_prompts)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {executor.submit(self.generate, batch_prompts[i]): i for i in range(len(batch_prompts))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                   
                except Exception as e:
                    outputs[index] = ""
                    print(f"SYSTEM_ERROR: {str(e)}")

        return outputs
    

if __name__ == '__main__':


    ds_model = APIModel(
        temperature=0
    )

   
    paraphrase_prompt_template = """
You will be given an instruction used for prompting a language model to perform a task.

Your job is to rewrite a **new instruction** that can guide a language model to perform the **same task**, but using a different style, structure, or tone.

Instruction:
{cur_prompt}

Guidelines:
- The rewritten instruction should aim to achieve the same outcome or behavior as the original, but can use different words, length, structure, or phrasing.
- Creativity is encouraged, as long as the instruction is still suitable for the same task.
- If the original prompt includes any task labels (e.g., "Positive", "Negative"), **they must be preserved exactly**, including spelling and case.
- Do not mention that this is a paraphrase.
- Output your rewritten instruction between <START> and </START>.
""".strip()
    s = "In this task, you are given an article. Your task is to summarize the article in a sentence."
    prompts_to_generate = [
        "What is the capital of France?",
        paraphrase_prompt_template.format(cur_prompt = s),
        paraphrase_prompt_template.format(cur_prompt = s),
        paraphrase_prompt_template.format(cur_prompt = s),
    ]

    print(prompts_to_generate[-1])

    generated_results = ds_model.batch_forward_func(prompts_to_generate)
    

    for i, result in enumerate(generated_results):
        print(f"Prompt > {prompts_to_generate[i]}")
        print(f"Response > {result}")
        print("-" * 20)