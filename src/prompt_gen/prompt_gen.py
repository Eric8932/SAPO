from .API import APIModel


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


paraphrase_prompt_template_zh = """
您将收到一条用于引导语言模型执行任务的指令。

您的任务是重写一条**新的指令**，使其能够引导语言模型执行**相同的任务**，但使用不同的风格、结构或语气。

指令：
{cur_prompt}

指南：
- 重写的指令应旨在达到与原指令相同的结果或行为，但可以使用不同的词语、长度、结构或措辞。
- 鼓励发挥创造力，只要指令仍然适用于相同的任务即可。
- 如果原指令包含任何任务标签（例如，“肯定”、“否定”），**必须完全保留**，包括拼写和大小写。
- 请勿提及这是改写。
- 将重写的指令输出在 <START> 和 </START> 之间。
""".strip()

def _clean_optim_response(optim_response):
    start_tags = ['<START>']
    end_tags = ['</START>', '<END>', '</END>']

    end_index = 1000000
    for start_tag in start_tags:
        start_index = optim_response.find(start_tag)
        if start_index!=-1:
            break
    start_tag ="" if start_index==-1 else start_tag
    start_index = 0 if start_index==-1 else start_index
    
    for end_tag in end_tags:
        end_index_temp = optim_response.find(end_tag, start_index)
        if end_index_temp !=-1 and end_index_temp<end_index:
            end_index = end_index_temp
    
    if end_index==100000:
        content = optim_response[start_index + len(start_tag):].strip()
    else:
        content = optim_response[start_index + len(start_tag):end_index].strip()
    return [content]


def prompt_gen(cur_prompt = "",number_new_prompts = 20):

    assert len(cur_prompt)>0
    optim_model = APIModel(0.6)
    if "态度" in cur_prompt:
        optimize_prompt = paraphrase_prompt_template_zh.format(cur_prompt=cur_prompt)
    else:
        optimize_prompt = paraphrase_prompt_template.format(cur_prompt=cur_prompt)
        
     
    response = optim_model.batch_forward_func([optimize_prompt]*number_new_prompts )
    optimized_prompts = [_clean_optim_response(res)[0].strip() for res in response]

    return optimized_prompts