from openai import OpenAI
from openai import AzureOpenAI
import glob
import json
import re
import jsonlines
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import heapq

def top_k_indices(arr, k):
    # 使用heapq.nlargest获取最大的k个值及其对应的下标
    # enumerate(arr)会生成一个包含索引和值的元组
    return [index for value, index in heapq.nlargest(k, ((val, idx) for idx, val in enumerate(arr)))]

def cosine_similarity(vector_a, vector_b):
    # 计算两个向量的点积
    dot_product = np.dot(vector_a, vector_b)
    
    # 计算两个向量的模（范数）
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen2-7B-Instruct")


meta_topic = ["技术","健康与保健","旅行与探险","食品与饮料","艺术与文化","科学与创新","时尚与风格","关系与约会","运动与健身","自然与环境","音乐与娱乐","政治与时事","教育与学习","金钱与金融","工作与事业","哲学与道德","历史与怀旧","社交媒体与交流","创造力与灵感","个人成长与发展","灵性与信仰","流行文化与趋势","美容与自我保健","家庭与育儿","创业与商业","文学与写作","游戏与技术","正念与冥想","多样性与包容","旅游与文化交流"]


def chat_qwen(prompt,model,tokenizer,device):
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response

embed = SentenceTransformer("/home/hpr/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
query = f"请扮演xx角色"
json_files = glob.glob('/home/hpr/Roleplay/*.json')
documents = []
for json_file in json_files:
    # 打开并加载每个.json文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        documents.append(data)
        
#print(documents)



def embedding_query(model,query,documents,top_k=1):
    model.max_seq_length = 8192
    query_embeddings = model.encode(query, prompt_name="query")
    document_embeddings = model.encode(documents)
    #scores = (query_embeddings @ document_embeddings.T) * 100
    scores = cosine_similarity(query_embeddings,document_embeddings.T)
    scores = scores.tolist()
    indexes = top_k_indices(scores,top_k)
    
    print(scores)
    return scores , indexes

#scores, indexes = embedding_query(embed,query,documents,top_k=1)

#for index in indexes:
#    print(documents[index])



###加载role profile
def open_profile(profile_path):
    with open(profile_path,"r",encoding="utf-8") as r:
        profile = r.read()
        profile = json.loads(profile)
    return profile

profile = open_profile("/home/hpr/Roleplay/wukong_2.json")

###选择多个topics
def choose_topics(num_topics):
    ###生成多个topics的prompt
    prompt = f"""
    从"待选主题"中选择与"角色介绍"匹配的{num_topics}个主题。
    请只输出这些主题。
    "角色介绍":
    {profile}
    "待选主题":    
    {meta_topic}
    """
    topics = chat_qwen(prompt,model,tokenizer,device)
    topics = topics.strip().split("\n")
    topics = [re.sub(r'^\d+\.\s*', '', topic) for topic in topics]
    return topics


def generate_question():
    topics = choose_topics(1)
    prompt = f"""
    根据"角色介绍",以{profile["Name"]}的角度, 从"主题"中的每个话题生成一个相关的讨论话题。
    请输出少于 50 个单词。

    "角色介绍":
    {profile}

    '主题':
    {topics}
    """
    print(topics)
    question = chat_qwen(prompt,model,tokenizer,device)
    return question


conversation_requirements = f"""
###以下是对剧情对话的要求：

每轮对话必须引入新的情节发展、突发事件或场景变化，揭示新的情感冲突、情节转折和关系发展，通过连续的戏剧情节吸引玩家的兴趣。
设计开放式的故事情节，避免引导'{profile["Name"]}'和'用户'之间的和解或理想化场景，不要终结话题，避免最后的情节点或话题。



###"用户"的对话要求:
根据角色"{profile["Name"]}"的内容进行进一步的对话。
避免重复性的对话，以及括号内容


###"{profile["Name"]}"的对话要求:
{profile["Name"]}的对话应简洁、清晰、朴实，避免生硬、陈词滥调或华而不实的词汇，避免激励性、泛泛而谈或空洞的陈词滥调，听起来不像是在读稿子。
{profile["Name"]}的说话方式应该与对应的'角色介绍'相互匹配，需要有对话的情感深度和真实感。
{profile["Name"]}的发言应包括对话内容和描述性语言，在括号中描述{profile["Name"]}的肢体互动或情绪反应，丰富场景的主要细节或过渡技巧，以增强故事的视觉效果和身临其境的感觉。
每个句子仅限一个括号描述，避免过多的括号打断对话。
"""

question = generate_question()



def generate_qa(out_path,question,profile,num_dialogues):
    with open(out_path,"a",encoding="utf-8") as w:
        writer = jsonlines.Writer(w)
        history = []
        prompt1 = f"""
        请扮演'人物简介'中的角色和"用户"，根据'讨论的话题'，以第一人称的视角互相开始一次对话。
        
        使用 JSON 格式(别忘了逗号),如：
        {{"role":'{profile["Name"]}',"content":""}},
        {{"role":"用户","content":""}},

        '讨论的话题':
        {question}

        '人物简介':
        {profile}

        '对话要求':
        {conversation_requirements}
        """
        dialogue = chat_qwen(prompt1,model,tokenizer,device)
        history.append(dialogue)
        

        for k in range(num_dialogues):####多轮对话的轮次
            
            
            prompt2 = f"""
            请扮演'人物简介'中的角色和"用户"，以第一人称的视角基于'对话信息'延续1次 "{profile["Name"]}"与"用户"对话, 请不要重复'对话信息'的'content'。
            
            使用 JSON 格式(别忘了逗号),如：
            {{"role":'{profile["Name"]}',"content":""}},
            {{"role":"用户","content":""}},


            '人物简介':
            {profile}

            '对话信息':
            {history[-1]}

            '对话要求':
            {conversation_requirements}

            """
            
            dialogue = chat_qwen(prompt2,model,tokenizer,device)
            history.append(dialogue)

        prompt3 = f"""
        请扮演'人物简介'中的角色，以第一人称的视角根据'对话信息'，结束对话
        
        使用 JSON 格式(别忘了逗号),如：
        {{"role":{profile["Name"]},"content":""}}

        '人物简介':
        {profile}

        '对话信息':
        {history[-1]}

        '对话要求':
        {conversation_requirements}
        """
        dialogue = chat_qwen(prompt3,model,tokenizer,device)
        history.append(dialogue)

        writer.write({"npc_name":profile["Name"],"question":question.strip(),"answer":history})
    writer.close()
    
    
    
generate_qa("/home/hpr/Roleplay/wukong2.jsonl",question,profile,num_dialogues=10)

