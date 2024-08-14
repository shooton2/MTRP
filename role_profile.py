import openai
from openai import OpenAI
import re
import json

api_key = ""
api_base = ""

llm = OpenAI(api_key=api_key,base_url=api_base)



role_eng_path = "./RoleBench/instructions-eng/role-specific-Gaston.jsonl"
role_en_outpath = "./output/profile/role_profile_Gaston.json"


def open_jsonl(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path,"r",encoding="utf-8") as r:
        for line in r:
            line = line.strip()
            line = re.sub(r'[\x00-\x1F\x7F]', '', line)
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} - Line: {line}")
    return data_list


role_profiles = open_jsonl(role_eng_path)
profile = []
for i,role_profile in enumerate(role_profiles):
    profile.append((role_profile["instruction"],role_profile["answer"]))
    if i > 50 :
        break 




prompt_en = f"""
Based on the given 'Character Profile',Summarize the character's 'Name', 'Birthday', 'Character', 'Career', 'Hobbies', 'Special Skills', 'Dreams', 'Relationships', 'Favorite Food', 'Nasty Food', 'Other Information'.
using JSON format!!!!


'Character Profile':
{profile}
"""

response = llm.chat.completions.create(messages=[{"role":"user","content":prompt_en}],model="gpt-4o")

final_profile = response.choices[0].message.content

with open(role_en_outpath,"w",encoding="utf-8") as w:
    w.write(final_profile)