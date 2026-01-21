# MTRP
Our dataset and model have been released at huggingface: https://huggingface.co/holdhold/MTRPQwen-7B

##1.Run role profile
'''bash
python role_profile.py
'''

role_eng_path is role data file.

role_en_outpath is output role profile path.

2.Run Multi_Turn_Dialog.py output/multi_turn_dialog/Gaston.jsonl output/profile/role_profile_Gaston.json

Multi_Turn_ch.py is used to generate chinese role in multi-turn dialogue data.
