import os
import random
def get_ext_files(wav_folder, is_train=False, ext='wav'):
    validation_files = []
    for root, dirs, files in os.walk(wav_folder):
        for file in files:
            if file.endswith(f'.{ext}'):
                validation_files.append(os.path.join(root, file))
    return validation_files

# 파일 목록 가져오기
files = get_ext_files("/mnt/bear2/zipped_datasets/korean_conversation_age/002.자유대화(노인남여)/01.데이터/1.Training")

# txt 파일에 한 줄씩 저장
with open("노인남여_Training_file_list.txt", "w") as f:
    for file in files:
        f.write(file + "\n")
