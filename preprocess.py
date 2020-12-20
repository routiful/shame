from kogpt2.data import sentencePieceTokenizer, toString
import sys, os
import re
import kss

def divideSentence():
    file_name = 'twitter_ai'
    load_path = '/home/user/shame/dataset/data/' + file_name + '.txt'
    save_file = open('/home/user/shame/dataset/data/' + file_name + '_1.txt', 'w', encoding='utf-8')

    file = open(load_path, 'r', encoding='utf-8')
    while True:
        line = file.readline()

        if not line:
            file.close()
            break

        for sent in kss.split_sentences(line):
            sent = _clean_str(sent)
            save_file.write(sent + '\n')

    save_file.close()

def _clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)

    return text

def makeDataFile():
    load_path = '/home/user/shame/dataset/data/'
    file_list = os.listdir(load_path)
    data_files = [file for file in file_list if file.endswith(".txt")]

    save_file = open('/home/user/shame/dataset/sin.txt', 'w', encoding='utf-8')

    for data_file in data_files:
        file = open(load_path + data_file, 'r', encoding='utf-8')
        while True:
            line = file.readline()

            if not line:
                file.close()
                break

            if line == '\n':
                continue

            line = _clean_str(line)
            save_file.write(line)

    save_file.close()


def makeDataUnderMaxTokenLen(max_token_len):
    # tokenizer
    sentencepieceTokenizer= sentencePieceTokenizer()

    # Files for read and write
    file_name = 'sin.txt'
    file = open('./dataset/' + file_name, 'r', encoding='utf-8')
    untokenized_file = open('./dataset/untokenized_' + file_name, 'w', encoding='utf-8')
    tokenized_file = open('./dataset/tokenized_' + file_name, 'w', encoding='utf-8')

    # Data for saving that will use on training
    untokenized = ""
    tokenized = ""
    data_length = 0

    # Preprocess datas
    while True:
        line = file.readline()

        if not line:
            break

        tokenized_line = sentencepieceTokenizer(line)

        # Data length for writing has to under 1022
        # input data can get 1024 token
        # but we need to use BOS and EOS token
        if data_length + len(tokenized_line) + 2 >= max_token_len: # bos와 eos 토큰 갯수 고려 +2
            untokenized_file.write(untokenized + '\n')
            tokenized_file.write(tokenized + '\n')

            untokenized = ""
            tokenized = ""
            data_length = 0

        untokenized = untokenized + "<s>" + line[:-1] + "</s>"
        tokenized = tokenized + "<s>" + toString(tokenized_line) + "</s>"

        data_length = data_length + len(tokenized_line) + 2 # bos와 eos 토큰 갯수 고려 +2

    file.close()
    untokenized_file.close()
    tokenized_file.close()


if __name__ == "__main__":
    # execute only if run as a script
    # divideSentence()
    makeDataFile()
    makeDataUnderMaxTokenLen(1024)
