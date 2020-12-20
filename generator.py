from datetime import datetime
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.model.sample import sample_sequence
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from kogpt2.utils import download, tokenizer
from kogpt2.utils import get_tokenizer

import argparse
import gluonnlp
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=1.3,
                    help="temperature 를 통해서 글의 창의성을 조절합니다.")
parser.add_argument('--top_p', type=float, default=0.7,
                    help="top_p 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--top_k', type=int, default=0,
                    help="top_k 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--text_size', type=int, default=100,
                    help="결과물의 길이를 조정합니다.")
parser.add_argument('--loops', type=int, default=10,
                    help="글을 몇 번 반복할지 지정합니다. -1은 무한반복입니다.")
parser.add_argument('--sent', type=str, default="나빠",
                    help="글의 시작 문장입니다.")
parser.add_argument('--load_path', type=str, default="./checkpoint/",
                    help="학습된 결과물을 저장하는 경로입니다.")

args = parser.parse_args()

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}


def main(temperature = 0.7, top_p = 0.8, top_k = 40, sent = "", text_size = 100, loops = 0, load_path = ""):
    ctx = 'cuda'
    cachedir = '~/kogpt2/'
    save_path = './checkpoint/'
    # download model
    model_info = pytorch_kogpt2
    model_path = download(model_info['url'],
                          model_info['fname'],
                          model_info['chksum'],
                          cachedir=cachedir)
    # download vocab
    vocab_info = tokenizer
    vocab_path = download(vocab_info['url'],
                          vocab_info['fname'],
                          vocab_info['chksum'],
                          cachedir=cachedir)
    # Device 설정
    device = torch.device(ctx)
    # 저장한 Checkpoint 불러오기
    file_list = os.listdir(load_path)
    for checkpoint_file in file_list:
        checkpoint = torch.load(load_path + checkpoint_file, map_location=device)

        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        kogpt2model.load_state_dict(checkpoint['model_state_dict'])

        kogpt2model.eval()
        vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                                  mask_token=None,
                                                                  sep_token=None,
                                                                  cls_token=None,
                                                                  unknown_token='<unk>',
                                                                  padding_token='<pad>',
                                                                  bos_token='<s>',
                                                                  eos_token='</s>')

        tok_path = get_tokenizer()
        model, vocab = kogpt2model, vocab_b_obj
        tok = SentencepieceTokenizer(tok_path)

        print("Loads checkpoint : ", load_path + checkpoint_file)


        now = datetime.now()
        f = open('./samples/' + str(now.year) + str(now.month) + str(now.day) + '_' + sent + '.txt', 'a', encoding="utf-8")

        head = [load_path + checkpoint_file, text_size, temperature, top_p, top_k]
        head = [str(h) for h in head]
        f.write(",".join(head))
        f.write(",")
        f.write(sent)
        f.write("\n")

        input = sent
        for _ in range(1, loops):
            toked = tok(input)

            input = sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p, top_k)
            input = input.replace("</s>", "")
            print(input)
            print('\n')

            f.write(input)

        f.close()

    print('end')


if __name__ == "__main__":
    # execute only if run as a script
    main(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, sent=args.sent, text_size=args.text_size, loops=args.loops+1, load_path=args.load_path)
