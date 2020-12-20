from gluonnlp.data import SentencepieceTokenizer
from kogpt2.data import ReadDataset
from kogpt2.model.sample import sample_sequence
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from kogpt2.utils import download, tokenizer
from kogpt2.utils import get_tokenizer
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from datetime import datetime
import argparse
import gluonnlp
import os
import re
import subprocess
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30,
                    help="epoch 를 통해서 학습 범위를 조절합니다.")
parser.add_argument('--save_path', type=str, default='./checkpoint/',
                    help="학습 결과를 저장하는 경로입니다.")
parser.add_argument('--load_path', type=str, default='./checkpoint/KoGPT2_checkpoint_.tar',
                    help="학습된 결과를 불러오는 경로입니다.")
parser.add_argument('--sample_duration', type=int, default=500,
                    help="샘플을 확인하기 위한 주기입니다.")
parser.add_argument('--save_ckpt', type=int, default=5,
                    help="체크포인트를 저장하는 주기(depend on epoch)입니다.")
parser.add_argument('--data_file_path', type=str, default='./dataset/tokenized_sin.txt',
                    help="학습할 데이터를 불러오는 경로입니다.")
parser.add_argument('--batch_size', type=int, default=1,
                    help="batch_size 를 지정합니다.")
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


def main(epoch, save_path, load_path, sample_duration, save_ckpt, data_file_path, batch_size):
    ctx = 'cuda'
    cachedir = '~/kogpt2/'

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

    # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

    # model_path 로부터 다운로드 받은 내용을 load_state_dict 으로 업로드
    kogpt2model.load_state_dict(torch.load(model_path))

    device = torch.device(ctx)
    kogpt2model.to(device)

    # 불러오기 부분
    try:
        checkpoint = torch.load(load_path, map_location=device)

        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        kogpt2model.load_state_dict(checkpoint['model_state_dict'])

        kogpt2model.eval()
    except:
        count = 0
    else:
        count = int(re.findall("\d+", load_path)[1])

    print(f'learning count {count}')
    # 추가로 학습하기 위해 .train() 사용
    kogpt2model.train()
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

    dataset = ReadDataset(data_file_path, vocab, tok)
    print(f'Read_Dataset ok(len_{len(dataset.data)})')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    learning_rate = 1e-5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    print(f'KoGPT-2 Transfer Learning Start')
    avg_loss = (0.0, 0.0)

    for epoch in range(epoch):
        for data in data_loader:
            optimizer.zero_grad()
            data = torch.stack(data) # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
            data = data.transpose(1,0)
            data = data.to(ctx)
            model = model.to(ctx)

            outputs = model(data, labels=data)
            loss, logits = outputs[:2]
            loss = loss.to(ctx)
            loss.backward()
            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
            optimizer.step()
            if count % 10 == 0:
                print(f'epoch no.{epoch} train no.{count} loss = {loss:5f} learning_rate = {scheduler.optimizer.state_dict()["param_groups"][0]["lr"]:5f} avg_loss = {avg_loss[0]/avg_loss[1]:5f}')

                # generator 진행
                if (count > 0 and count % sample_duration == 0):
                    sent = sample_sequence(
                            model.to('cpu'),
                            tok,
                            vocab,
                            sent="나는 너를",
                            text_size=100,
                            temperature=1.3,
                            top_p_val=0.7,
                            top_k_val=0)
                    sent = sent.replace("</s>", "")
                    print(f'output {sent}')

            count += 1
            if ((count > 0) and ((count % (len(dataset.data) * save_ckpt)) == 0)):
                # 모델 저장
                now = datetime.now()
                try:
                    torch.save({
                        'epoch': epoch,
                        'train_no': count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, save_path + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_KoGPT2_checkpoint_' + str(epoch + 1) + '.tar')
                except:
                    pass

        scheduler.step()
    print(f'end')

if __name__ == "__main__":
    main(args.epoch, args.save_path, args.load_path, args.sample_duration, args.save_ckpt, args.data_file_path, args.batch_size)
