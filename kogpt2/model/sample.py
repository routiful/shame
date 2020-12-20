import torch
import torch.nn.functional as F
import random
import kss
import re

def top_p(logits, vocab, threshold = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    indexs = sorted_indices.tolist()

    sorted_softmax_logits = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum( sorted_softmax_logits, dim=-1)


    sorted_indices_to_remove = cum_probs > threshold
    top_p_index = 0

    # Top-p에 해당하는 index를 획득
    for i in range(len(sorted_indices_to_remove)):
      if sorted_indices_to_remove[i]== True:
        top_p_index = 0 if i==0 else i-1
        break

    # for i in range(top_p_index):
      # print('gen '+str(i)+': '+vocab.to_tokens(indexs[i]))

    rand_num = random.randint(0, top_p_index) # top-p 분포에서 랜덤 샘플링
    top_p_sample_num = indexs[rand_num]
    gen_word = vocab.to_tokens(top_p_sample_num)
    # print('selected token: '+gen_word+ ' softmax value:'+str(sorted_softmax_logits[rand_num]))

    return gen_word

def top_k(predict, vocab, k):
  # topk 중 랜덤으로 선택된 값을 반환.
  gen = []

  probs, indexs = torch.topk(predict, k=k,dim=-1)
  # probs = probs.squeeze().tolist()[-1]
  # indexs = indexs.squeeze().tolist()[-1]
  probs = probs.tolist()
  indexs = indexs.tolist()

  for i in range(len(indexs)):
    gen.append((vocab.to_tokens(indexs[i]), probs[i]))
  # print('topk word and value: ', gen)

  rand_num = random.randint(0, k - 1)
  gen_word = vocab.to_tokens(indexs[rand_num])

  return gen_word

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p_val, top_k_val):
    ctx = 'cuda'
    device = torch.device(ctx)

    toked = tok(sent) # 받은 문장
    count = 0
    generated_text = ''

    if len(toked) > 1024:
        return 0

    # while 1:
        # input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
        # predicts = model(input_ids)
        # pred = predicts[0]

        # last_pred = pred.squeeze()[-1]
        # # top_p 샘플링 방법
        # # sampling.py를 통해 random, top-k, top-p 선택 가능.
        # gen = top_p(last_pred, vocab, top_p_val)
        # # gen = top_k(last_pred, vocab, 5)

        # if count > text_size:
            # sent += gen.replace('▁', ' ')
            # toked = tok(sent)
            # count = 0
            # break

        # sent += gen.replace('▁', ' ')
        # toked = tok(sent)
        # count += 1
        # sent = re.sub(r'(<s>|</s>)', '' , ''.join(sent).replace('▁', ' ').strip())

    # return sent

    while 1:  # 이부분도 적절하게 바꾸기.
        # 시작 토큰 넣기
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)

        input_ids = input_ids.to(ctx)
        model = model.to(ctx)

        predicts = model(input_ids)
        pred = predicts[0]

        # temperature 적용
        logits = pred
        logits = logits[:, -1, :] / temperature
        # top k
        logits = top_k_logits(logits, top_k_val)
        # top p
        logits = top_p_logits(logits, top_p=top_p_val)

        #logits = logits.to(ctx)

        # 확률적을 뽑고
        log_probs = F.softmax(logits, dim=-1)
        # 이전 것들 저장해서 다음 학습에 사용
        prev = torch.multinomial(log_probs, num_samples=1)
        # 결과 나오게 (사전에서 gpt2가 뽑은 결과)
        gen = vocab.to_tokens(prev.squeeze().tolist())

        # 끝나면 본격적으로 만들어 놓기.
        if gen == '</s>' or count > text_size:
            # print(count)
            # print('to_tokens:', vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist()))
            #print(sent)
            sent += gen.replace('▁', ' ')
            generated_text += gen.replace('▁', ' ')
            sent += '\n'
            generated_text += '\n'
            toked = tok(sent)
            count = 0
            break

        sent += gen.replace('▁', ' ')
        generated_text += gen.replace('▁', ' ')
        toked = tok(sent)
        sent = re.sub(r'(<s>|</s>)', '' , ''.join(sent).replace('▁', ' ').strip())
        count += 1
    return sent
