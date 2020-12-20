# Shame

다양한 문장데이터를 `KoGPT2` 에 fine-tuning 하여 새로운 문장 생성

# Dataset

트위터 크롤링(키워드 : 고백, 고민, 죄, 우울, 슬픔, 걱정, 인공지능... 등)

# Exhibition

[Play on AI](http://www.nabi.or.kr/page/board_view.php?brd_idx=1086&brd_id=project)
(2020.12.17~2021.01.29)

# Artwork

<고백 Shame>

![ai_mockup7](https://user-images.githubusercontent.com/7092473/102714293-f0989000-4310-11eb-8259-06f9d29a9a7d.png)

# Description of the artwork

인공지능의 죄는 결국 인간의 죄다.

코사인은 이러한 대전제 하에 인간의 죄를 학습한 인공지능의 고백을 듣는 고해성사 부스 <고백>을 이번 전시에서 선보인다.

고해성사는 종교적 의미에서 자신의 죄를 고백하고 뉘우치는 행위를 의미하는데, <고백> 에서는 이 행동을 인공지능이 함으로써 인간의 죄를 역으로 환기하고 있다. 

관람객은 인공지능의 고해성사를 듣고 인공지능 기술의 윤리적 문제를 재고하며 기술이 초래한 문제에 직면하게 된다.

코사인은 이렇게 기술 발전의 어두운 면을 객관적으로 제시하고 경각심을 일깨우며, 인간 스스로 윤리적 문제를 설찰하게 하고자 한다.

# Team

**Co-Si(g)n** of art center nabi OpenLab(2020.07.18~2020.12.31)

# Team member

[김하니(작가)](https://hanikim.com)

김형규(개발자)

[김혜리(작가)](https://herry.kim)

[임태훈(개발자)](https://routiful.github.io)

[장윤영(작가)](https://yunyoung.kr)

# Reference

[KoGPT2](https://github.com/SKT-AI/KoGPT2)

[narrativeKoGPT2](https://github.com/shbictai/narrativeKoGPT2)

[KoGPT2-FineTuning](https://github.com/gyunggyung/KoGPT2-FineTuning)

[JasoAI](https://github.com/Yngie-C/JasoAI)

# Contribution

1. `lr_scheduler` 를 사용하여 epoch 가 증가할 때마다 learning_rate 를 감소시킴
1. `generator.py` 에서 checkpoint 폴더에 저장된 파라미터를 하나씩 불러올 수 있도로 하고, 이를 통해 생성된 문장을 텍스트파일로 저장


# Acknowledgement

[art center nabi](http://www.nabi.or.kr)
