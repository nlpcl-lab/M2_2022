# M2_2022

## environment
1. conda create -n m2 python=3.8
2. pip install pandas portalocker sklearn spacy matplotlib wandb torchtext einops sacrebleu
python -m spacy download en

3. (gpu10) conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
4. pip install pytorch-lightning transformers

## 신뢰도 증강기
### 설명
신뢰도 증강기는 입력 문장에 대해 신뢰도를 높일 수 있는 선호 키워드를 바탕으로 증강된 문장를 생성하는 소프트웨어임.
커뮤니티 단위의 신뢰도 증강을 도모하고자, 주어진 문장를 증강 후 해당 문장에 대한 커뮤니티간 신뢰도 분포 양극화가 감소하는 경우 
증강이 성공한 것으로 간주함.
따라서 주어진 문장에 대한 커뮤니티간 신뢰도 분포의 평균 또는 표준편차 차이가 감소하는 경우 신뢰도 증강이 성공한 것으로 하여 성공 비율을
산출하며, 감소한 신뢰도 평균값 차이와 감소한 표준편차 값의 평균을 계산해 기존 입력 문장 대비 증강된 문장의 커뮤니티 단위 신뢰도 증가율을 산출함.

### 시스템 사양
시험성적서 참조

### 운영SW 및 시험도구
Ubuntu 20.04.3 LTS
python 3.8.13
Conda 4.12.0
numpy 1.23.2
pandas 1.4.3
scikit-learn 1.1.2

### 시험 절차
1. 시험데이터 확인
   1. 내용: 대상 문장과 사용자 통계자료(json file)
      1. total_docs.json
   2. 데이터 수: print(len: # of data)
      1. main.py line 121-126: 10만개 문서에 대해 8:1:1 비율로 학습/검증/테스트
2. 소스코드 확인
   1. 데이터 입력
      1. main.py line 120-149
   2. 문서 증강 모델 학습
      1. main.py line all
   3. 신뢰도 증강 성공률과 신뢰도 증가율 산출 출력
      1. evaluate.py line 43-55, 86-87
   
## experiments
문서 증강 모델 학습: python main.py

문서 증강 모델 평가: python evaluate.py
