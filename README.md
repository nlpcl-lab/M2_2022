# M2_2022

## environment
1. conda create -n m2 python=3.8
2. pip install pandas portalocker opencv-python sklearn spacy matplotlib wandb torchtext einops
python -m spacy download en

3. (nlpgpu10) conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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
   2. 데이터 수: print(len: # of data)
2. 소스코드 확인
   1. 데이터 입력
   2. 시험 데이터 샘플링
   3. 입력 문장에 대해 증강된 문장 생성
   4. 신뢰도 증강 성공률과 신뢰도 증가율 산출 출력
3. 스크립트 실행
   1. 입력 문장에 대해 증강된 문장을 생성
   2. 입력 문장과 증강된 문장에 대한 신뢰도 및 신뢰도 표준편차를 계산
   3. 신뢰도 증강 성공률과 신뢰도 증가율 산출 및 출력
      1. 신뢰도 증강 성공률 = 
      2. 신뢰도 증가율 =
4. 결과확인
   1. 스크립트 실행 결과 출력된 로그로 신뢰도 증강 성공률과 신뢰도 증가율 확인
   
## experiments
