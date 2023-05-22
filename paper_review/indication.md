# Indication as Prior Knowledge for Multimodal Disease Classification in Chest Radiographs with Transformers  

<blockquote><b>Author</b>: Grzegorz Jacenków, Alison Q. O'Neil, Sotirios A. Tsaftaris<br>
<b>Comments</b>: Accepted at the IEEE International Symposium on Biomedical Imaging (ISBI) 2022 as an oral presentation<br>
<b>Subject</b>: Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG)<br>
<i>Submitted on 12 Feb 2022</i></blockquote><br> 

## 본 논문을 읽게된 이유  
<br>
MIMIC-CXR 즉, 기존 MIMIC에서 제공하는 임상데이터와 연결되는 환자의 흉부 방사선 사진 데이터를 얻을 수 있었기 때문에, 이를 살펴보고 있었다. 또한 특정 환자가 왜 흉부 방사선 사진을 찍었는지 그 특정 질병이 궁금했고, 이 데이터를 가지고 어떤 연구가 이루어지는 지 궁금했다. 따라서 MIMIC-CXR을 이용하여 실험을 진행한 논문을 찾아보게 되었고, 처음엔 <a href="https://link.springer.com/chapter/10.1007/978-3-030-60946-7_11#chapter-info">Towards Automated Diagnosis with Attentive Multi-modal Learning Using Electronic Health Records and Chest X-Rays</a>을 읽으려 했으나, 유로 논문이어서 다음 기회에 읽기로 했다. 따라서 이 논문을 인용한 본 논문을 읽게 되었다.(본 논문에서는 위 논문의 제안 모델과 성능을 비교했다.)<br>
<br>

## Background  
<br>
<p>indication(적응증)이란 어떤 약이나 수술로 치료 효과를 볼 수 있을 것이라 생각되는 질환이나 증세로, 특정 검사나 치료를 시행해야하는 합당한 이유를 말한다. indication은 간혹 diagnosis와 혼동되기도 한다. diagnosis는 특정 질환에 대한 평가이지만, indication은 그런 검사를 하는 <b>이유</b>를 말한다.<a href="https://ko.wikipedia.org/wiki/%EC%A0%81%EC%9D%91%EC%A6%9D">(출처)</a></p><br>

<p>Multimodal(다중양식)이란 Verbal, Vocal, Visual 처럼 다양한 데이터의 형태를 의미한다. 더불어, Multimodal Learning이란 인간의 "5가지 감각기관"과 같이 Multimodal로부터 다양한 정보를 처리하고 연결시키는 모델을 만들어서, 인간의 인지적 학습방법과 같이 세계를 이해하는 학습 방법이다.<a href="https://datascience0321.tistory.com/31">(출처)</a></p><br>

<p>의료진은 특정 환자의 X-ray를 촬영할 것을 방사선의에게 요청할 때, Scan request라는 것을 제출하는데, 여기에는 촬영을 요청하는 이유(의심되는 질병, 과거력 등)가 적혀있다. 이러한 이유들을 <b>"Indication Field"</b>라고 부른다. (the motivation for the patient's screening examination)</p><br>

<p>흉부 방사선 사진은 가장 흔한 영상 진찰 중 하나이다.(폐렴, 암, 심지어 COVID-19)</p>

<br>

## 기존의 CDSS(임상 의사결정 지원 시스템)의 문제점  
최근 (2019, 2020) CDSS를 위해 제안된 두 논문<a href="https://arxiv.org/abs/2008.10418">[2]</a>,<a href="https://arxiv.org/abs/1907.12330">[3]</a>은 다음 두 가지를 고려하지 않았음  
 - mostly focus on a single modality (e.g. patient’s X-ray)  
 - do not take into account complementary information which might be already available in a hospital’s database (e.g. patient’s clinical history)  

=> indication field를 활용하는 것을 제안, 이 field에는 환자의 과거력, 촬영 요청 이유 등 여러 단서들이 포함되어 있으므로, 모델링에 활용하면 더욱 좋은 성능을 보일 것이다.  

=> 따라서 본 논문에서 저자는 text based side information을 통한 Vision-and-language model을 설계하여 높은 정확도의 disease classification을 제안하고 있다. 

## Chest X-Ray Classification  
<br>

흉부 방사선 사진을 분류하는 대부분의 선행 연구는 ResNet-50을 사용하였다.  

[1]“Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification,” Scientific Reports  
[10]K. He et al., “Deep Residual Learning for Image Recognition,” in IEEE CVPR, I. M. Baltruschat et al.  

위 논문들은 환자의 인구통계학적 정보와 Chest x-ray를 함께 활용하는 multimodal 연구였으나, multimodal fusion을 하는 방식이 Early fusion 즉, final classification을 하기 전에 단순히 concat을 했다는 점과 단순히 인구통계학적 정보는 분류 과제에 도움이 되지 않음을 지적하고 있었다.

저자는 attention 기법이 적용된 BERT로 두 modal의 fusion을 구현하고, text는 indication field 정보를 활용하면 분류에 큰 도움을 줄 것이라 가정했다.  


## Learning with Radiology Reports  
<br>
[11] TieNet 에서는 classification, reprot generation task를 위한 image-text pairs의 embedding을 생성하는 연구를 진행했다. attention with CNN, RNN을 사용한 것이 특징이지만, 여기서 사용한 Text는 방사선의가 사진을 본 후 기록한 소견이므로, <b>방사선의의 추가적인 개입으로 데이터가 수집되어야 한다.</b> -> <i>높은 비용이 수반되기 때문에 단점의 뉘앙스로 기록을 한 것이라 추측한다.</i>  
<br>
<br>
<p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/text%EC%A0%95%EB%B3%B4.png?raw=true" width="70%"></p>
<br>

<p align ="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig1.jpg?raw=true" width ="70%"></p>

<br>

## Method  
<br>
image 형태인 patient's X-ray 와 side information인 text 형태의 indication 정보를 multimodal fusion하는 형식의 모델  
<br>
<br>
<blockquote>*multimodal fusion이란?<br>
논문 "Baltrušaitis, Tadas, Chaitanya Ahuja, and Louis-Philippe Morency. "Multimodal machine learning: A survey and taxonomy."에서 정의한 multimodal machine learning 기법 중 하나로, 예측 단계(회귀, 분류)에서 다양한 modality의 정보를 결합하는 것이다.<br>
multimodal fusion은 모델은 더욱 robust하게 만들어주며, Unimodality에서 자체적으로 볼 수 없는 보충적인 정보를 얻을 수 있다. multimodal fusion은 크게 2가지 접근 방식이 있으며 자세한 내용은 <a href="https://datascience0321.tistory.com/31">여기</a>를 참고할 수 있다. 본 논문에서 사용한 multimodal fusion은 model based approach로, Neural network 기반의 fusion 방식이다. </blockquote>  
<br>
<p align='center'><b>Multimodal fusion - model agnostic approaches</b></p>
<p align='center'><img src ="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbu1WWf%2Fbtr0hd0Z0lJ%2Fg5CaLDSVC8w4XW1OMWSbI0%2Fimg.png" width="70%"></p>
<br>
<p align='center'><b>Multimodal fusion - model based approaches</b></p>
<p align='center'><img src ="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqXoS2%2Fbtr0nDj4EJ8%2FNXldEl2QMrLevOzZHkyKi0%2Fimg.png" width="70%"></p>

<br>

논문에서 밝히기를, Vision-and-language model을 구현하기 위해 text modal로 pre-trained된 BERT를 가져와서 text side information(indication)과 image feature로 fine tune을 진행하는 것이 기본 아이디어이다.  

<hr>

### <i>현재 visual-and-language task SOTA 모델</i>  
<br>

현재 visual and language task의 SOTA 모델은 <a href="https://arxiv.org/pdf/1908.08530v4.pdf">VL-BERT: PRE-TRAINING OF GENERIC VISUALLINGUISTIC REPRESENTATIONS</a>논문에서 소개한 Visual BERT이다. 

<p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/visual%20bert.jpg?raw=true" width="70%"></p>

이 모델은 처음에 image-caption pair가 있고, 이를 BERT를 활용하여 훈련시킨 것이 가장 큰 특징이다. 이때 image는 interest 영역에 bbox 좌표 정보가 있는 데이터셋이다. 따라서 이 모델은 Object detection을 수행하여 ROI를 얻은 다음, 이를 caption과 align하여 BERT구조에 입력해야하는 것 같다.  

<br>

<p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/SOTA%20VLBERT.jpg?raw=true" width="70%"></p>

<br>

Visual BERT는 방대한 양의 데이터로 pre train이 이루어진다. biomedical community에는 아직 transformer model을 pre-train하기 위한 general한 multimodal dataset이 부족한 상태이기 때문에 위 모델처럼 pre-train을 하는 것이 한계가 있다고 언급하고 있다.<br>
<br>
저자는 이러한 문제에 접근하기 위해, 이미 존재하는 pre-trained unimodal model을 가져와 multimodal BERT(MMBT)처럼 fine tune하는 방식을 채택했다. MMBT는 ResNet을 이용하여 생성된 feature map을 token으로 처리하여 입력하는 방식이다. MMBT의 모델 구조는 다음과 같다.

<p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/mmbt.jpg?raw=true" width="70%"></p>

<br>

MMBT의 BERT는 English Wikipedia로 사전 학습된 모델(12-layer 768 dimensional)을 사용했으며, Image encoder인 ResNet은 ImageNet 데이터로 사전 학습된 모델을 사용했다.  


### Model Architecture

본 논문에서는 MMBT의 구조를 이용하여 모델링했다고 밝히고 있다. 

따라서 여기까지의 정보를 종합해 본 논문에서 했던 모델링은 다음과 같다.

1. 768 차원의 입력 시퀀스를 받는 pre-trained BERT를 가져온다.  
2. image는 ResNet-50을 사용하여 fine tune을 진행한다. 이때, ResNet의 출력차원인 7x7x2048의 feature map을 1x1x49 차원으로 reshape(1x1 conv를 사용한 것으로 생각된다.)  
3. indication 정보는 가져온 텍스트 임베딩 값과 2에서 진행한 이미지 임베딩 값을 concat 한 후, positional embedding, segment embedding을 추가하여 input을 구성한다.
4. indication 정보와, image embedding 정보를 concat하여 BERT의 입력으로 사용한다.  
5. 최종적으로 두 모달을 통해 disease multiclass classification을 수행하기 위해, 768-768-14의 multi-layer perceptron과 GELU 활설함수를 사용한다.



<br>
<p align="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig2.jpg?raw=true" width="70%"></p>

<br>

### Loss Fucntion  

<br>

We optimise a binary <b>cross-entropy loss</b> with <b>class weighting</b> that is inversely proportional to the number of examples in the training set.


<br>

## Experiment  
<br>

### Dataset   
<br>

 - We use the MIMIC-CXR dataset
 - The dataset consists of 377,110 chest X-ray images associated with 227,835 *postscreening
reports(촬영 과정 이후 작성된 보고서) of 65,379 patients(taken at the BIDMC(Beth Israel Deaconess Medical Center Emergency Department.))
    - 총 65,379명의 환자, 227,835 부의 reports, 377,110개의 chest x-ray  

 - 각 환자의 x-ray 사진은 정면, 후면, 측면으로 이루어져 있음 -> 본 논문에서는 frontial image 즉 정면 이미지를 활용  
 <br>
 <p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig3.jpg?raw=true" width="35%"></p>  
 <br>

 - reports는 "indication" or "history" field를 사용  


<b><i>최종 사용할 데이터셋 (final evaluation)</i></b>  
<blockquote>
- 210,538 studies  <br>
- training (205,923) <br> 
- validation (1695)  <br>
- test sets (2920)  <br>

</blockquote><br>

### Labeling  

<br>

- The original data are not labelled for the classification
task.  
- We use the <a href="https://arxiv.org/abs/1901.07031">CheXpert Labeler</a> to extract fourteen labels from full radiology reports
- atelectasis, cardiomegaly, consolidation, edema, enlarged cardiomediastinum, fracture, lung lesion, lung opacity, no finding, pleural effusion, pleural (other), pneumonia, pneumothorax, and support devices. 
- We set the task as a multilabel problem with positive-vs-rest classification(CheXpert labeler를 사용하면 각 label에 대해서 positive, negative, uncertatine, no mention으로 4가지 중 1가지 값으로 할당되는데, 본 논문에서는 오직 positive만 고려하겠다는 것. 즉 각 label은 positive 혹은 rest로 이분법적으로 분류되는 상황)

<i>*CheXpert Labeler</i>  
 - The CheXpert task is to predict the probability of 14 different observations from multi-view chest radiographs.  
 - <b>Mention Extraction</b>: The labeler extracts mentions from a list of observations from the Impression section of radiology reports, which summarizes the key findings in the radiographic study.  
 - <b>Mention Classification</b>: After extracting mentions of observations, the aim is to classify them as negative (“no evidence of pulmonary edema, pleural effusions or pneumothorax”), uncertain (“diffuse reticular pattern may represent mild interstitial pulmonary edema”), or positive (“moderate bilateral effusions and bibasilar opacities”). The ‘uncertain’ label can capture both the uncertainty of a radiologist in the diagnosis as well as ambiguity inherent in the report (“heart size is stable”).  
 - <b>Mention Aggregation</b>: The classification for each mention of observations comes to arrive at a final label for 14 observations that consist of 12 pathologies as well as the “Support Devices” and “No Finding” observations, as shown in the table above.  
<br>
 <p align='center'><img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*ZtEaNahlteTHK1wPaHORDw.png"></p>  

 <b>왜 하필 14가지 관찰?</b>  

1. No Finding(병변 없음)  
2. Enlarged Cardiom(심장 확대증)  
3. Cardiomegaly(심근비대증)  
4. Lung Opacity(폐 투명도 감소)  
5. Lung Lesion(폐 병변)  
6. Edema(부종)  
7. Consolidation(응고)   
8. Pneumonia(폐렴)   
9. Atelectasis(폐 축소증)  
10. Pneumothorax(기흉)   
11. Pleural Effusion(흉수)   
12. Pleural Other(기타 흉막 질환)  
13. Fracture(골절)   
14. Support Devices(의료기기 사용)  

자주 등장하는 병변, 진단이어서 14가지로 정했다고 추측했다.  

David McClosky의 biomedical model은 자연어 처리를 이용한 의학 정보 추출을 위한 기계 학습 모델이다.  
이 모델은 흉부 방사선 보고서와 같은 의학적인 텍스트에서 형태소, 구문, 의미 분석을 통해 관련된 정보를 추출할 수 있다.  
마지막으로 각 문장 별 universal dependency graph를 그리기 위해 Stanford CoreNLP를 사용했다. => 여기서 14가지로 labeling이 될 것으로 판단  
<br>
<p align ="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig4.jpg?raw=true"></p>  

<br>

### Pre-processing  

 - Image resize(224x224)
 - Normalise the image to zero mean and unit of standard deviation  
 - The text input has been stripped from special character("__", "/")  

### Baselines  

 - CheXpert Labeler: We apply this method to the indication fields.  
 - BERT: We use the unimodal BERT network which is the backbone of the proposed method with no access to the imaging input.(the same classification head to fine-tune the network for classification.)
 - ResNet-50: We use the ResNet-50 network pretrained on ImageNet(fine-tune to classify the chest radiographs.)  
 - <a href="https://link.springer.com/chapter/10.1007/978-3-030-60946-7_11">Attentive</a>: multimodal model(ResNet-50, BioWord2Vec, attention mechanism)  
    - 논문명: Towards Automated Diagnosis with Attentive Multi-modal Learning Using Electronic Health Records and Chest X-Rays
<br>
<p align = 'center'><img src="https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-60946-7_11/MediaObjects/505976_1_En_11_Fig1_HTML.png"></p>

<br>

### Experiment Setup  

- 14 epochs with a batch size of 128  
- Adam optimiser with weight decay(0.01)  
- learning rate $5\times10^{-5}$  
- multi-label micro $F_{1}$ score evaluated on the validation set 
    - micro f1 = micro-precision = micro-recall = accuracy  
<p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig7.jpg?raw=true" width="70%"></p>  


<br>

### Results: Classification Performance  
<br>

<p align ="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/table1.jpg?raw=true" width="90%"></p> 

<br>

 - We observe the CheXpert Labeler has the weakest performance across all of the reported metrics.  
    - The method is a rule-based approach, so it cannot learn associations between the content of indication fields and the labels(text 기반 명시적 언급 정보 파악)

 - BERT outperform the labeler in all metrics (+53.3% improvement in micro AUROC)  
 - image-only based classifier(ResNet-50) outperforms the BERT in all metric(except recall with macro)  
    - text 정보에 의한 clinician’s suspicion 에 비해 이미지가 더 많은 정보를 포함하고 있음을 알 수 있음  
 - Attentive which uses both image, text, outperforms the image and text only method in all reported metrics with micro AUROC improved by 1.9% comparing to the best unimodal baseline.(middle fusion)  

 - multimodal BERT(proposed) outperforms all unimodal and multimodal baselines with 2% margin.(ealry fusion)  

Attentive와 다르게 MMBT는 저 수준에서 modal fusion을 했지만 modal 사이의 상호, 상관 관계를 더욱 잘 학습할 수 있었다.  
<br>
<p align="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig5.jpg?raw=true" width="60%"></p>
<br>

### Result: Robustness to Textual Input  
<br>

- 과다한 진료로 인해 의사들은 imaging examination을 작성하는 동안 typographical error(오타)를 만들 수 있다.  
- 저자는 일반적인 실수와 동의어 사용과 같은 텍스트 입력의 변경에 대한 모델의 강건성에 대한 주요 성능 메트릭과 함께 모델을 평가했다.
- 이를 위해, MMBT 모델을 텍스트 변경으로 text 했다.
<br>


<blockquote>
<br>
- <b>Character Swap:</b> swapping two consecutive characters random <br>
e.g. fever → fevre<br>
- <b>Keyboard Typo:</b> selecting a random character and replacing with an adjacent one<br>
e.g. fever → f3ver<br>
- <b>Synonyms:</b> selecting a synonym for a given biomedical term using the UMLS database<br> 
e.g. fever → pyrexia(열, 발열)<br>
- <b>Missing Field/Stop Words Noise:</b>replacing the indication field with an empty string or a sentence using only stop words<br>
- <b>Indication Swap:</b>selecting a random indication from another patient ensuring no single positive class is shared between two patients(즉, label이 다른 indication으로 replace)<br>
<br>
</blockquote>
<br>

<p align='center'><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig6.jpg?raw=true" width="90%"></p>
<br>



## Review  
  
  - *본 논문을 한마디로 정의하면*  
    - indication 이라는 "사전지식" Text modal과 환자의 흉부 X-Ray Image modal을 활용한 multimodal disease multiclass classification 모델 제안  
<br>

  - *본 논문이 가장 크게 기여한 부분*    
    - 환자의 사전 지식을 활용하여 더욱 정확한 이미지 진단이 가능함을 증명한 것     
    - 의사의 과로로 인한 실수가 있더라도, 모델이 강건함을 보여준 실험이 인상적이었다.    
    - 이미지를 보고 질병을 분류하는 공모전 같은 대회들은 진단이 완료된 시점에서 수집된 데이터이기 때문에 실용성이 없었다. 하지만 본 논문을 통해 진단이 이루어지기 전의 정보만을 활용하여 질병 예측을 하기 때문에 의사의 임상 업무를 도울 수 있는 실용적인 모델이라는 생각을 했다.    
<br>

  - *본 논문에서 아쉬운 것*  
    - f1 score는 precision과 recall에 동일한 가중치를 두고 조화 평균을 계산한다는 단점이 있음    
    - 의료, 헬스케어 분야에서는 특히 건강한 사람을 아픈 사람으로 분류하는 것과 아픈 사람을 건강한 사람으로 분류하는 것은 서로 다른 위험 비용이 발생할 수 있음    
    - multi class의 경우 예측 오류마다 의미가 더욱 다를것임(x를 y로 예측 하는 것과 z를 w로 예측하는 것은 서로 다른 비용이 발생)    
    - 이러한 f1 score는 본 논문의 도메인 지식을 고려하지 않은 평가지표로 생각됨   
    - 이미지 전처리에 적은 노력: 이미지 데이터를 단순히 resize한 것에서 전처리가 끝났다. X-ray 데이터를 더욱 선명히 처리하거(CLAHE), 샘플 수를 보완하고, 다양한 특징을 학습하도록 augmentation이 추가되었다면 더욱 좋았을 것 같다.    
    

<br>

  - *본 논문과 관련된 본인의 아이디어*      
    - 평가지표 개선: 실제로는 건강한 사람에게 병이 있다고 판단하는 제 1종 오류와 실제로 병이 있는데 건강하다고 판단하는 제 2종 오류를 고려하여 weight를 바꿔가면서 실험했으면 더 좋았을것 같다.  
     - 1종 오류의 경우 추가적인 검사, 치료, 수술 등이 필요하게 되어 환자에게 불필요한 위험과 비용을 초래할 수 있기 때문이다.      
     - 2종 오류의 경우 병을 발견하지 못하고 치료를 지연시키는 등의 문제가 발생할 수 있기 때문이다.    
     - precision이 높다-> False Positive가 낮다. 모델이 양성으로 예측한 샘플이 실제 양성일 확률이 높아진다.-> 1 종 오류가 줄어든다.    
     - recall이 높다 -> False Negative가 낮다. -> True Negative가 높다 -> 모델이 예측한 샘플이 음성을 때 실제 음성일 확률이 높아진다. -> 2종 오류가 줄어든다.    
     - 따라서 $F_{beta}$score를 사용하여 beta를 0부터 1까지 조정하면서 실험 결과를 보여주면 더 좋을 것 같다.    
<br>        
<p align="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig10.jpg?raw=true"></p>
<br>   
<p align="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig9.jpg?raw=true"></p>     

 -  
     - 이미지 전처리 CLAHE(Contrast Limited Adaptive Histogram Equlization) 추가하기    
      - CLAHE는 "타일"이라는 bounding box내의 픽셀 값들을 균일화하여 픽셀 분포를 원만하게 변환한다.  
      - X-ray같은 어두운 이미지를 더욱 선명하게 해주는 효과를 가져올 수 있다.    
<br>
<p align="center"><img src="https://github.com/Jeong-Eul/Data-Mining-Study/blob/main/Paper/Indication/fig8.jpg?raw=true
"></p>  






