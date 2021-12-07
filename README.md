# dekorde 
트랜스포머 밑바닥부터 구현 -  수업  &amp;  Sesame's Pirates

## 강의노트
- [WEEK7: Transformer - why?](https://www.notion.so/WEEK7-Transformer-why-8e3712fb674a4ba2a85bf6da9cd36af0)
- [WEEK8: Transformer: How does Self-attention work?](https://www.notion.so/WEEK8-Transformer-How-does-Self-attention-work-e02fc6b942f64b2ba82ce7e35afc817d)
- [WEEK9: How does Multihead Self attention work?](https://www.notion.so/WEEK9-How-does-Multihead-Self-attention-work-cddce1ae09eb4b0fb067a2474cbf8515)
- [WEEK9: How does Residual Connection & Layer normalisation work?](https://www.notion.so/WEEK9-How-does-Residual-Connection-Layer-normalisation-work-b4a41db45a014378bc1c4a0f6da3757e)
- [WEEK9: How does Positional Encoding  work?](https://www.notion.so/WEEK9-How-does-Positional-Encoding-work-0d0e5b9d17464af08f39b4977c073beb)


## TO-DO's
- [ ] inference 구현하기
  - 여기 참고: https://github.com/bkoch4142/attention-is-all-you-need-paper/blob/0542f33ef9330b1850a27fcd5071e4d8acfdbfba/src/architectures/machine_translation_transformer.py#L75-L90
- [ ] ignore_index 파라미터 사용하기.
- [ ] 사전학습된 토크나이저를 사용하지 말고, 직접 토크나이저를 학습하기
  - 여기 참고: https://github.com/bkoch4142/attention-is-all-you-need-paper/blob/master/src/tokenizer.py
- [ ] keras_preprocessing 사용하는 대신에, 그냥 torch.utils만을 사용하기. 이것만으로도 충분히 쓸만함.
- [ ] pytorch-lightning &  wandb & torchmetrics 적용하기!
- [ ] 모델이 학습 데이터에서 조차 converge를 하지 않을때는? -> dropout 조심하기!! 
