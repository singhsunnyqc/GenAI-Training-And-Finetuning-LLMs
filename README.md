# LLM Training and Finetuning

## Overview
Explore different ways to train and finetune an opensource foundation model from HuggingFace. In this problem, we train and finetune models that can automatically identify emotional states such as "anger" or "joy" that people express about your company's product on Twitter.

To do this, we are going follow the given steps:
1. Pulling in the emotions data from the Hugging Face Hub begin exploring it using basic EDA methods.
2. Afterwards, we'll gain some familiarity with Pytorch to tokenize the data and begin understanding the intricies of tokenization.
3. Next we'll gain some familiarty with using LLM hidden states to predict sentiment. In this way we are going to use the output of the model to directly see how it can influence the emotion prediction.
4. Lastly, we are going to fine tune a language model from scratch on the sentiment data as well as perform fine tuning using a LoRA implementation.


### Train Model using Feature extraction
Using a transformer as a feature extractor is fairly simple. We freeze the body's weights during training and use the hidden states as features for the classifier. The advantage of this approach is that we can quickly train a small or shallow model. Such a model could be a neural classification layer or a method that does not rely on gradients, such as a random forest. This method is especially convenient if GPUs are unavailable, since the hidden states only need to be precomputed once.

### Fine-tuning using LoRA (Low-Rank Adaptation) technique
We train the whole model end-to-end, which also updates the parameters of the pretrained model.

### Performance assessment
We use techniques like F1-Score, Accuracy Score and Confusion Matrix to evaluate the model performance.


<img width="639" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/4ab96afe-e5aa-41e7-bd2a-2146bffe0e6b" />
