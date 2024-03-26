<img width="842" alt="image" src="https://github.com/sroecker/rhods-birdwatching/assets/362733/531af5ec-2f00-46b3-b40d-2c06991f8235">

# Simple image classification using fast.ai

Either download data from Kaggle directly and extract it to this folder:
https://www.kaggle.com/datasets/gpiosenka/100-bird-species
or run ``00_setup.ipynb``

Run ``01_quick_train.ipynb`` to fine-tune a ResNet34 model with the provided data.
Run ``02_to_onnx.ipynb`` to convert the fine-tuned model to ONNX which can be deployed on [RHOAI](https://ai-on-openshift.io/getting-started/openshift-ai/) using OpenVINO.

Adapt the PREDICTION URL in ``streamlit/app.py`` and deploy the streamlit app for drag and drop scoring as shown in the screenshot.

ONNX conversion taken from: https://github.com/tkeyo/fastai-onnx
See this nice blog post for details: https://dev.to/tkeyo/export-fastai-resnet-models-to-onnx-2gj7
