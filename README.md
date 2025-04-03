# dental-cavity-segmentation


```
conda create -n dental python=3.12 -y
conda activate dental
pip install -r requirements.txt
```



### code1
- fine tune Yolo model for the segmentation ( more that 150 images trained )
- how to run 
```
streamlit run yolo_seg_app.py
```

### code 2
- based on the Yolo detection model and Segment anything model (SAM from meta) 
- how to run
```
cd Code2/backend
streamlit run app.py
```

### code 3
- based on detectron2 ( there is some issue with requirements.txt, got error when try to download the requirements.txt file of detectron)
- so i trained a fine-tuned model on google colab and results are very good
- link of the colab https://colab.research.google.com/drive/1F-coc0gpBvKYGrg7MKXOstvERFhjPaTo?usp=sharing