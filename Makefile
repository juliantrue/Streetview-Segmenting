# API Parameters
API_KEY = "AIzaSyCkJ1m-ymoFC7hQYdTlROa0bLfiT-EqW9Y"
BASE_URL = "https://maps.googleapis.com/maps/api/streetview?"

# Directories
RAW_DATA = "./data/raw/"
INTRIM_DATA = "./data/interim/"
LOGS = "./logs/"
INFERENCE_DATA_DIR ?= "./data/raw/location=43.6560811, -79.3801714"
LATLON ?= "43.6564436,-79.3810598"

.PHONY: clean
clean:
	rm -f $(INTRIM_DATA)*
	rm -f $(LOGS)*.log

.PHONY: inference_mask_RCNN
inference_mask_RCNN:
	python3 ./src/Detectron/tools/infer_simple.py \
    --cfg ./src/Detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir $(INTRIM_DATA) \
    --image-ext png \
		--output-ext png \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
		$(INFERENCE_DATA_DIR)



.PHONY: view_stream
view_stream:
	python3 ./view_stream.py \
		--path $(RAW_DATA) \
		--logging_dir $(LOGS) \
		--key $(API_KEY) \
		--url $(BASE_URL) \
		--location $(LATLON)
