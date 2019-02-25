# API Parameters
API_KEY = ""
BASE_URL = "https://maps.googleapis.com/maps/api/streetview?"

# Directories
TRAINING_DATA = "./data/train/facades"
RAW_DATA = "./data/raw/"
PROCESSED_DATA = "./data/processed"
MODEL = "./models"
LOGS = "./logs/"
INFERENCE_DATA_DIR ?= "./data/raw/location=43.6560811, -79.3801714"

# Location params for streetview
LATLON ?= "43.6564436,-79.3810598"

.PHONY: clean
clean:
	rm -rf $(LOGS)*.log

.PHONY: train_mask_RCNN
train_mask_RCNN:
	python3 ./src/facades_base.py train \
		--dataset $(TRAINING_DATA) \
		--model ./models \
		--logs $(LOGS) \
	&& python3 ./src/facades_h2e_a3e_1000s.py train \
		--dataset $(TRAINING_DATA) \
		--model ./models \
		--logs $(LOGS) \
	&& python3 ./src/facades_imaug_h2e_a3e_100s.py train \
		--dataset $(TRAINING_DATA) \
		--model ./models \
		--logs $(LOGS) \
	&& python3 ./src/facades_imaug_h2e_a3e_1000s.py train \
		--dataset $(TRAINING_DATA) \
		--model ./models \
		--logs $(LOGS)

.PHONY: view_stream
view_stream:
	python3 ./src/view_stream.py \
		--path $(RAW_DATA) \
		--logging_dir $(LOGS) \
		--key $(API_KEY) \
		--url $(BASE_URL) \
		--location $(LATLON)

.PHONY: inference_mask_RCNN
inference_mask_RCNN:
	python3 ./src/facades_base.py inference \
		--model ./models/mask_rcnn_facades3.h5 \
		--logs $(LOGS) \
		--dataset $(TRAINING_DATA) \
		--inference_dir ./data/interim

.PHONY:build-image
build-image:
	docker build -t streetview:gpu -f ./build/Dockerfile.DL ./build/

.PHONY: run-image
run-image:
	chmod +x ./build/run_image.sh \
	&& ./build/run_image.sh streetview:gpu ${PWD}
