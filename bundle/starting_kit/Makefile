ifndef submission
$(error submission is undefined)
endif

DATA_DIR = ../feedback_phase/input_data
REF_DIR = ../feedback_phase/reference_data
INGESTION_DIR = ../ingestion_program
SCORING_DIR = ../scoring_program

build:
	rm -Rf package
	mkdir package
	mkdir package/predictions
	mkdir package/datasets
	rsync -ar --exclude='**/test_y.npy' $(DATA_DIR)/* package/datasets/
	cp -R $(INGESTION_DIR)/ingestion.py package/ingestion.py
	cp -R $(submission)/* package

run:
	cd package; python3 ingestion.py datasets predictions .

score:
	rm -Rf scoring
	mkdir scoring
	mkdir scoring/input
	mkdir scoring/input/ref
	mkdir scoring/input/res
	mkdir scoring/output
	rsync -ar $(REF_DIR)/* scoring/input/ref/
	rsync -ar --include='*/' --include='metadata' --exclude='*' package/datasets/ scoring/input/ref/
	cp -R package/predictions/* scoring/input/res/
	cp $(SCORING_DIR)/scoring.py scoring/scoring.py
	cd scoring; python3 scoring.py input output

clean:
	rm -Rf scoring
	rm -Rf package

zip:
	cd $(submission);  zip -r ../submission.zip *

all: clean build run score