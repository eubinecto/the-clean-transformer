
tokenizer:
	python3 main_build_tokenizer.py

kor2eng:
	python3 main_build_kor2eng.py

train:
	python3 main_train.py

eval:
	python3 main_eval.py

# pseudo-tests
test_train:
	python3 main_train.py --fast_dev_run

test_eval:
	python3 main_eval.py --fast_dev_run