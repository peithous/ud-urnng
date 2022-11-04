.PHONY=test

test:
	pylint */*.py --ignore-patterns='(.)*_test.py'
