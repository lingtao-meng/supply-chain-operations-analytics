.PHONY: test-fast verify

test-fast:
	python tests/test_sql_syntax.py
	python tests/test_ie_quality.py
	python tests/test_ie_scheduling.py
	python express/tests/test_hub.py
	python -m unittest discover -s manufacturing -p 'test_*.py'
	python manufacturing/verify_results.py

verify: test-fast
	bash scripts/verify_full.sh
