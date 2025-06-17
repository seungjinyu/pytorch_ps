build:
	xcrun clang -O3 -shared -o libneon_abs.dylib -fPIC neon_abs.c -target arm64-apple-macos11
run:
	python resnet18_zip_v3.py
draw: 
	python draw_v3.py
drawg:
	python draw_delta_gra.py

full:
	python resnet18_zip_v3.py
	python draw_v3.py
	python draw_delta_gra.py

NUM_RUNS ?= 10

run-multi:
	@echo "Launching $(NUM_RUNS) runs of gradvsdelta.py"
	@for i in $$(seq 1 $(NUM_RUNS)); do \
		TIME=$$(date +%Y%m%d_%H%M%S); \
		echo "[Run $$i at $$TIME]"; \
		python gradvsdelta.py --run_id=$$TIME > logs/gradvsdelta_$$TIME.log 2>&1; \
		sleep 1; \
	done