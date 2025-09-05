PY=python
OUT=./figs

.PHONY: all figs clean check

all: figs

figs:
	$(PY) dcb_figs_all.py $(OUT)

check:
	@test -f $(OUT)/manifest.json || (echo "manifest.json missing"; exit 1)

clean:
	rm -rf $(OUT)
