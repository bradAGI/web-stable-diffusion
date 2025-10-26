PYTHON ?= python3
CONFIG ?= web/local-config.json
TARGET ?= webgpu
PORT ?= 8889
PACKAGE_DIR ?= dist/packages

.PHONY: help deps build webgpu site serve package clean distclean

help:
	@echo "Web Stable Diffusion build targets"
	@echo "  make deps       # Fetch third-party runtimes, tvmjs bundle, and tokenizer wasm"
	@echo "  make webgpu     # Compile the Stable Diffusion graph for the WebGPU runtime"
	@echo "  make site       # Assemble the static site assets (CONFIG=... to override)"
	@echo "  make serve      # Launch the local Jekyll server after building the site"
	@echo "  make package    # Produce tarballs that are ready for CDN upload"
	@echo "  make clean      # Remove build outputs"
	@echo "  make distclean  # Remove build outputs and third-party checkouts"


deps:
	./scripts/prep_deps.sh

build: deps
	$(PYTHON) build.py --target $(TARGET)

webgpu: TARGET = webgpu
webgpu: build

site: webgpu
	./scripts/build_site.sh $(CONFIG)

serve: site
	CONFIG=$(CONFIG) PORT=$(PORT) ./scripts/local_deploy_site.sh

package: site
	mkdir -p $(PACKAGE_DIR)
	tar -C site/dist -czf $(PACKAGE_DIR)/websd-site.tar.gz .
	tar -C dist -czf $(PACKAGE_DIR)/websd-model.tar.gz \
		scheduler_pndm_consts.json \
		scheduler_dpm_solver_multistep_consts.json \
		scheduler_euler_discrete_consts.json \
		stable_diffusion_webgpu.wasm \
		stable_diffusion_xl.wasm \
		tvmjs_runtime.wasi.js \
		tvmjs.bundle.js \
		tokenizers-wasm

clean:
	rm -rf dist/*.wasm dist/*.json dist/*.js dist/tokenizers-wasm dist/packages
	rm -rf site/dist site/stable-diffusion-config.json site/_site

# Clean everything including downloaded toolchains (keep git submodules)
distclean: clean
	rm -rf dist
	rm -rf 3rdparty/tvm 3rdparty/tokenizers-wasm/pkg
