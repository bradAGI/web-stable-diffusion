#!/bin/bash
set -euxo pipefail

CONFIG_PATH=${CONFIG:-${1:-web/local-config.json}}
SERVE_PORT=${PORT:-8889}

scripts/build_site.sh "${CONFIG_PATH}"

echo "symlink parameter location to site.."

mkdir -p site/_site
ln -sfn "$(pwd)/dist/params" site/_site/web-sd-shards-v1-5
ln -sfn "$(pwd)/dist/params_xl" site/_site/web-sd-shards-xl
cd site && jekyll serve --skip-initial-build --host localhost --port "${SERVE_PORT}"
