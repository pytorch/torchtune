
pip install ${PYTORCH_PIP_PREFIX} torchtune --index-url ${PYTORCH_PIP_DOWNLOAD_URL}

python  ./tests/smoke_tests.py
