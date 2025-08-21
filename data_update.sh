cd /Users/weidian/code/algorithms/quant/qlib
SAVE_PATH=qlib_bin.tar.gz
DATA_PATH=qlib_data/cn_data

wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz

rm -rf ${DATA_PATH}
mkdir -p ${DATA_PATH}

tar -zxvf qlib_bin.tar.gz -C ${DATA_PATH} --strip-components=1
rm -f qlib_bin.tar.gz