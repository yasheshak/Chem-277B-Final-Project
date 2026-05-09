import gdown
import os

file_urls = ["https://drive.google.com/file/d/1Jxa0AOQ30DfSVtvgvSjSbUY11fMr9xc2/view?usp=sharing", "https://drive.google.com/file/d/1B6AZlllWDzQtLmL49ctKuZqib7h7aQqA/view?usp=sharing"]
output_files = ["OMol25_data/data0000.aselmdb", "OMol25_data/data0002.aselmdb"]

os.makedirs("OMol25_data")

for file_url, output in zip(file_urls, output_files):
    if not os.path.exists(output):
        gdown.download(file_url, output, quiet = False)