# Pexels
PEXELS_KEY = "vjSYVv9dg3bRVIsXWq3qakwtPmYDpeNGIYB2SCAX103naOeTaFqimsBk"  # API KEY
PEXELS_QUERY = "group photo"  # 搜索关键字
PEXELS_IMAGE_DIR = "./datas/imgs/pexels"
PEXELS_IMAGE_URL_TXT_DIR = "./datas/docs/pexels"  # 图片链接保存文件夹
PEXELS_LOG_PATH = "./logs/pexels.log"  # 日志
PEXELS_PAGES = "1"  # 爬取范围，如果填写单个数字如 “5” 则爬取从第一页到第五页的内容，也可以填入 "1-5" 可以起到同样的效果（注意不要超过 API 限制）
PEXELS_PER_PAGE = 100  # 每一页的图片数
PEXELS_IMAGE_TYPE = "original"  # 图片的质量规格，分为 ["original", "large", "large2x", "medium", "small", "portrait", "landscape", "tiny"]


# Unsplash
UNSPLASH_KEY = "bJyoYNddqpf7zmInZgoHJBKhURohYbWaiykyG1oy_Gg"
UNSPLASH_QUERY = "group photo"
UNSPLASH_IMAGE_DIR = "./datas/imgs/unsplash"
UNSPLASH_IMAGE_URL_TXT_DIR = "./datas/docs/unsplash"
UNSPLASH_LOG_PATH = "./logs/unsplash.log"
UNSPLASH_PAGES = "1"
UNSPLASH_PER_PAGE = 100
UNSPLASH_IMAGE_TYPE = "raw"  # 图片的质量规格，分为 ["raw", "full", "regular", "small", "thumb"]


# Huaban
HUABAN_QUERY = "合照"
HUABAN_IMAGE_DIR = "./datas/imgs/huaban"
HUABAN_IMAGE_URL_TXT_DIR = "./datas/docs/huaban"
HUABAN_LOG_PATH = "./logs/huaban.log"
HUABAN_PAGES = "1-100"
HUABAN_PER_PAGE = 100
