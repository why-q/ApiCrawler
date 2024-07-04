## 准备工作

```bash
pip install Scrapy
```

## 运行实例

```bash
scrapy crawl pexels
scrapy crawl unsplash
scrapy crawl huaban
```

Pexels 和 Unsplash 需要首先注册一个开发者账户申请 API KEY 后填入，修改参数只需要修改 `api_crawler/config/base_config.py` 即可。具体参数请参考代码文件下的详细注释。最后得到的是一个包含图片链接和图片 ID 的 txt 文件，之后可以使用另外一个库的 `download_image.py` 下载即可。