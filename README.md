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

--- 2024.9.11 更新 ---

添加 Freepik、iStock、gettyimages、Adobe Stock 和 shutterstock 五个网站的爬虫接口，目前全部只能爬取预览图片，需要额外安装 Selenium 库并下载对应自己电脑的 [ChromeDriver](https://developer.chrome.com/docs/chromedriver/downloads) 到文件夹 `driver/` 中。Freepik 由于采取了 lazy load 的机制，所以需要使用 Selenium 并等待加载过程，故爬取速度较慢（20s/page）。

```bash
pip install selenium
```

同样需要修改配置文件，运行命令与前文相同：

```bash
scrapy crawl freepik
scrapy crawl istock
scrapy crawl gettyimages
scrapy crawl adobestock
scrapy crawl shutterstock
```