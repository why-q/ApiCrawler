import re

import scrapy
from bs4 import BeautifulSoup

import api_crawler.config as config


class AdobeStockSpider(scrapy.Spider):
    name = "adobestock"
    allowed_domains = ["adobe.com"]
    custom_settings = {
        "IMAGES_STORE": config.ISTOCK_IMAGE_DIR,
        "LOG_FILE": config.ISTOCK_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.AdobeStockPipeline": 1},
    }

    base_url = "https://stock.adobe.com/search/images?filters%5Bcontent_type%3Aphoto%5D=1&filters%5Bcontent_type%3Aillustration%5D=1&filters%5Bcontent_type%3Azip_vector%5D=1&filters%5Bcontent_type%3Avideo%5D=0&filters%5Bcontent_type%3Atemplate%5D=0&filters%5Bcontent_type%3A3d%5D=0&filters%5Bcontent_type%3Aaudio%5D=0&filters%5Binclude_stock_enterprise%5D=0&filters%5Bcontent_type%3Aimage%5D=1&k={query}&order=relevance&safe_search=1&price%5B%24%5D=1&limit=100&search_type=pagination&search_page={page}&load_type=page&get_facets=0"

    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
    }

    def __init__(
        self,
        query: str = config.ISTOCK_QUERY,
        pages: str = config.ISTOCK_PAGES,  # "10" or "3-15"
        *args,
        **kwargs,
    ):
        super(AdobeStockSpider, self).__init__(*args, **kwargs)

        # TODO Check the query format
        # self.query = query.replace(" ", "-")
        self.query = query

        try:
            if "-" in pages:
                start_page, end_page = map(int, pages.split("-"))
                assert 0 < start_page < end_page
                self.pages = range(start_page, end_page + 1)
            else:
                self.pages = range(1, int(pages) + 1)
        except ValueError as e:
            raise f"Error occured while parsing `pages` parameter: {e}"

    def start_requests(self):
        for page in self.pages:
            url = self.base_url.format(query=self.query, page=page)
            yield scrapy.Request(url, headers=self.header, callback=self.parse)

    def parse(self, response):
        # with open("response.html", "w", encoding="utf-8") as f:
        #     f.write(response.text)

        picture_elements = response.xpath("//picture")

        for picture in picture_elements:
            webp_src = picture.xpath('.//source[@type="image/webp"]/@srcset').get()
            jpg_src = picture.xpath(".//img/@src").get()

            if not webp_src or not jpg_src:
                continue

            if (
                "https:" not in webp_src
                or ".webp" not in webp_src
                or "https:" not in jpg_src
                or ".jpg" not in jpg_src
            ):
                continue

            """
                Get id from src:

                src is like: https://t3.ftcdn.net/jpg/03/19/10/82/360_F_319108213_MoLZZt5WTXq6NpdG1FbCbyCtDStlSVBf.jpg

                id is like: 319108213
            """

            image_url = jpg_src
            match = re.search(r"/(\d+)_F_(\d+)_", image_url)

            if match:
                image_id = match.group(2)

                print("Found image: ", image_url)

                yield {"image_id": image_id, "image_url": image_url}
