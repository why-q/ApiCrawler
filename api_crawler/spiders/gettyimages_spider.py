import re

import scrapy
from bs4 import BeautifulSoup

import api_crawler.config as config


class GettyImagesSpider(scrapy.Spider):
    name = "gettyimages"
    allowed_domains = ["gettyimages.com"]
    custom_settings = {
        "IMAGES_STORE": config.ISTOCK_IMAGE_DIR,
        "LOG_FILE": config.ISTOCK_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.GettyImagesPipeline": 1},
    }
    base_url = "https://www.gettyimages.com/search/2/image?page={page}&phrase={query}&sort=mostpopular"

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
        super(GettyImagesSpider, self).__init__(*args, **kwargs)

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
        soup = BeautifulSoup(response.text, "html.parser")

        img_elements = soup.find_all("img", class_="BLA_wBUJrga_SkfJ8won")
        for img in img_elements:
            image_url = img.get("src")
            if image_url:
                """
                    Get image id from src:

                    src is like: https://media.gettyimages.com/id/600096494/zh/%E7%85%A7%E7%89%87/half-face-of-young-man.jpg?s=612x612&w=0&k=20&c=q5W_9Tl6QvG_NzBmcSXYbq3pnqzwAZOokYkc6Y4Kg0I=

                    id is: 600096494
                """
                match = re.search(r"/id/(\d+)/", image_url)
                if match:
                    image_id = match.group(1)

                    print("Found image: ", image_url)

                    yield {
                        "image_url": image_url,
                        "image_id": image_id,
                    }
                else:
                    print(f"Error processing an image: {image_url}")
