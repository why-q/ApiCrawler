import json

import scrapy

import api_crawler.config as config
from typing import Literal


class UnsplashSpider(scrapy.Spider):
    name = "unsplash"
    allowed_domains = ["unsplash.com"]
    custom_settings = {
        "IMAGES_STORE": config.UNSPLASH_IMAGE_DIR,
        "LOG_FILE": config.UNSPLASH_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.UnsplashImagePipeline": 1},
    }

    base_url = "https://api.unsplash.com/search/photos?query={query}&page={page}&per_page={per_page}&client_id={key}"

    def __init__(
        self,
        key: str = config.UNSPLASH_KEY,
        query: str = config.UNSPLASH_QUERY,
        pages: str = config.UNSPLASH_PAGES,
        per_page: str = config.UNSPLASH_PER_PAGE,
        ima_type: Literal[
            "raw", "full", "regular", "small", "thumb"
        ] = config.UNSPLASH_IMAGE_TYPE,
        *args,
        **kwargs,
    ):
        super(UnsplashSpider, self).__init__(*args, **kwargs)

        self.key = key
        self.per_page = int(per_page)
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

        self.img_type = ima_type
        if ima_type not in ["raw", "full", "regular", "small", "thumb"]:
            raise ValueError(f"Invalid image type: {ima_type}")

    def start_requests(self):
        for page in self.pages:
            url = self.base_url.format(
                query=self.query, page=page, per_page=self.per_page, key=self.key
            )
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        data = json.loads(response.text)
        for photo in data["results"]:
            yield {
                "image_url": photo["urls"][self.img_type],
                "image_id": photo["id"],
            }
