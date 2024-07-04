import json
import scrapy
import api_crawler.config as config
from typing import Literal


class PexelsSpider(scrapy.Spider):
    name = "pexels"
    allowed_domains = ["pexels.com"]
    custom_settings = {
        "IMAGES_STORE": config.PEXELS_IMAGE_DIR,
        "LOG_FILE": config.PEXELS_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.PexelsImagePipeline": 1},
    }
    base_url = (
        "https://api.pexels.com/v1/search?query={query}&page={page}&per_page={per_page}"
    )

    def __init__(
        self,
        key: str = config.PEXELS_KEY,
        query: str = config.PEXELS_QUERY,
        pages: str = config.PEXELS_PAGES,  # "10" or "3-15"
        per_page: int = config.PEXELS_PER_PAGE,  # limit: 15-80,
        img_type: Literal[
            "original",
            "large",
            "large2x",
            "medium",
            "small",
            "portrait",
            "landscape",
            "tiny",
        ] = config.PEXELS_IMAGE_TYPE,
        *args,
        **kwargs,
    ):
        super(PexelsSpider, self).__init__(*args, **kwargs)

        self.key = key
        self.query = query.replace(" ", "-")
        self.headers = {"Authorization": f"{key}"}

        self.per_page = int(per_page)
        assert (
            15 <= self.per_page <= 80
        ), f"Pexels API limits the number of photos per page from 15 to 80, got {self.per_page}"
        try:
            if "-" in pages:
                start_page, end_page = map(int, pages.split("-"))
                assert 0 < start_page < end_page
                self.pages = range(start_page, end_page + 1)
            else:
                self.pages = range(1, int(pages) + 1)
        except ValueError as e:
            raise f"Error occured while parsing `pages` parameter: {e}"

        self.img_type = img_type
        if img_type not in [
            "original",
            "large",
            "large2x",
            "medium",
            "small",
            "portrait",
            "landscape",
            "tiny",
        ]:
            raise ValueError(f"Invalid image type: {img_type}")

    def start_requests(self):
        for page in self.pages:
            url = self.base_url.format(
                query=self.query, page=page, per_page=self.per_page
            )
            yield scrapy.Request(url, headers=self.headers, callback=self.parse)

    def parse(self, response):
        data = json.loads(response.text)
        for photo in data["photos"]:
            yield {
                "image_url": photo["src"][self.img_type],
                "image_id": photo["id"],
            }
