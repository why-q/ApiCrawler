import json
import scrapy
import api_crawler.config as config


class HuabanSpider(scrapy.Spider):
    name = "huaban"
    allowed_domains = ["huaban.com"]
    custom_settings = {
        "IMAGES_STORE": config.HUABAN_IMAGE_DIR,
        "LOG_FILE": config.HUABAN_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.HuabanImagePipeline": 1},
    }
    base_url = "https://huaban.com/v3/search/file?text={query}&sort=all&limit={per_page}&page={page}"
    image_host_url = "https://gd-hbimg.huaban.com"

    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
    }

    def __init__(
        self,
        query: str = config.HUABAN_QUERY,
        pages: str = config.HUABAN_PAGES,  # "10" or "3-15"
        per_page: int = config.HUABAN_PER_PAGE,  # limit: 40-80,
        *args,
        **kwargs,
    ):
        super(HuabanSpider, self).__init__(*args, **kwargs)

        self.query = query.replace(" ", "-")

        self.per_page = int(per_page)
        assert (
            40 <= self.per_page <= 100
        ), f"Huaban API limits the number of photos per page from 40 to 100, got {self.per_page}"
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
            url = self.base_url.format(
                query=self.query, page=page, per_page=self.per_page
            )
            yield scrapy.Request(url, headers=self.header, callback=self.parse)

    def parse(self, response):
        data = json.loads(response.text)
        for pin in data["pins"]:
            yield {
                "image_url": f"{self.image_host_url}/{pin['file']['key']}_fw1200webp",
                "image_id": pin["file_id"],
            }
