import re

import scrapy
from scrapy import Request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import api_crawler.config as config


class ShutterStockSpider(scrapy.Spider):
    name = "shutterstock"
    allowed_domains = ["www.shutterstock.com"]
    custom_settings = {
        "IMAGES_STORE": config.SHUTTERSTOCK_IMAGE_DIR,
        "LOG_FILE": config.SHUTTERSTOCK_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.ShutterStockPipeline": 1},
        "LOG_LEVEL": "DEBUG",
        "HTTPERROR_ALLOWED_CODES": [403],
    }

    # TODO filter image type
    base_url = (
        "https://www.shutterstock.com/search/{query}?image_type=photo&page={page}"
    )

    def __init__(
        self,
        query: str = config.SHUTTERSTOCK_QUERY,
        pages: int = config.SHUTTERSTOCK_PAGES,
        *args,
        **kwargs,
    ):
        super(ShutterStockSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.pages = range(1, int(pages) + 1)

        # options
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")

        service = Service(executable_path="./driver/chromedriver.exe")
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def start_requests(self):
        for page in self.pages:
            url = self.base_url.format(query=self.query, page=page)
            print("Crawling url: ", url)

            yield Request(
                url,
                callback=self.parse,
                dont_filter=True,
            )

    def parse(self, response):
        try:
            self.driver.get(response.url)

            wait = WebDriverWait(self.driver, 20)
            wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "mui-1l7n00y-thumbnail"))
            )

            img_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "img.mui-1l7n00y-thumbnail"
            )

            for img in img_elements:
                image_url = img.get_attribute("src")

                """
                    Getting image id from image url

                    The src is like: https://www.shutterstock.com/image-photo/closeup-shot-woman-glowing-skin-260nw-2494192567.jpg

                    The id is: 2494192567
                """

                match = re.search(r"-(\d+)\.jpg$", image_url)
                if match:
                    image_id = match.group(1)

                    print("Found image: ", image_url)

                    yield {
                        "image_url": image_url,
                        "image_id": image_id,
                    }
                else:
                    print(f"Error processing an image: {image_url}")

        except Exception:
            pass

    def closed(self, reason):
        if hasattr(self, "driver"):
            self.driver.quit()
