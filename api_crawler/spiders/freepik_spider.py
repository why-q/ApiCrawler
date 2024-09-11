import re
import time

import scrapy
from scrapy import Request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

import api_crawler.config as config


class FreepikSpider(scrapy.Spider):
    name = "freepik"
    allowed_domains = ["freepik.com"]
    custom_settings = {
        "IMAGES_STORE": config.FREEPIK_IMAGE_DIR,
        "LOG_FILE": config.FREEPIK_LOG_PATH,
        "ITEM_PIPELINES": {"api_crawler.pipelines.FreepikImagePipeline": 1},
        "LOG_LEVEL": "DEBUG",
    }

    base_url = "https://www.freepik.com/search?ai=excluded&format=search&last_filter=page&last_value={page}&page={page}&people=include&people_range=1&query={query}&type=photo"

    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.freepik.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(
        self,
        query: str = config.FREEPIK_QUERY,
        pages: int = config.FREEPIK_PAGES,
        *args,
        **kwargs,
    ):
        super(FreepikSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.pages = range(1, int(pages) + 1)

        # Attention: Recently, we cannnot use headless mode the crawl the website.
        chrome_options = Options()

        service = Service(executable_path="./driver/chromedriver.exe")
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def start_requests(self):
        for page in self.pages:
            url = self.base_url.format(query=self.query, page=page)
            print("Crawling url: ", url)

            yield Request(
                url,
                self.parse,
            )

    def parse(self, response):
        try:
            self.driver.get(response.url)

            time.sleep(
                10
            )  # Wait for page to load completely, since the website use lazy load.

            container = self.driver.find_element(
                By.CSS_SELECTOR,
                "#__next > div._dsw1x81._dsw1x80._dsw1x82 > div._dsw1x87._1286nb1h._1286nb1k._1286nb1n > div > div._nkl6i52._1286nb12yv._1286nb133v._1286nb197",
            )

        except Exception as e:
            self.logger.error(f"Error to visit URL by Selenium: {e}")
            return

        # Save HTML for debugging
        # with open("freepik.html", "w", encoding="utf-8") as f:
        #     f.write(self.driver.page_source)

        divs = container.find_elements(By.CLASS_NAME, "_1286nb1m")

        for div in divs:
            figures = div.find_elements(By.TAG_NAME, "figure")
            for figure in figures:
                try:
                    img = figure.find_element(By.TAG_NAME, "img")
                    image_url = img.get_attribute("src")

                    if image_url.startswith("https://img.freepik.com"):
                        """
                        Each URL is like: `https://img.freepik.com/free-photo/close-up-portrait-green-eyed-dark-haired-woman-with-healthy-skin-cream-her-face-girl-without-makeup-white-wall_197531-13905.jpg?ga=GA1.1.1865334328.1725937059&semt=ais_hybrid`

                        https://img.freepik.com/free-vector/makeup-accessories-background_23-2147806488.jpg?t=st=1725968846~exp=1725972446~hmac=04a966751fc0b274500ecbe6d480398497909c89d4b5664bfdd83342f561faac&w=1380

                        The id is: `197531-13905`

                        We use `re` to get it.
                        """
                        id_match = re.search(r"_(\d+-\d+)\.jpg", image_url)

                        if id_match:
                            image_id = id_match.group(1)

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
