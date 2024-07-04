from pathlib import Path
import api_crawler.config as config


class PexelsImagePipeline:
    def __init__(
        self,
        txt_dir: str = config.PEXELS_IMAGE_URL_TXT_DIR,
        img_dir: str = config.PEXELS_IMAGE_DIR,
    ):
        txt_dir = Path(txt_dir)
        if not txt_dir.exists():
            txt_dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = (
            txt_dir
            / f"{config.PEXELS_QUERY.replace(' ', '-')}_{config.PEXELS_IMAGE_TYPE}_{config.PEXELS_PAGES}_{config.PEXELS_PER_PAGE}.txt"
        )

        self.img_dir = Path(img_dir)
        if not self.img_dir.exists():
            self.img_dir.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.items = []
        self.ids = set()

    def open_spider(self, spider):
        self.file = open(self.txt_path, "w", encoding="utf-8")

    def close_spider(self, spider):
        for item in self.items:
            self.file.write(item)
        self.file.close()

    def process_item(self, item: dict, spider):
        if item.get("image_url") and item.get("image_id"):
            if item["image_id"] not in self.ids:
                self.items.append(f"{item['image_url']} {item['image_id']}\n")
                self.ids.add(item["image_id"])

        return item


class UnsplashImagePipeline:
    def __init__(
        self,
        txt_dir: str = config.UNSPLASH_IMAGE_URL_TXT_DIR,
        img_dir: str = config.UNSPLASH_IMAGE_DIR,
    ):
        txt_dir = Path(txt_dir)
        if not txt_dir.exists():
            txt_dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = (
            txt_dir
            / f"{config.PEXELS_QUERY.replace(' ', '-')}_{config.UNSPLASH_IMAGE_TYPE}_{config.PEXELS_PAGES}_{config.PEXELS_PER_PAGE}.txt"
        )

        self.img_dir = Path(img_dir)
        if not self.img_dir.exists():
            self.img_dir.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.items = []
        self.ids = set()

    def open_spider(self, spider):
        self.file = open(self.txt_path, "w", encoding="utf-8")

    def close_spider(self, spider):
        for item in self.items:
            self.file.write(item)
        self.file.close()

    def process_item(self, item: dict, spider):
        if item.get("image_url") and item.get("image_id"):
            if item["image_id"] not in self.ids:
                self.items.append(f"{item['image_url']} {item['image_id']}\n")
                self.ids.add(item["image_id"])

        return item


class HuabanImagePipeline:
    def __init__(
        self,
        txt_dir: str = config.HUABAN_IMAGE_URL_TXT_DIR,
        img_dir: str = config.HUABAN_IMAGE_DIR,
    ):
        txt_dir = Path(txt_dir)
        if not txt_dir.exists():
            txt_dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = (
            txt_dir
            / f"{config.HUABAN_QUERY.replace(' ', '-')}_{config.HUABAN_PAGES}_{config.HUABAN_PER_PAGE}.txt"
        )

        self.img_dir = Path(img_dir)
        if not self.img_dir.exists():
            self.img_dir.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.items = []
        self.ids = set()

    def open_spider(self, spider):
        self.file = open(self.txt_path, "w", encoding="utf-8")

    def close_spider(self, spider):
        for item in self.items:
            self.file.write(item)
        self.file.close()

    def process_item(self, item: dict, spider):
        if item.get("image_url") and item.get("image_id"):
            if item["image_id"] not in self.ids:
                self.items.append(f"{item['image_url']} {item['image_id']}\n")
                self.ids.add(item["image_id"])

        return item
