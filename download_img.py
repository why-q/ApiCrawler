import asyncio
import sys
import time
from io import BytesIO
from pathlib import Path

import aiohttp
import click
import pillow_heif
from PIL import Image

PLATFORM_CONFIG = {
    "url_id_platforms": [
        "pexels",
        "unsplash",
        "huaban",
        "freepik",
        "istock",
        "gettyimages",
        "adobestock",
    ],
    "url_only_platforms": ["xhs", "weibo"],
    "heif_platforms": ["xhs"],
}


async def download_image(session, url, output_path, max_retries=3, convert_heif=False):
    if output_path.exists():
        click.echo(f"Skipping existing img: {url}")
        return

    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    try:
                        image = Image.open(BytesIO(image_data))
                        image = image.convert("RGBA")
                        image.save(output_path, "PNG")
                        click.echo(f"Downloaded and converted: {url}")
                        return
                    except Image.UnidentifiedImageError:
                        if convert_heif:
                            heif_file = pillow_heif.read_heif(BytesIO(image_data))
                            image = Image.frombytes(
                                heif_file.mode,
                                heif_file.size,
                                heif_file.data,
                                "raw",
                                heif_file.mode,
                                heif_file.stride,
                            )
                            image = image.convert("RGBA")
                            image.save(output_path, "PNG")
                            click.echo(f"Downloaded and converted HEIF: {url}")
                            return
                        else:
                            click.echo(f"Failed to convert to PNG: {url}")
                else:
                    click.echo(f"Wrong response status of {url}: {response.status}")
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                click.echo(
                    f"Error downloading {url}: {e}. Retrying... (Attempt {attempt + 1})"
                )
                await asyncio.sleep(5)
            else:
                click.echo(
                    f"Failed to download {url} after {max_retries} attempts: {e}"
                )
                return


async def download_images(
    txt_file, output_dir, max_concurrent=10, max_retries=3, platform="xhs"
):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_download(url, id_=None):
            async with semaphore:
                if platform in PLATFORM_CONFIG["url_id_platforms"]:
                    output_path = Path(output_dir) / f"{id_}.png"
                else:
                    output_path = Path(output_dir) / f"{url.split('/')[-1]}.png"
                await download_image(
                    session,
                    url,
                    output_path,
                    max_retries,
                    convert_heif=(platform in PLATFORM_CONFIG["heif_platforms"]),
                )

        with open(txt_file, "r") as file:
            if platform in PLATFORM_CONFIG["url_id_platforms"]:
                lines = [line.strip().split() for line in file]
                tasks = [
                    asyncio.create_task(bounded_download(url, id_))
                    for url, id_ in lines
                ]
            else:
                image_urls = file.readlines()
                tasks = [
                    asyncio.create_task(bounded_download(url.strip()))
                    for url in image_urls
                ]

        await asyncio.gather(*tasks)


@click.command()
@click.option(
    "--platform",
    type=click.Choice(
        PLATFORM_CONFIG["url_id_platforms"] + PLATFORM_CONFIG["url_only_platforms"]
    ),
    default="xhs",
    help="Platform to download from",
)
@click.option(
    "--txt_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Text file paths containing image URLs",
)
@click.option(
    "--txt_dir",
    type=click.Path(exists=True),
    help="Directory containing text files with image URLs",
)
@click.option(
    "--output_dir",
    type=click.Path(),
    default="./data/img/",
    help="Output directory for downloaded images",
)
@click.option(
    "--max_retries", type=int, default=3, help="Maximum number of download retries"
)
@click.option(
    "--max_concurrent",
    type=int,
    default=10,
    help="Maximum number of concurrent downloads",
)
def main(platform, txt_paths, txt_dir, output_dir, max_retries, max_concurrent):
    """
    Main function: Process command-line arguments and execute image download tasks.

    Example:

    You can specify txt_paths to download images from specific text files:

    python download_img.py --platform xhs --txt_paths file1.txt file2.txt --output_dir ./data/img/ --max_retries 3 --max_concurrent 10

    or use txt_dir to specify a directory containing multiple text files:

    python download_img.py --platform xhs --txt_dir ./data/txt/ --output_dir ./data/img/ --max_retries 3 --max_concurrent 10
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if txt_dir:
        txt_paths = list(Path(txt_dir).glob("*.txt"))
    elif not txt_paths:
        click.echo("Error: Either --txt_paths or --txt_dir must be provided.")
        return

    async def process_files():
        for txt_path in txt_paths:
            for _ in range(max_retries):
                try:
                    await download_images(
                        txt_path, output_dir, max_concurrent, max_retries, platform
                    )
                    click.echo(f"Finished downloading {txt_path}")
                    break
                except aiohttp.client_exceptions.ClientConnectionError:
                    click.echo("Connection error, retrying in 30 seconds...")
                    time.sleep(30)

    try:
        asyncio.get_event_loop().run_until_complete(process_files())
    except KeyboardInterrupt:
        click.echo("Download interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
