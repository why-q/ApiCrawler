import abc
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import click
import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ExtractorParamType(click.ParamType):
    name = "extractor"

    def convert(self, value, param, ctx):
        try:
            module_name, class_name = value.rsplit(".", 1)
            module = importlib.import_module(module_name)
            extractor_class = getattr(module, class_name)
            if not issubclass(extractor_class, FeatureExtractor):
                self.fail(f"{value} is not a subclass of FeatureExtractor", param, ctx)
            return extractor_class
        except (ImportError, AttributeError, ValueError):
            self.fail(f"Invalid extractor: {value}", param, ctx)


EXTRACTOR = ExtractorParamType()


def create_extractor(ctx, param, value):
    batch_size = ctx.params.get("batch_size", 32)
    num_workers = ctx.params.get("num_workers", 4)
    return value(batch_size=batch_size, num_workers=num_workers)


class FeatureExtractor(abc.ABC):
    @property
    @abc.abstractmethod
    def feature_dimension(self) -> int:
        pass

    @abc.abstractmethod
    def extract_features(self, image_paths: List[Path]) -> Dict[str, np.ndarray]:
        pass


class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.preprocessor(image), img_path.name


class TransfaceExtractor(FeatureExtractor):
    """
    Prefer: https://github.com/DanJun6737/TransFace

    WARNING: The environment of this repository is difficult to configure. Not recommended.
    """

    @property
    def feature_dimension(self) -> int:
        return 512

    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()

        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_recognition_func = pipeline(
            Tasks.face_recognition,
            "damo/cv_vit_face-recognition",
            device=self.device,
        )

    def preprocess(self, image):
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

    def extract_features(self, image_paths: List[Path]) -> Dict[str, np.ndarray]:
        from modelscope.outputs import OutputKeys

        dataset = ImageDataset(image_paths, self.preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        features = {}
        for batch, file_names in dataloader:
            try:
                batch = batch.to(self.device)
                results = self.face_recognition_func(batch)

                for i, file_name in enumerate(file_names):
                    try:
                        feature = results[i][OutputKeys.IMG_EMBEDDING][0]
                        features[file_name] = feature.cpu().numpy()
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"No face detected in {file_name}: {e}")
                        features[file_name] = np.full(self.feature_dimension, 1e-8)
            except Exception as e:
                print(f"Error processing batch: {e}")
                for file_name in file_names:
                    features[file_name] = np.full(self.feature_dimension, 1e-8)

        return features


class ISCExtractor(FeatureExtractor):
    """
    Prefer: https://github.com/lyakaap/ISC21-Descriptor-Track-1st
    """

    @property
    def feature_dimension(self) -> int:
        return 256

    def __init__(
        self,
        weight_name="isc_ft_v107",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16,
        num_workers=4,
    ):
        from isc_feature_extractor import create_model

        self.model, self.preprocessor = create_model(
            weight_name=weight_name, model_dir="./datas/models", device=device
        )
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, image_paths: List[Path]) -> Dict[str, np.ndarray]:
        dataset = ImageDataset(image_paths, self.preprocessor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        features = {}
        with torch.no_grad():
            for batch, file_names in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)

                for i, file_name in enumerate(file_names):
                    features[file_name] = output[i].cpu().numpy()

        return features


class VectorDatabase:
    def __init__(self, dimension: int, index_file: Path):
        self.index = faiss.IndexFlatIP(dimension)  # use cosine similarity
        self.id_to_index = {}
        self.index_to_id = {}
        self.index_file = index_file
        if self.index_file.exists():
            self.load_index()

    def add_vectors(self, id_vector_dict: Dict[str, np.ndarray]):
        for id, vector in id_vector_dict.items():
            if id not in self.id_to_index:
                index = self.index.ntotal
                normalized_vector = vector / np.linalg.norm(vector)
                self.index.add(normalized_vector.reshape(1, -1))
                self.id_to_index[id] = index
                self.index_to_id[index] = id

    def search_similar(self, id: str, threshold: float) -> List[Tuple[str, float]]:
        if id not in self.id_to_index:
            return []
        vector = self.index.reconstruct(self.id_to_index[id])
        D, I = self.index.search(vector.reshape(1, -1), self.index.ntotal)
        similar = [
            (self.index_to_id[i], float(d))
            for d, i in zip(D[0], I[0])
            if float(d) >= threshold and self.index_to_id[i] != id
        ]
        return similar

    def delete_vector(self, id: str):
        if id in self.id_to_index:
            index = self.id_to_index[id]
            self.index.remove_ids(np.array([index]))
            del self.id_to_index[id]
            del self.index_to_id[index]

    def check_id_exists(self, id: str) -> bool:
        return id in self.id_to_index

    def save_index(self):
        faiss.write_index(self.index, str(self.index_file))
        metadata_file = self.index_file.with_suffix(".json")
        with metadata_file.open("w") as f:
            json.dump(
                {
                    "id_to_index": self.id_to_index,
                    "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
                },
                f,
            )

    def load_index(self):
        self.index = faiss.read_index(str(self.index_file))
        metadata_file = self.index_file.with_suffix(".json")
        with metadata_file.open("r") as f:
            metadata = json.load(f)
            self.id_to_index = metadata["id_to_index"]
            self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}

    def get_similarity(self, id1: str, id2: str) -> float:
        if id1 not in self.id_to_index or id2 not in self.id_to_index:
            return None
        vector1 = self.index.reconstruct(self.id_to_index[id1])
        vector2 = self.index.reconstruct(self.id_to_index[id2])
        similarity = np.dot(vector1, vector2)
        return float(similarity)


class ImageDeduplicator:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        vector_db: VectorDatabase,
        similarity_threshold: float,
    ):
        self.feature_extractor = feature_extractor
        self.vector_db = vector_db
        self.similarity_threshold = similarity_threshold
        self.custom_filters = []

    def add_custom_filter(self, filter_func):
        self.custom_filters.append(filter_func)

    def crawl_images(self, folder_list: List[Path]) -> List[Path]:
        image_paths = []
        for folder in folder_list:
            for file_path in folder.rglob("*"):
                if file_path.suffix.lower() in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                ):
                    image_paths.append(file_path)
        return image_paths

    def build_or_update_database(self, image_paths: List[Path]):
        new_images = [
            path
            for path in image_paths
            if not self.vector_db.check_id_exists(path.name)
        ]
        if new_images:
            features = self.feature_extractor.extract_features(new_images)
            for id, vector in features.items():
                features[id] = vector / np.linalg.norm(vector)  # normalize
            self.vector_db.add_vectors(features)
        self.vector_db.save_index()

    def deduplicate(
        self, folder_list: List[Path], output_path: Path
    ) -> Tuple[List[Path], Dict[str, List[Dict[str, float]]]]:
        image_paths = self.crawl_images(folder_list)
        self.build_or_update_database(image_paths)

        duplicates = {}
        duplicate_set = set()
        unique_images = []
        for path in image_paths:
            img_id = path.name
            if img_id in duplicate_set:
                continue

            similar = self.vector_db.search_similar(img_id, self.similarity_threshold)
            if similar:
                for custom_filter in self.custom_filters:
                    similar = [s for s in similar if custom_filter(img_id, s[0])]
                    if not similar:
                        break

                if similar:
                    duplicates[img_id] = [{s[0]: s[1]} for s in similar]
                    duplicate_set.update(s[0] for s in similar)

            unique_images.append(path)

        # Save duplicates dictionary
        with output_path.open("w") as f:
            json.dump(duplicates, f, indent=2)

        return unique_images, duplicates

    def get_image_similarity(self, image_path1: Path, image_path2: Path) -> float:
        id1 = image_path1.name
        id2 = image_path2.name

        if not self.vector_db.check_id_exists(
            id1
        ) or not self.vector_db.check_id_exists(id2):
            raise ValueError(f"Image ID {id1} or {id2} not found in the database")

        similarity = self.vector_db.get_similarity(id1, id2)
        if similarity is None:
            raise ValueError(f"Unable to compute similarity between {id1} and {id2}")
        return similarity


@click.group()
def cli():
    """Image deduplication and similarity calculation tool."""
    pass


@cli.command()
@click.option(
    "--folder",
    "-f",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Folders containing images to deduplicate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file for duplicate information",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    help="Similarity threshold for deduplication",
)
@click.option(
    "--index-file",
    "-i",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("image_index.bin"),
    help="Index file for vector database",
)
@click.option(
    "--batch-size", "-b", type=int, default=16, help="Batch size for feature extraction"
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=4,
    help="Number of worker processes for data loading",
)
@click.option(
    "--custom-filters",
    "-cf",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Python files containing custom filter functions (can be specified multiple times)",
)
@click.option(
    "--extractor",
    "-e",
    type=EXTRACTOR,
    default="__main__.ISCExtractor",
    callback=create_extractor,
    help="Fully qualified name of the FeatureExtractor subclass to use",
)
def deduplicate(
    folder,
    output,
    threshold,
    index_file,
    batch_size,
    num_workers,
    custom_filters,
    extractor,
):
    """
    Deduplicate images in specified folders.

    This command processes images in the given folders, identifies duplicates
    based on the specified cosine similarity threshold, and saves the duplicate
    information to an output file.

    The cosine similarity ranges from -1 to 1, where 1 indicates identical images,
    0 indicates no similarity, and -1 indicates completely opposite images.
    Images with similarity >= threshold are considered duplicates.

    Custom filters can be specified multiple times. Each should be a Python file containing
    a function named 'custom_filter' that takes two image IDs as input and returns a boolean.
    All specified filters will be applied in the order they are provided.

    Also, you can specify a custom feature extractor by providing its fully qualified name.
    The extractor must be a subclass of FeatureExtractor. Custom filters should be defined in separate Python files. Each file should contain a function named 'custom_filter' with the following signature:

    def custom_filter(img_id1: str, img_id2: str) -> bool:
        # Custom logic here
        # Return True if the pair should be considered for deduplication
        # Return False if the pair should be excluded from deduplication

    Multiple custom filters can be applied, and they will be executed in the
    order they are provided.

    Example usage:
    python script.py deduplicate -f /path/to/folder1 -f /path/to/folder2 -o duplicates.json -t 0.98 -i image_index.bin -b 64 -w 8 -cf filter1.py -cf filter2.py -e your_module.YourExtractor
    """
    vector_db = VectorDatabase(
        dimension=extractor.feature_dimension, index_file=index_file
    )
    deduplicator = ImageDeduplicator(
        extractor, vector_db, similarity_threshold=threshold
    )

    for filter_file in custom_filters:
        spec = importlib.util.spec_from_file_location(
            f"custom_filter_module_{filter_file.stem}", filter_file
        )
        custom_filter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_filter_module)
        deduplicator.add_custom_filter(custom_filter_module.custom_filter)

    unique_images, duplicates = deduplicator.deduplicate(folder, output)

    click.echo(f"Found {len(duplicates)} duplicate groups")
    click.echo(f"Remaining unique images: {len(unique_images)}")
    click.echo(f"Duplicate information saved to {output}")


@cli.command()
@click.argument("image1", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("image2", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--index-file",
    "-i",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("image_index.bin"),
    help="Index file for vector database",
)
@click.option(
    "--batch-size", "-b", type=int, default=16, help="Batch size for feature extraction"
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=4,
    help="Number of worker processes for data loading",
)
@click.option(
    "--extractor",
    "-e",
    type=EXTRACTOR,
    default="__main__.ISCExtractor",
    callback=create_extractor,
    help="Fully qualified name of the FeatureExtractor subclass to use",
)
def similarity(image1, image2, index_file, batch_size, num_workers, extractor):
    """
    Calculate similarity between two images.

    This command computes the cosine similarity between two specified images.
    The similarity score ranges from -1 to 1, where 1 indicates identical images,
    0 indicates no similarity, and -1 indicates completely opposite images.

    The command will use an existing index file if available, or create a new one
    if not found. You can specify the batch size and number of workers for the
    feature extraction process.

    You can specify a custom feature extractor by providing its fully qualified name.
    The extractor must be a subclass of FeatureExtractor.

    Example usage:
    python script.py similarity /path/to/image1.jpg /path/to/image2.jpg -i image_index.bin
    """
    vector_db = VectorDatabase(
        dimension=extractor.feature_dimension, index_file=index_file
    )
    deduplicator = ImageDeduplicator(extractor, vector_db, similarity_threshold=0.95)

    try:
        similarity = deduplicator.get_image_similarity(image1, image2)
        click.echo(
            f"Cosine similarity between {image1.name} and {image2.name}: {similarity:.4f}"
        )
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option(
    "--json_path",
    "-j",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--image_dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
def unique(json_path: Path, image_dir: Path, output_dir: Path):
    """
    Copy unique images to the output directory.

    This command copies unique images to the output directory. The JSON file should contain a dictionary with image names as keys, generated by the deduplicate command.

    Example usage:
    python script.py unique -j duplicates.json -i /path/to/images -o /path/to/output
    """
    import shutil

    try:
        with json_path.open("r") as f:
            output_json = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        return
    except IOError:
        print(f"Error: Unable to read file: {json_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    duals = set()
    for _, v in output_json.items():
        for dual in v:
            duals.update(dual.keys())

    valid_extensions = {".png", ".jpg"}

    for image in image_dir.glob("*"):
        if (
            image.is_file()
            and image.suffix.lower() in valid_extensions
            and image.name not in duals
        ):
            try:
                shutil.copy(str(image), str(output_dir / image.name))
                print(f"{image.name} -> {output_dir}")
            except IOError as e:
                print(f"Error copying {image.name}: {e}")


if __name__ == "__main__":
    cli()
