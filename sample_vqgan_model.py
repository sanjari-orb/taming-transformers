
from typing import Optional, List, Dict
from PIL import Image
import io
import albumentations
from typing import Optional, Tuple, Iterator, List
import logging
import os
import glob
import urllib.parse
import argparse
import torch
from google.cloud import storage

GCS_UPLOAD_TIMEOUT_SECS = 300

def get_default_storage_client():
    return storage.Client()

def get_blob(
    bucket_name: str,
    blob_name: str,
    download_blob: bool = False,
    client: Optional[storage.Client] = None,
) -> storage.blob.Blob:
    if not client:
        client = get_default_storage_client()
    bucket = client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    if download_blob:
        blob = bucket.get_blob(blob_name)
    else:
        blob = bucket.blob(blob_name)
    return blob


def get_gcs_path(bucket_name: str, object_relative_path: str) -> str:
    return f"gs://{bucket_name}/{object_relative_path}"

def upload_blob(
    bucket_name: str,
    destination_blob_name: str,
    source_file_name: str,
    client: Optional[storage.Client] = None,
) -> str:
    """
    Uploads a file to the bucket.

    The ID of your GCS bucket
    bucket_name = "your-bucket-name"

    The path to your file to upload
    source_file_name = "local/path/to/file"

    The ID of your GCS object
    destination_blob_name = "storage-object-name"

    Returns the URI of the uploaded file.
    """

    blob = get_blob(bucket_name, destination_blob_name, client=client)
    blob.upload_from_filename(source_file_name, timeout=GCS_UPLOAD_TIMEOUT_SECS)
    logging.debug(f"File {source_file_name} uploaded to {destination_blob_name}.")
    return get_gcs_path(bucket_name, destination_blob_name)

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="",
        help="path for input directory containing images",
    )

    parser.add_argument(
        "-o",
        "--output_gcs_path",
        type=str,
        default="",
        help="gcs path to upload the images directory to",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="full path of model checkpoint",
    )

    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="full path of yaml config the model was trained with",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="batch size to use for forward pass"
    )
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import glob
import numpy as np
from taming.data.utils import custom_collate
import shutil

def batch(image, processor):
    if not image.mode == "RGB":
        image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        processed = processor(image=image)
        return {
            "image": (processed["image"]/127.5 - 1.0).astype(np.float32)
        
        }
def sample(model, batch):
    x = model.get_input(batch, 'image')
    xrec, _ = model(x)
    if x.shape[1] > 3:
      assert xrec.shape[1] > 3
      x = model.to_rgb(x)
      xrec = model.to_rgb(xrec)
    return x, xrec

def save_image(x, path):
     c,h,w = x.shape
     assert c==3
     x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
     Image.fromarray(x).save(path)

def decode_gcs_uri(uri: str) -> Tuple[str, str]:
    p = urllib.parse.urlparse(uri)
    bucket_name = p.netloc
    object_path = p.path.lstrip("/")
    if p.query:
        object_path += "?" + p.query
    if p.fragment:
        object_path += "#" + p.fragment
    return bucket_name, object_path


if __name__ == 'main':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    ckpt_path = args.model
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    config_path = args.config_path
    config = OmegaConf.load(config_path)

    model = VQModel(**config.model.params) 
    missing, unexpected = model.load_state_dict(state_dict)

    input_dir = args.input_dir
    output_dir = '/tmp/webclip_images/'
    images_fns = glob.glob(input_dir + "/*.png")
    rescaler = albumentations.SmallestMaxSize(max_size=336)
    cropper = albumentations.RandomCrop(
      height=336, 
      width=336)
    preprocessor = albumentations.Compose(
      [rescaler, cropper])
    for fn in images_fns:
        img_num = fn.split('/')[-1]
        pil_image = Image.open(fn)
        inp = batch(pil_image, processor=preprocessor)
        x, xrec = sample(model, custom_collate([inp]))
        save_image(xrec, output_dir + img_num)

    shutil.make_archive('tmp.zip', 'zip', output_dir)
    bucket, object = decode_gcs_uri(args.output_gcs_path)
    upload_blob(bucket, object, 'tmp.zip')
