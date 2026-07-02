#########################################################
## Endpoint fastapi app to select roi bounding box
#########################################################
import asyncio
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import onnxruntime
from starlette.middleware.cors import CORSMiddleware

MAX_UPLOAD_BYTES = 1024 * 1024
MAX_IMAGE_PIXELS = 1_000_000
MAX_IMAGE_SIDE = 860
INFERENCE_TIMEOUT_SECONDS = 2.0
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
DEFAULT_BBOX = [0.25, 0.25, 0.75, 0.75]

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

session = onnxruntime.InferenceSession('out_dyn.onnx')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://lexicam.app', 'https://www.lexicam.app', 'http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image(image_bytes):
    """
    Loading and validating the image. Raises HTTPException if the image is invalid or too large.
    """

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail='Invalid image') from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)

            if image.width <= 0 or image.height <= 0:
                raise HTTPException(status_code=400, detail='Invalid image dimensions')

            if image.width * image.height > MAX_IMAGE_PIXELS or max(image.size) > MAX_IMAGE_SIDE:
                raise HTTPException(status_code=413, detail='Image dimensions too large')

            if image.mode != 'RGB':
                image = image.convert('RGB')
            else:
                image = image.copy() # needed because of with statement

    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail='Invalid image') from exc

    return image


def run_detection(image_bytes):
    image_PIL = load_image(image_bytes)
    model_input = np.asarray(image_PIL, dtype=np.float32).transpose(2, 0, 1) / 255.0
    output = session.run(None, {'input': model_input})
    bbox = output[0].tolist()

    if not bbox:
        return {'bbox': DEFAULT_BBOX}

    bbox[0] = [
        bbox[0][0] / image_PIL.width,
        bbox[0][1] / image_PIL.height,
        bbox[0][2] / image_PIL.width,
        bbox[0][3] / image_PIL.height,
    ]

    return {'bbox': bbox[0]}


@app.post("/detect")
async def detectImageSurge(image: UploadFile = File(...)):

    try:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail='Unsupported image type')

        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='Image upload too large')

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_detection, image_bytes),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail='Detection timed out') from exc










