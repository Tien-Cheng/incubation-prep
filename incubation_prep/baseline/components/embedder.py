from enum import Enum
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.embedder.embedder_pytorch import batch
from torchvision import transforms


class Embedder(str, Enum):
    triton = "triton"
    mobilenet = "mobilenet"
    torchreid = "torchreid"


class DeepSORTEmbedder:
    def __init__(
        self,
        embedder: str = Embedder.mobilenet,
        bgr: bool = False,
        embedder_model_name: Optional[str] = None,
        embedder_wts: Optional[str] = None,
        embedder_model_version: Optional[str] = None,
        triton_url: Optional[str] = None,
    ):
        # use a separate instance of deepsort
        # so we can share 1 embedder for all trackers
        self.embedder = embedder
        self.bgr = bgr # Assume image already RGB
        self.embedder_model_name = embedder_model_name
        self.embedder_wts = embedder_wts
        self.embedder_model_version = str(embedder_model_version)
        self.deepsort = DeepSort(
            embedder=embedder if embedder != Embedder.triton else None,
            embedder_model_name=embedder_model_name,
            embedder_wts=embedder_wts,
            bgr=bgr,
        )

        if embedder == Embedder.triton:
            self._init_triton(
                embedder_model_name, self.embedder_model_version, triton_url
            )

    def _init_triton(self, model_name: str, model_version: str, url: str):
        from tritonclient.grpc import InferenceServerClient, InferInput

        parsed_url = urlparse(url)
        self.triton = InferenceServerClient(parsed_url.netloc)
        self.triton_metadata = self.triton.get_model_metadata(
            model_name, model_version, as_json=True
        )
        self.triton_shape = [
            int(s) for s in self.triton_metadata["inputs"][0]["shape"][1:]
        ]
        self.triton_input_name = self.triton_metadata["inputs"][0]["name"]
        self.triton_datatype = self.triton_metadata["inputs"][0]["datatype"]

        def create_input_placeholders(batch_size: int = 1):
            return [
                InferInput(
                    self.triton_input_name,
                    [batch_size] + self.triton_shape,
                    self.triton_datatype,
                )
            ]

        self._create_input_placeholders_fn = create_input_placeholders

    def __call__(
        self, frame: np.ndarray, dets: List[Tuple[List[Union[int, float]], float, str]]
    ):
        cropped_dets = self.deepsort.crop_bb(frame, dets)
        if self.embedder != Embedder.triton:
            embeds = self.deepsort.embedder.predict(cropped_dets)
        else:
            preproc_imgs = [self.preprocess_triton(img) for img in cropped_dets]
            embeds = []
            for det_batch in batch(preproc_imgs, bs=32):
                placeholders = self._create_input_placeholders_fn(len(det_batch))
                placeholders[0].set_data_from_numpy(
                    np.asarray(det_batch).reshape(-1, *self.triton_shape)
                )
                response = self.triton.infer(
                    model_name=self.embedder_model_name,
                    model_version=self.embedder_model_version,
                    inputs=placeholders,
                )
                result = response.as_numpy(self.triton_metadata["outputs"][0]["name"])

                # Convert to list of tensors

                embeds.extend(list(result))
        return embeds

    def preprocess_triton(self, np_image):
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        if self.bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image

        input_image = cv2.resize(
            np_image_rgb, (self.triton_shape[1], self.triton_shape[2])
        )
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_image = trans(input_image)
        input_image = input_image.view(1, 3, self.triton_shape[1], self.triton_shape[2])

        return input_image.numpy()
