import os

import tqdm
import torch
import numpy as np
from PIL import Image

import networks
from .video import VideoManagement


class VideoImageIterator:
    TMP_DIR = "dain-tmp"
    STORYBOARD_DIR = os.path.join(TMP_DIR, "storyboard")
    SEGMENT_DIR = os.path.join(TMP_DIR, "segments")

    def __init__(self, source_video: str):
        self.__source_video = source_video
        os.makedirs(self.TMP_DIR, exist_ok=True)
        for root, _, files in os.walk(self.TMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        self._manager = VideoManagement(source_video)
        self._create_video_segments()
        self._create_first_screen_from_segments()

    def _create_video_segments(self):
        basename = os.path.basename(self.__source_video)
        output_file = os.path.join(self.TMP_DIR, basename)
        self._manager.update_keyframes(output_file)
        self._manager.divide_video_to_segments(self.TMP_DIR)

    def _create_first_screen_from_segments(self):
        segments_files = self._get_segments_files()
        if len(segments_files) < 1:
            return
        for file in segments_files[:-1]:
            basename = os.path.basename(file).split(".")
            basename[-1] = "png"
            basename = ".".join(basename)
            input_file = os.path.join(self.TMP_DIR, file)
            output_file = os.path.join(self.TMP_DIR, basename)
            self._manager.get_first_picture_from_video(input_file, output_file)

    def _get_segments_files(self) -> list[str]:
        return sorted((f for f in os.listdir(self.TMP_DIR) if (f.startswith("cut")) and f.endswith(".mp4")))

    def __iter__(self):
        # Create path
        os.makedirs(self.STORYBOARD_DIR, exist_ok=True)
        os.makedirs(self.SEGMENT_DIR, exist_ok=True)
        # We divide video to segments for economy disk space and then we iterate by segments
        segment_files = self._get_segments_files()
        for i, segment_file in enumerate(segment_files):
            segment_path = os.path.join(self.TMP_DIR, segment_file)
            # Decompose the segment into frames
            self._manager.video_storyboard(segment_path, self.STORYBOARD_DIR)

            png_files = sorted(
                (f for f in os.listdir(self.STORYBOARD_DIR) if f.endswith(".png"))
            )

            # Returning pairs of frames
            for j, png_file in enumerate(png_files[:-1]):
                first = os.path.join(self.STORYBOARD_DIR, png_file)
                second = os.path.join(self.STORYBOARD_DIR, png_files[j + 1])
                yield first, second

            # if we are on the last frame of a segment, but this segment is not the last,
            # then we return the first frame from the next segment.
            if i < len(segment_files) - 2:
                next_segment_name = segment_files[i + 1].rsplit(".", 1)[0] + ".png"
                first = os.path.join(self.STORYBOARD_DIR, png_files[-1])
                second = os.path.join(self.TMP_DIR, next_segment_name)
                yield first, second

            # After iteration by frames, we create new segment in other place with generated frames
            output_video = os.path.join(self.SEGMENT_DIR, segment_file)
            self._manager.encode_video_with_audio(
                self.STORYBOARD_DIR, segment_path, output_video
            )
            # Clearing all created frames for create new frames from next segment
            self._cleanup_storyboard()

        # After iterating through the segments,
        # we created new segments and combine them to create new ones.
        dir_path, original_filename = os.path.split(self.__source_video)
        output_path = os.path.join(dir_path, "dain-" + original_filename)
        self._manager.create_video_from_segments(self.SEGMENT_DIR, output_path)
        # Deleting all new segments from disk space
        self._cleanup_segments()

    def __len__(self):
        return self._manager.frame_count - 1

    def _cleanup_storyboard(self):
        self._cleanup(self.STORYBOARD_DIR)

    def _cleanup_segments(self):
        self._cleanup(self.SEGMENT_DIR)

    def _cleanup(self, dir_path: str):
        for file in os.listdir(dir_path):
            filepath = os.path.join(dir_path, file)
            os.remove(filepath)


class FrameTensor:
    def __init__(
        self,
        tensor: torch.Tensor,
        padding: tuple[int, int, int, int],
        original_shape: tuple[int, int],
    ):
        """
        :param tensor: Tensor Form (1, C, H_padded, W_padded)
        :param padding: (left, right, top, bottom)
        :param original_shape: (height, width) — original size before padding
        """
        assert tensor.dim() == 4, "Expected Tensor as (1, C, H, W)"
        self.tensor = tensor
        self.left, self.right, self.top, self.bottom = padding
        self.height, self.width = original_shape

    def cropped(self) -> torch.Tensor:
        """Return Tensor without padding, shape: (C, H, W)"""
        return self.tensor[
            0, :, self.top : self.top + self.height, self.left : self.left + self.width
        ]

    def clipped(self) -> torch.Tensor:
        """Truncates values in the range [0, 1]"""
        return self.cropped().clamp(0, 1)

    def as_image(self) -> np.ndarray:
        """(H, W, C), dtype=uint8"""
        img = self.clipped().mul(255.0).permute(1, 2, 0).cpu().numpy()
        return np.round(img).astype(np.uint8)


class InterpolationIntermediateFramesInVideo:
    SAVED_MODEL = "./model_weights/best.pth"

    def __init__(
        self,
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self._device.type)
        self.model = self._get_model()

    def run(self, source_video: str):
        iterator = VideoImageIterator(source_video)
        for first_image, second_image in tqdm.tqdm(iterator):
            directory, basename = os.path.split(first_image)
            basename_split = basename.rsplit(".")
            basename_split[0] += "i01"
            interpolate_image = os.path.join(directory, ".".join(basename_split))
            self._interpolate(first_image, second_image, interpolate_image)

    def _interpolate(self, first_image: str, second_image: str, output_image: str):
        x0 = self._load_tensor_from_path(first_image)
        x1 = self._load_tensor_from_path(second_image)
        channel, height, width = x0.size()
        if channel != 3:
            return
        left, right = self._get_padding(width)
        top, bottom = self._get_padding(height)

        pader = torch.nn.ReplicationPad2d((left, right, top, bottom)).to(self._device)

        with torch.no_grad():
            x0 = pader(torch.unsqueeze(x0, 0))
            x1 = pader(torch.unsqueeze(x1, 0))

            y_s, _, _ = self.model(torch.stack((x0, x1), dim=0))
            y_ = y_s[0] # 0 ==> interpolated, 1 ==> rectified

        np_img = FrameTensor(
            y_,
            padding=(left, right, top, bottom),
            original_shape=(height, width),
        ).as_image()
        img = Image.fromarray(np_img)
        img.save(output_image)

    def _get_padding(self, size: int) -> tuple[int, int]:
        if size != ((size >> 7) << 7):
            pad = ((size >> 7) + 1) << 7
            first = (pad - size) // 2
            second = pad - size - first
            return first, second
        return 32, 32

    def _get_np_image(self, image_path: str) -> np.typing.NDArray:
        return np.array(Image.open(image_path))

    def _load_tensor_from_path(self, image_path: str) -> torch.Tensor:
        np_image = self._get_np_image(image_path)
        transpose = np.transpose(np_image, (2, 0, 1)).astype("float32")
        return torch.from_numpy(transpose / 255.0).float().to(self._device)

    def _get_model(self):
        model = networks.DAIN(training=False)
        match self._device.type:
            case "cuda":
                model = model.cuda()
                pretrained_dict = torch.load(
                    self.SAVED_MODEL, map_location=lambda storage, loc: storage
                )
            case _:
                pretrained_dict = torch.load(self.SAVED_MODEL)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        return model.eval()
