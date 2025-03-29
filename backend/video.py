import os
import re
import json
import shlex
import logging
import subprocess

from typing import Callable
from decimal import Decimal
from decimal import getcontext


getcontext().prec = 15  # Set high accuracy


logger = logging.getLogger(__name__)


class VideoManagement:
    def __init__(self, path_to_file: str):
        self.__source_file = path_to_file
        self.__frames: int | None = None
        self.__file_with_updated_keyframes: str | None = None
        logger.debug(f"Initial 'VideoManagement' with file: {self.__source_file}")

    @property
    def frame_count(self) -> int:
        """
        Returns frame count from a video using ffprobe
        """
        if self.__frames is None:
            logger.debug("Frame counting using ffprobe...")
            cmd = (
                f"ffprobe -v error -count_frames -select_streams v:0 "
                f"-show_entries stream=nb_read_frames "
                f"-of default=nokey=1:noprint_wrappers=1 {self.__source_file}"
            )
            try:
                result = subprocess.run(
                    shlex.split(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                self.__frames = int(result.stdout.strip())
                logger.info(f"Frame count: {self.__frames}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error ffprobe: {e.stderr.strip()}")
                raise
            except ValueError:
                logger.error(f"Failed to convert ffprobe output to integer")
                raise
        else:
            logger.debug(f"Frame count has been counted earlier: {self.__frames}")
        return self.__frames

    def update_keyframes(
        self,
        output_dir: str,
        progress_bar: Callable[[int, int], None] | None = None,
    ):
        if self.__file_with_updated_keyframes is not None:
            logger.debug("Video with updated keyframes exist")
            return

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.__source_file)
        output_file = os.path.join(output_dir, base_name)
        cmd = (
            f"ffmpeg -i {self.__source_file} -c:v h264_nvenc "
            f"-qp 0 -rc constqp -preset slow -g 150 "
            f"-forced-idr 1 -c:a copy {output_file}"
        )

        logger.info(f"Running transcoding with keyframes every 150 frames...")
        logger.debug(f"Command: {cmd}")
        try:
            process = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            frame_pattern = re.compile(r"frame=\s*(\d+)")
            total_count = self.frame_count
            current = 0
            if progress_bar:
                progress_bar(current, total_count)
            if process.stderr is None:
                logger.error(
                    "Error when try get stderr from process in update_keyframes"
                )
                raise RuntimeError("stderr has not been giving from ffmpeg process")
            for line in process.stderr:
                _match = frame_pattern.search(line)
                if _match and progress_bar:
                    current = int(_match.group(1))
                    progress_bar(current, total_count)
            return_code = process.wait()
            if return_code != 0:
                logger.error(f"Error ffmpeg when updating keyframes")
                raise RuntimeError(f"ffmpeg stop working with error")
            if not os.path.exists(output_file):
                logger.error(
                    "Transcoding video with updated keyframes has not been created"
                )
                raise FileExistsError("The resulting video was not found.")
            self.__file_with_updated_keyframes = output_file
            logger.info(f"Video with updated keyframes has been created: {output_file}")
        except Exception as e:
            logger.exception("Error when update keyframes in the video")
            raise

    def divide_video_to_segments(
        self,
        output_dir: str,
        progress_bar: Callable[[int, int], None] | None = None,
    ):
        if self.__file_with_updated_keyframes is None:
            raise FileNotFoundError("Video with updated keyframes has not been created")

        os.makedirs(output_dir, exist_ok=True)

        # Get keyframes
        ffprobe_cmd = (
            f"ffprobe -loglevel error -select_streams v "
            f"-show_frames -show_entries frame=key_frame,best_effort_timestamp_time "
            f'-of json "{self.__file_with_updated_keyframes}"'
        )

        logger.debug(f"Request get keyframes: {ffprobe_cmd}")

        try:
            result = subprocess.run(
                shlex.split(ffprobe_cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            timestamps = []
            for frame in data.get("frames", []):
                if frame.get("key_frame") == 1:
                    if ts := frame.get("best_effort_timestamp_time"):
                        timestamps.append(ts)

            logger.info(f"Keyframes found: {len(timestamps)}")
            if progress_bar:
                progress_bar(1, 2)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error when called ffprobe: {e.stderr.strip()}")
            raise

        if not timestamps:
            logger.error(f"Timestamps is empty")
            raise RuntimeError(f"Keyframes not found")

        # Remove first timestamp and convert to text
        segment_times = ",".join(timestamps[1:])

        output_pattern = os.path.join(output_dir, "cut_%03d.mp4")
        ffmpeg_cmd = (
            f"ffmpeg -y -i {self.__file_with_updated_keyframes} -c copy -map 0 "
            f"-f segment -segment_time 99999 -segment_times {segment_times} "
            f"-reset_timestamps 1 {output_pattern}"
        )
        logger.debug(f"Ffmpeg command to split video: {ffmpeg_cmd}")

        try:
            process = subprocess.Popen(
                shlex.split(ffmpeg_cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            return_code = process.wait()
            if return_code != 0:
                logger.error("Ffmpeg stop working with error when try splitting video")
                raise RuntimeError("Ffmpeg stop working with error")

            logger.info(
                f"Splitting video finished successfully: Files saved in: {output_dir}"
            )
            if progress_bar:
                progress_bar(2, 2)

        except Exception as e:
            logger.exception("Error when splitting video to segments")
            raise

    def video_storyboard(self, frame_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        cmd = f"ffmpeg -i {frame_path} " f"start_number 0 {output_dir}/frame_%06d.png"
        try:
            result = subprocess.run(
                shlex.split(cmd),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Ffmpeg make storyboard with file '{frame_path}' end with error: {e.stderr.strip()}"
            )
            raise

    def get_audio_duration(self, video_path: str) -> Decimal:
        cmd = (
            f"ffprobe -v error -select_streams a:0 "
            f"-show_entries stream=duration "
            f'-of default=nokey=1:noprint_wrappers=1 "{video_path}"'
        )
        try:
            result = subprocess.run(
                shlex.split(cmd),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Ffprobe can not get duration from audio: {e.stderr.strip()}")
            raise
        return Decimal(result.stdout.strip())

    def get_video_duration(self, video_path: str) -> Decimal:
        cmd = (
            f"ffprobe -v error -select_streams v:0 "
            f"-show_entries stream=duration "
            f"-of default=nokey=1:noprint_wrappers=1 {video_path}"
        )
        try:
            result = subprocess.run(
                shlex.split(cmd),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Ffprobe can not get duration from video: {e.stderr.strip()}")
            raise
        return Decimal(result.stdout.strip())

    def extract_audio(self, video_path: str, audio_output: str):
        cmd = f"ffmpeg -y -i {video_path} -q:a 0 -map a {audio_output}"
        try:
            subprocess.run(
                shlex.split(cmd),
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Ffmpeg can not extract audio from video: {e.stderr.strip()}")
            raise

    def encode_video_with_audio(
        self,
        frames_dir: str,
        original_video: str,
        output_video: str,
    ):
        frame_files = sorted((f for f in os.listdir(frames_dir) if f.endswith(".png")))
        num_frames = Decimal(len(frame_files))
        if num_frames == 0:
            logger.error(f"png files does not exist in frame dir '{frames_dir}'")
            raise RuntimeError(f"Not PNG frames into a frames dir: {frames_dir}")
        audio_temp = os.path.join(frames_dir, "temp.aac")
        self.extract_audio(original_video, audio_temp)
        try:
            duration = self.get_video_duration(original_video)
        except subprocess.CalledProcessError as e:
            os.remove(audio_temp)
            raise

        fps = num_frames / duration
        fps_str = str(fps.quantize(Decimal("1.0000000000")))

        input_txt_path = os.path.join(frames_dir, "input.txt")
        with open(input_txt_path, "w") as f:
            for frame in frame_files:
                frame_path = os.path.join(frames_dir, frame)
                f.write(f"file '{os.path.abspath(frame_path)}'")

        cmd = (
            f"ffmpeg -y -r {fps_str} -f concat -safe 0 -i {input_txt_path} "
            f"-i {audio_temp} "
            f"-c:v h264_nvenc -rc constqp -qp 0 "
            f"-c:a copy {output_video}"
        )
        try:
            subprocess.run(
                shlex.split(cmd),
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Ffmpeg error when tried create video from storyboard: {e.stderr.strip()}"
            )
            raise
        finally:
            os.remove(audio_temp)
            os.remove(input_txt_path)

        # Checking created video has expected length video and audio part
        final_video_duration = self.get_video_duration(output_video)
        expected_duration = self.get_video_duration(original_video)

        final_audio_duration = self.get_audio_duration(output_video)
        expected_audio_duration = self.get_audio_duration(original_video)

        video_delta = abs(final_video_duration - expected_duration)
        audio_delta = abs(final_audio_duration - expected_audio_duration)
        if video_delta > Decimal("0.01") or audio_delta > Decimal("0.01"):
            logger.warning(
                f"Input video and Final video has different length. "
                f"Audio delta: {audio_delta}. "
                f"Video delta: {video_delta}."
            )
