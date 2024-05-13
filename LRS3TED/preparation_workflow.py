import argparse
import csv
import glob
import io
import itertools
import os
import subprocess
from multiprocessing import Pool

import cv2
import tqdm


def condition_continuous(x):
    ix = iter(x)
    pv = next(ix)
    for v in ix:
        assert pv + 1 == v
        pv = v


def condition_increase(x):
    ix = iter(x)
    pv = next(ix)
    for v in ix:
        assert pv < v
        pv = v


def condition_find_video(download_dir, video_id):
    pattern = os.path.join(download_dir, f"{video_id}.*")
    files = glob.glob(pattern)
    return len(files) > 0


def find_video_file(download_dir, video_id):
    video_extensions = ["mp4", "avi", "mov", "mkv", "webm"]
    for ext in video_extensions:
        path = os.path.join(download_dir, f"{video_id}.{ext}")
        if os.path.exists(path):
            return path
    return None


def download_video(video_id, download_dir, logging=False):
    """
    Downloads a video from YouTube using the `yt-dlp` utility.

    :param video_id: YouTube ID of the video.
    :param download_dir: Path where the video will be saved.
    :param logging: Boolean indicating whether to log details to a file.
    :return: Tuple of (boolean indicating success, path to the downloaded video)
    """
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(os.path.join(download_dir, "logs"), exist_ok=True)
    log_path = os.path.join(download_dir, "logs", f"{video_id}.log")
    stderr = open(log_path, "a") if logging else subprocess.PIPE

    command = ["yt-dlp", "--quiet", "-f", "best[ext=mp4]", "--output", f"{download_dir}/{video_id}.%(ext)s", "--no-continue", video_id]
    process = subprocess.run(command, stderr=stderr, stdout=subprocess.PIPE, text=True)
    success = process.returncode == 0

    if logging:
        stderr.close()

    return success


def collect_clip_info(file_path):
    """
    Collects clipping frame information from a given text file.

    :param file_path: Path to the text file containing frame data.
    :return: Dictionary containing video_id and a list of frames with their properties.
    """
    clip_info = {}

    try:
        with open(file_path, "r") as file:
            meta, frame, *_ = file.read().split("\n\n")

        for line in meta.split("\n"):
            k, v = line.split(":")
            clip_info[k] = v.strip()

        f = io.StringIO(frame.replace(" \t", ",").replace(" \n", "\n"))
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        assert header == "FRAME X Y W H".split()
        clip_info.update({k: list(map(t, v)) for k, v, t in zip(header, zip(*csv_reader), [int, float, float, float, float])})

        # DUMMY CODE: Some files do not have third section
        # f = io.StringIO(phoneme)
        # csv_reader = csv.reader(f, delimiter=" ")
        # header = next(csv_reader)
        # assert header == "WORD START END ASDSCORE".split()
        # clip_info.update({k:list(map(t, v)) for k, v, t in zip(header, zip(*csv_reader), [str, float, float, float])})

    except Exception as e:
        print(f"Error processing the file {file_path}: {e}")

    return clip_info


def process_video(video_id, clip_dir, output_dir, skip_exist, logging, fps):
    download_dir = os.path.join(output_dir, "_videos_raw")
    video_path = find_video_file(download_dir, video_id)
    if not skip_exist or video_path is None:
        success = download_video(video_id, download_dir, logging)
        if not success:
            raise Exception(f"fail to download {video_id}")
        video_path = find_video_file(download_dir, video_id)

    clip_infos = {}
    for path in sorted(glob.glob(os.path.join(clip_dir, "*.txt"))):
        clip_id, _ = os.path.splitext(os.path.basename(path))
        clip_info = collect_clip_info(path)
        clip_infos[clip_id] = clip_info
        condition_continuous(clip_info["FRAME"])
    condition_continuous(map(int, clip_infos.keys()))
    condition_increase(map(lambda x: x["FRAME"][0], clip_infos.values()))

    clip_iter = iter(clip_infos.items())
    clip_id, clip_info = next(clip_iter)
    clip_frames = []

    cap = cv2.VideoCapture(video_path)

    # DUMMY CODE: some video response as it have 0 frames
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # assert list(clip_infos.values())[-1]["FRAME"][-1] < frame_count

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frames = itertools.takewhile(lambda ret_frame: ret_frame[0], (cap.read() for _ in itertools.count()))
    for frame_raw_id, (_, frame) in enumerate(frames):
        i = int(frame_raw_id * fps // frame_rate)
        if clip_info["FRAME"][-1] < i:
            clip = next(clip_iter, None)
            if clip is None:
                break
            clip_id, clip_info = clip
            clip_frames = []
        elif clip_info["FRAME"][0] <= i:
            rel_i = i - clip_info["FRAME"][0]
            height, width, _ = frame.shape
            X, Y, W, H = map(lambda x: clip_info[x][rel_i], "XYWH")
            cropped_frame = frame[int(Y * height) : int((Y + H) * height), int(X * width) : int((X + W) * width)]
            clip_frames.append(cropped_frame)

        if clip_info["FRAME"][0] == i:
            start_time = frame_raw_id / frame_rate
        if i == clip_info["FRAME"][-1]:
            end_time = (frame_raw_id + 1) / frame_rate

            height = max(frame.shape[0] for frame in clip_frames)
            width = max(frame.shape[1] for frame in clip_frames)
            vonly_path = os.path.join(output_dir, f"{video_id}_{clip_id}_video.mp4")
            aonly_path = os.path.join(output_dir, f"{video_id}_{clip_id}.aac")
            final_path = os.path.join(output_dir, f"{video_id}_{clip_id}.mp4")
            out = cv2.VideoWriter(vonly_path, fourcc, frame_rate, (width, height))
            for frame in clip_frames:
                resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                out.write(resized)
            out.release()
            assert 0 == subprocess.call(f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -vn -acodec copy {aonly_path}", shell=True)
            assert 0 == subprocess.call(f"ffmpeg -i {vonly_path} -i {aonly_path} -c:v copy -c:a copy -shortest {final_path}", shell=True)
            os.remove(vonly_path)
            os.remove(aonly_path)
    cap.release()


def process_video_wrapper(kwargs):
    return process_video(**kwargs)


def collect_video_info(base_dir):
    video_ids = []
    for video_dir in glob.glob(os.path.join(base_dir, "*", "*")):
        path, video_id = os.path.relpath(video_dir, base_dir).split(os.sep)
        video_ids.append((path, video_id))
    return video_ids


def process_dataset(source_dir, output_dir, num_workers, skip_exist, fps):
    video_infos = collect_video_info(source_dir)
    task_kwargs = [
        dict(
            video_id=video_id,
            clip_dir=os.path.join(source_dir, path, video_id),
            output_dir=os.path.join(output_dir, path),
            skip_exist=skip_exist,
            logging=True,
            fps=fps,
        )
        for path, video_id in video_infos
    ]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs), desc=f"Downloading videos into {output_dir}")
    for _ in tqdm.tqdm(pool.imap_unordered(process_video_wrapper, task_kwargs), **tqdm_kwargs):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to download and process videos for the LRS3-TED dataset.")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory containing the dataset's raw files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save processed videos and metadata.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for processing the dataset.")
    parser.add_argument("--skip-exist", action="store_true", help="Skip processing videos that already exist in the output directory.")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second to use as reference for video processing.")
    args = parser.parse_args()

    process_dataset(args.source_dir, args.output_dir, args.num_workers, args.skip_exist, args.fps)
