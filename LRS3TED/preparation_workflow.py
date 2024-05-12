import argparse
import os
from multiprocessing import Pool
import subprocess
import glob
import tqdm


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


def process_video(video_id, output_dir, skip_exist, logging):
    download_dir = os.path.join(output_dir, "_videos_raw")
    if skip_exist and os.path.exists(os.path.join(download_dir, f"{video_id}.*")):
        return True
    return download_video(video_id, download_dir, logging)


def process_video_wrapper(kwargs):
    return process_video(**kwargs)


def collect_video_info(base_dir):
    video_ids = []
    for video_dir in glob.glob(os.path.join(base_dir, "*", "*")):
        path, video_id = os.path.relpath(video_dir, base_dir).split(os.sep)
        video_ids.append((path, video_id))
    return video_ids


def process_dataset(source_dir, output_dir, num_workers, skip_exist):
    video_infos = collect_video_info(source_dir)
    task_kwargs = [
        dict(
            video_id=video_id,
            output_dir=os.path.join(output_dir, path),
            skip_exist=skip_exist,
            logging=True,
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
    parser = argparse.ArgumentParser(description="Download LRS3-TED dataset")
    parser.add_argument("-s", "--source_dir", type=str, required=True, help="Path to the directory with the dataset")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Where to save the videos?")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="Number of workers for downloading")
    parser.add_argument("-skip", "--skip_exist", action="store_true", help="Skip downloading if file already exists")
    args = parser.parse_args()

    process_dataset(args.source_dir, args.output_dir, args.num_workers, args.skip_exist)
