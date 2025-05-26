import argparse
import yt_dlp
from roboflow import Roboflow


def sample_youtube(url , dir):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': f'{dir}/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def roboflow(api_key, workspace, project_name, version_number, format="yolov8"):
    
    print("Status : Downloading Datasets [Roboflow]...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(format , location=f'./datasets/{project_name}')
    print("Status : Downloaded Dataset [Roboflow]...")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task runner with named arguments")

    subparsers = parser.add_subparsers(dest="command")

    # Youtbe Downloader Command
    yt_parser = subparsers.add_parser("yt")
    yt_parser.add_argument("--url", type=str, required=True, help="Video Url")
    yt_parser.add_argument("--dir", type=str, default="./samplevideo",help="Age of the person")

    # Roboflow Downloader Command
    rbf_parser = subparsers.add_parser('rbf')
    rbf_parser.add_argument('--api' , required=True, type=str)
    rbf_parser.add_argument('--workspace',required=True, type=str)
    rbf_parser.add_argument('--name' ,required=True, type=str, default="")
    rbf_parser.add_argument('--version',required=True, type=int, default=1)
    rbf_parser.add_argument('--format' , type=str , default="yolov8")
    rbf_parser.add_argument('--output_dir' , type=str , default="./datasets")
    
    args = parser.parse_args()

    if args.command == "yt":
        sample_youtube(args.url, args.dir)
    elif args.command == "rbf":
        roboflow(args.api, args.workspace , args.name , args.version , args.format)
