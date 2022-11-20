# Client to stream incoming video to pipeline
import click

from .pipeline import Pipeline

pipe = Pipeline()


@click.command()
@click.option("--video_path", required=True, type=click.Path(exists=True))
@click.option("--output_path", required=True, type=click.Path())
def stream_video(video_path, output_path):
    pipe(video_path, output_path)


if __name__ == "__main__":
    stream_video()
