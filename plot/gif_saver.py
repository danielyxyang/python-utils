import logging
import os

from IPython.display import Image, display

logger = logging.getLogger(__name__)

class GIFSaver():
    def __init__(self, output, filename, dpi=150):
        self.filepath = os.path.join(output, "gif", filename)
        self.dpi = dpi
        self.frame_count = 0

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def filepath_frame(self, i):
        return "{}-{}.png".format(self.filepath, i)

    def filepath_gif(self):
        return "{}.gif".format(self.filepath)

    def add_frame(self, fig):
        fig.savefig(self.filepath_frame(self.frame_count), dpi=self.dpi)
        self.frame_count += 1

    def finish(self, optimize=True, show=True, **kwargs):
        """Finish the GIF by saving and displaying it.

        To create the GIF, you must install the package `imageio` and `tqdm`.
        The GIF is then created using Pillow [1]. To optimize the size of the
        GIF with gifsicle [2], further install the package `pygifsicle`.

        [1] https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        [2] https://github.com/LucaCappelletti94/pygifsicle

        Args:
            optimize (bool, optional): Flag whether to optimize GIF using
                gifsicle. Defaults to True.
            show (bool, optional): Flag whether to show GIF. Defaults to True.
        """
        import imageio.v3 as imageio
        from tqdm import tqdm

        # read gif images
        images = []
        for i in tqdm(range(self.frame_count), desc="Create GIF"):
            images.append(imageio.imread(self.filepath_frame(i)))
        # create gif
        imageio.imwrite(self.filepath_gif(), images, **kwargs)
        # optimize gif
        if optimize:
            import pygifsicle
            pygifsicle.optimize(self.filepath_gif())
        # display gif
        if show:
            display(Image(open(self.filepath_gif(), "rb").read(), width=400))
        logger.info(f"GIF saved to \"{self.filepath_gif()}\".")
