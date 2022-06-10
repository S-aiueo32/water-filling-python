from __future__ import annotations

from pathlib import Path

import click
import cv2
import numpy as np


@click.command()
@click.argument("input", type=click.Path())
@click.option("--output", type=click.Path(), default=None)
def cmd(input: str, output: str | None):  # type: ignore
    from .jung_2019 import WaterFilling

    input: Path = Path(input)  # type: ignore
    output: Path = output or input.with_name(f"{input.stem}_filled{input.suffix}")  # type: ignore

    water_filling = WaterFilling()

    input_image: np.ndarray = cv2.imread(str(input), cv2.IMREAD_COLOR)  # type: ignore
    output_image = water_filling(input_image)

    cv2.imwrite(str(output), output_image)


if __name__ == "__main__":
    cmd()
