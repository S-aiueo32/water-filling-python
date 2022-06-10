import cv2
import numpy as np
from scipy.signal import convolve2d


def relu(input: np.ndarray) -> np.ndarray:
    input[input < 0] = 0
    return input


def inv_relu(input: np.ndarray) -> np.ndarray:
    input[input > 0] = 0
    return input


class WaterFilling:
    four_neighbours_filter: np.ndarray = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    def __init__(self, neta: float = 0.2, k_s: float = 0.2, l: float = 0.85) -> None:
        self.neta = neta
        self.k_s = k_s
        self.l = l

    def __call__(self, image: np.ndarray) -> np.ndarray:
        Y, Cr, Cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).transpose((2, 0, 1))
        G = self.flood(Y)
        Y_new = self.incremental(G, Y)
        return cv2.cvtColor(np.stack([Y_new, Cr, Cb], axis=-1), cv2.COLOR_YCrCb2BGR)  # type: ignore

    @staticmethod
    def downsample(Y: np.ndarray, k_s: float) -> np.ndarray:
        height, width = map(lambda x: int(x * k_s), Y.shape)
        Y_bar = cv2.resize(Y, (width, height), interpolation=cv2.INTER_LINEAR)
        return Y_bar  # type: ignore

    def flood(self, Y: np.ndarray, n_iter: int = 2500) -> np.ndarray:
        Y_bar = self.downsample(Y.copy().astype(np.float32), self.k_s)
        w = np.zeros_like(Y_bar, dtype=np.float32)

        for t in range(n_iter):
            G = Y_bar + w
            h_hat = G.max()

            w_psi = np.exp(-t) * (h_hat - G)
            w_phi = (
                inv_relu(np.roll(G, shift=1, axis=0) - G)
                + inv_relu(np.roll(G, shift=-1, axis=0) - G)
                + inv_relu(np.roll(G, shift=1, axis=1) - G)
                + inv_relu(np.roll(G, shift=-1, axis=1) - G)
            )

            w_new = relu(w_psi + self.neta * w_phi + w)
            w[1:-1, 1:-1] = w_new[1:-1, 1:-1]

        output = cv2.resize(G, Y.shape[::-1], interpolation=cv2.INTER_LINEAR)  # type: ignore
        output = output.astype(np.uint8)

        return output

    def incremental(
        self, G: np.ndarray, I: np.ndarray, n_iters: int = 100
    ) -> np.ndarray:
        G = G.copy().astype(np.float32)
        I = I.copy().astype(np.float32)

        w = np.zeros_like(G, dtype=np.float32)

        for _ in range(n_iters):
            _G = G + w
            J_w = convolve2d(_G, self.four_neighbours_filter, mode="same") - 4 * _G
            w[1:-1, 1:-1] = relu(self.neta * J_w + w)[1:-1, 1:-1]

        output = (I / (G + w)) * self.l * 255
        output = output.clip(0, 255.0)
        output = output.astype(np.uint8)

        return output
