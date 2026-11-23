"""Девять классических признаков резкости.

Перенесены из BASED почти без изменений. Каждый признак считает один и тот же
оператор на размытом входе и на результате и сравнивает отклики.
Картинки везде uint8 BGR, как их отдаёт cv2.imread.

В комментариях оставлены заметки о том, что пробовалось при подборе
(в скобках изменение корреляции Кендалла на BASED).
"""

import cv2
import numpy as np
import skimage.feature
import skimage.filters
from skimage.metrics import structural_similarity

NAMES = ["laplac", "fft", "gabor", "hough", "hog", "ssim_m", "sobel", "lbp", "reblur"]


def _gray128(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 128))


def laplac(blur, res):
    a = cv2.Laplacian(cv2.resize(blur, (128, 128)), cv2.CV_64F, ksize=3)
    b = cv2.Laplacian(cv2.resize(res, (128, 128)), cv2.CV_64F, ksize=3)
    return np.linalg.norm(a - b)


def _highpass(gray, size):
    # вырезаем квадрат низких частот вокруг центра спектра и возвращаемся обратно
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    spec = np.fft.fftshift(np.fft.fft2(gray))
    spec[cy - size:cy + size, cx - size:cx + size] = 0
    rec = np.fft.ifft2(np.fft.ifftshift(spec))
    # 1e-8 чтобы на плоских кропах не получить log(0)
    return np.log(np.abs(rec) + 1e-8)


def fft(blur, res, size=30):
    return np.linalg.norm(_highpass(_gray128(blur), size) - _highpass(_gray128(res), size))


def _gabor_stats(gray):
    real, _ = skimage.filters.gabor(
        gray, frequency=0.15, theta=np.pi / 3, sigma_x=3, sigma_y=3, mode="wrap"
    )
    return np.array(cv2.meanStdDev(real))


def gabor(blur, res):
    # уменьшение в 4 раза качество не меняет (-0.006)
    # три частоты (0.05, 0.10, 0.15) вместо одной хуже (-0.02)
    return -np.linalg.norm(_gabor_stats(_gray128(blur)) - _gabor_stats(_gray128(res)))


def _hough_lines(img):
    edges = cv2.Canny(img, 150, 255)
    lines = cv2.HoughLinesP(edges, 200, np.pi / 3, 150, None, 0, 0)
    canvas = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            # толстые линии работают лучше тонких
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=5)
    return canvas


def hough(blur, res):
    return np.linalg.norm(_hough_lines(blur).astype(np.float64) - _hough_lines(res))


_hog = cv2.HOGDescriptor(
    (64, 64), (16, 16), (8, 8), (4, 4), _nbins=13,
    _derivAperture=1, _gammaCorrection=True, _L2HysThreshold=0.1,
)


def hog(blur, res):
    # перевод в серое (+0.05), уменьшение в 4 раза (+0.04)
    return np.linalg.norm(_hog.compute(_gray128(blur)) - _hog.compute(_gray128(res)))


def ssim_m(blur, res):
    # SSIM по каналам YUV с весами 6:1:1
    a = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)
    b = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
    y, u, v = (structural_similarity(a[..., i], b[..., i]) for i in range(3))
    return 6 * y + u + v


def _sobel(img):
    gx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    gy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)
    return grad


def sobel(blur, res):
    # большое ядро (+0.05), гистограмма границ вместо нормы (-0.04),
    # уменьшение в 4 раза (-0.05), SSIM вместо нормы (-0.12)
    a = cv2.equalizeHist(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))
    b = cv2.equalizeHist(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY))
    return np.linalg.norm(_sobel(a) - _sobel(b))


def _lbp(img):
    # LBP считается по карте границ, а не по самой картинке (+0.09)
    edges = np.rint(_sobel(img)).astype(np.uint8)
    gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    return skimage.feature.local_binary_pattern(gray, P=4, R=8, method="uniform")


def lbp(blur, res):
    # подбор P и R дал +0.11, гистограмма паттернов хуже (-0.06)
    return -np.linalg.norm(_lbp(blur) - _lbp(res))


def _reblur(img):
    # размываем ещё раз и смотрим, сколько линий пропало.
    # у резкой картинки пропадает много, у размытой почти ничего
    again = cv2.GaussianBlur(img, (17, 17), 0)
    return 1 + np.linalg.norm(_hough_lines(img).astype(np.float64) - _hough_lines(again))


def reblur(blur, res):
    return abs(_reblur(blur) - _reblur(res))


_FUNCS = [laplac, fft, gabor, hough, hog, ssim_m, sobel, lbp, reblur]


def compute(blur, res):
    """Вектор из девяти признаков для пары (размытый вход, результат)."""
    return np.array([f(blur, res) for f in _FUNCS], dtype=np.float32)


def squash(x):
    # признаки отличаются на порядки (ssim около 8, sobel тысячи),
    # поэтому перед сетью сжимаем их знаковым логарифмом
    return np.sign(x) * np.log1p(np.abs(x))
