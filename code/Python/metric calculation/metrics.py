import torch
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from torch.nn.functional import conv2d


def _cross_entropy(img1, fused):
    P1 = np.histogram(img1.flatten(), range(0, 257), density=True)[0]
    P2 = np.histogram(fused.flatten(), range(0, 257), density=True)[0]

    result = 0
    for k in range(256):
        if P1[k] != 0 and P2[k] != 0:
            result += P1[k] * np.log2(P1[k] / P2[k])

    return result


def _mutinf(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    M, N = a.shape

    # Initialize histograms
    hab = np.zeros((256, 256))
    hab_ = np.zeros((256, 256))
    ha = np.zeros(256)
    hb = np.zeros(256)

    # Normalize
    a = (a - a.min()) / (a.max() - a.min()) if a.max() != a.min() else np.zeros((M, N))
    b = (b - b.min()) / (b.max() - b.min()) if b.max() != b.min() else np.zeros((M, N))

    a = (a * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)

    # Ensure values are in the correct range
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Build histograms
    for i in range(M):
        for j in range(N):
            index_x = a[i, j]
            index_y = b[i, j]
            hab[index_x, index_y] += 1  # Joint histogram
            ha[index_x] += 1  # Histogram for a
            hb[index_y] += 1  # Histogram for b

    # Calculate joint entropy
    hsum = np.sum(hab)
    index = hab != 0
    p = hab[index] / hsum
    Hab = -np.sum(p * np.log2(p))

    # Calculate entropy for a
    hsum = np.sum(ha)
    index = ha != 0
    p = ha[index] / hsum
    Ha = -np.sum(p * np.log2(p))

    # Calculate entropy for b
    hsum = np.sum(hb)
    index = hb != 0
    p = hb[index] / hsum
    Hb = -np.sum(p * np.log2(p))

    # Calculate mutual information
    return Ha + Hb - Hab


def _filter(win_size, sigma, dtype, device):
    # This code is inspired by
    # https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/utils.py#L45
    # https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/functional/filters.py#L38
    # Both links do the same, but the second one is cleaner
    coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
    g = coords**2
    g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * sigma**2))
    g /= torch.sum(g)
    return g


def ag(image):
    Gx, Gy = np.zeros_like(image), np.zeros_like(image)

    Gx[:, 0] = image[:, 1] - image[:, 0]
    Gx[:, -1] = image[:, -1] - image[:, -2]
    Gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2

    Gy[0, :] = image[1, :] - image[0, :]
    Gy[-1, :] = image[-1, :] - image[-2, :]
    Gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
    return np.mean(np.sqrt((Gx**2 + Gy**2) / 2))


def compute_gradients_and_orientations(image, h1, h3):
    gx = conv2d(image, h3, padding=1)
    gy = conv2d(image, h1, padding=1)
    g = torch.sqrt(gx**2 + gy**2)

    # avoiding division by zero error
    a = torch.atan2(gy, gx)
    a[torch.isnan(a)] = np.pi / 2  # handling the case when gx is 0

    return g, a


def compute_quality(g1, a1, g2, a2, gF, aF, Tg, Ta, kg, ka, Dg, Da):
    G = torch.where(g1 > gF, gF / g1, torch.where(g1 == gF, gF, g1 / gF))
    A = 1 - torch.abs(a1 - aF) / (np.pi / 2)
    Qg = Tg / (1 + torch.exp(kg * (G - Dg)))
    Qa = Ta / (1 + torch.exp(ka * (A - Da)))
    return Qg * Qa


def cross_entropy(image_F, image_A, image_B):
    cross_entropy_VI = _cross_entropy(image_A, image_F)
    cross_entropy_IR = _cross_entropy(image_B, image_F)
    return (cross_entropy_VI + cross_entropy_IR) / 2.0


def edge_intensity(image_F):
    pF = torch.from_numpy(image_F).float().unsqueeze(0).unsqueeze(0)
    h1 = (
        torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h3 = (
        torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode="replicate")
    conv.weight.data = h3
    gx = conv(pF)
    conv.weight.data = h1
    gy = conv(pF)
    res = torch.mean((gx**2 + gy**2) ** 0.5)

    return res.item()


def entropy(image):
    res = np.histogram(image.flatten(), range(0, 257), density=True)[0]
    res = torch.from_numpy(res)
    res = res[res != 0]
    res = torch.sum(-res * res.log2())
    return res


def getvif(preds, target, sigma_n_sq=2.0):
    preds = torch.from_numpy(preds).float()
    target = torch.from_numpy(target).float()
    dtype = preds.dtype
    device = preds.device

    preds = preds.unsqueeze(0).unsqueeze(0)  # Add channel dimension
    target = target.unsqueeze(0).unsqueeze(0)
    # Constant for numerical stability
    eps = torch.tensor(1e-10, dtype=dtype, device=device)

    sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)

    preds_vif, target_vif = torch.zeros(1, dtype=dtype, device=device), torch.zeros(
        1, dtype=dtype, device=device
    )
    for scale in range(4):
        n = 2.0 ** (4 - scale) + 1
        kernel = _filter(n, n / 5, dtype=dtype, device=device)[None, None, :]

        if scale > 0:
            target = conv2d(target.float(), kernel)[:, :, ::2, ::2]
            preds = conv2d(preds.float(), kernel)[:, :, ::2, ::2]

        mu_target = conv2d(target, kernel)
        mu_preds = conv2d(preds, kernel)
        mu_target_sq = mu_target**2
        mu_preds_sq = mu_preds**2
        mu_target_preds = mu_target * mu_preds

        if scale == 0:
            target = target.byte()
            preds = preds.byte()
        sigma_target_sq = torch.clamp(
            conv2d((target**2).float(), kernel) - mu_target_sq, min=0.0
        )
        sigma_preds_sq = torch.clamp(
            conv2d((preds**2).float(), kernel) - mu_preds_sq, min=0.0
        )
        sigma_target_preds = conv2d((target * preds).float(), kernel) - mu_target_preds

        g = sigma_target_preds / (sigma_target_sq + eps)
        sigma_v_sq = sigma_preds_sq - g * sigma_target_preds

        mask = sigma_target_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        sigma_target_sq[mask] = 0

        mask = sigma_preds_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = 0

        mask = g < 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        g[mask] = 0
        sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)

        preds_vif_scale = torch.log10(
            1.0 + (g**2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq)
        )
        preds_vif = preds_vif + torch.sum(preds_vif_scale, dim=[1, 2, 3])
        target_vif = target_vif + torch.sum(
            torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3]
        )
    vif = preds_vif / target_vif
    if torch.isnan(vif):
        return 1.0
    else:
        return vif.item()


def mse(image_F, image_A, image_B):
    return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2


def mutinf(image_F, image_A, image_B):
    mi_VI = _mutinf(image_A, image_F)
    mi_IR = _mutinf(image_B, image_F)
    return mi_VI + mi_IR


def psnr(image_F, image_A, image_B):
    return 10 * np.log10(np.max(image_F) ** 2 / mse(image_F, image_A, image_B))


def qabf(image_F, image_A, image_B):
    pA = torch.from_numpy(image_A).float().unsqueeze(0).unsqueeze(0)
    pB = torch.from_numpy(image_B).float().unsqueeze(0).unsqueeze(0)
    pF = torch.from_numpy(image_F).float().unsqueeze(0).unsqueeze(0)

    h1 = (
        torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h2 = (
        torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h3 = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    gA, aA = compute_gradients_and_orientations(pA, h1, h3)
    gB, aB = compute_gradients_and_orientations(pB, h1, h3)
    gF, aF = compute_gradients_and_orientations(pF, h1, h3)

    L = 1
    Tg, Ta = 0.9994, 0.9879
    kg, ka = -15, -22
    Dg, Da = 0.5, 0.8

    QAF = compute_quality(gA, aA, gB, aB, gF, aF, Tg, Ta, kg, ka, Dg, Da)
    QBF = compute_quality(gB, aB, gA, aA, gF, aF, Tg, Ta, kg, ka, Dg, Da)

    wA = gA**L
    wB = gB**L
    deno = torch.sum(wA + wB)
    nume = torch.sum(QAF * wA + QBF * wB)
    output = nume / deno

    return output.item()


def qcb(image_F, image_A, image_B):
    image_A = image_A.astype(np.float64)
    image_B = image_B.astype(np.float64)
    image_F = image_F.astype(np.float64)
    image_A = (
        (image_A - image_A.min()) / (image_A.max() - image_A.min())
        if image_A.max() != image_A.min()
        else image_A
    )
    image_A = np.round(image_A * 255).astype(np.uint8)
    image_B = (
        (image_B - image_B.min()) / (image_B.max() - image_B.min())
        if image_B.max() != image_B.min()
        else image_B
    )
    image_B = np.round(image_B * 255).astype(np.uint8)
    image_F = (
        (image_F - image_F.min()) / (image_F.max() - image_F.min())
        if image_F.max() != image_F.min()
        else image_F
    )
    image_F = np.round(image_F * 255).astype(np.uint8)

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622
    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001

    M, N = image_A.shape

    # Use the correct meshgrid for frequency space
    u, v = np.meshgrid(np.fft.fftfreq(N, 0.5), np.fft.fftfreq(M, 0.5))
    u *= N / 30
    v *= M / 30
    r = np.sqrt(u**2 + v**2)
    Sd = np.exp(-((r / f0) ** 2)) - a * np.exp(-((r / f1) ** 2))

    # Ensure Sd matches the shape of the images
    Sd = Sd[:M, :N]  # This should ensure proper matching

    # Fourier Transform
    fused1 = np.fft.ifft2(np.fft.fft2(image_A) * Sd).real
    fused2 = np.fft.ifft2(np.fft.fft2(image_B) * Sd).real
    ffused = np.fft.ifft2(np.fft.fft2(image_F) * Sd).real

    x = np.linspace(-15, 15, 31)
    y = np.linspace(-15, 15, 31)
    X, Y = np.meshgrid(x, y)
    sigma = 2
    G1 = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    sigma = 4
    G2 = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    G1 = torch.from_numpy(G1).float().unsqueeze(0).unsqueeze(0)
    G2 = torch.from_numpy(G2).float().unsqueeze(0).unsqueeze(0)
    fused1 = torch.from_numpy(fused1).float().unsqueeze(0).unsqueeze(0)
    fused2 = torch.from_numpy(fused2).float().unsqueeze(0).unsqueeze(0)
    ffused = torch.from_numpy(ffused).float().unsqueeze(0).unsqueeze(0)

    buff = conv2d(fused1, G1, padding=15)
    buff1 = conv2d(fused1, G2, padding=15)
    contrast_value = buff / buff1 - 1
    contrast_value = torch.abs(contrast_value)
    C1P = (k * (contrast_value**p)) / (h * (contrast_value**q) + Z)
    buff = conv2d(fused2, G1, padding=15)
    buff1 = conv2d(fused2, G2, padding=15)
    contrast_value = buff / buff1 - 1
    contrast_value = torch.abs(contrast_value)
    C2P = (k * (contrast_value**p)) / (h * (contrast_value**q) + Z)
    buff = conv2d(ffused, G1, padding=15)
    buff1 = conv2d(ffused, G2, padding=15)
    contrast_value = buff / buff1 - 1
    contrast_value = torch.abs(contrast_value)
    CfP = (k * (contrast_value**p)) / (h * (contrast_value**q) + Z)

    mask = C1P < CfP
    Q1F = CfP / C1P
    Q1F[mask] = (C1P / CfP)[mask]
    mask = C2P < CfP
    Q2F = CfP / C2P
    Q2F[mask] = (C2P / CfP)[mask]

    ramda1 = (C1P**2) / (C1P**2 + C2P**2)
    ramda2 = (C2P**2) / (C1P**2 + C2P**2)

    Q = ramda1 * Q1F + ramda2 * Q2F
    return Q.mean().item()


def qcv(image_F, image_A, image_B):
    alpha_c = 1
    alpha_s = 0.685
    f_c = 97.3227
    f_s = 12.1653

    window_size = 16
    alpha = 5

    # Preprocessing
    
    image_A = image_A.astype(np.float64)
    image_B = image_B.astype(np.float64)
    image_F = image_F.astype(np.float64)
    image_A = (
        (image_A - image_A.min()) / (image_A.max() - image_A.min())
        if image_A.max() != image_A.min()
        else image_A
    )
    image_A = np.round(image_A * 255)
    image_B = (
        (image_B - image_B.min()) / (image_B.max() - image_B.min())
        if image_B.max() != image_B.min()
        else image_B
    )
    image_B = np.round(image_B * 255)
    image_F = (
        (image_F - image_F.min()) / (image_F.max() - image_F.min())
        if image_F.max() != image_F.min()
        else image_F
    )
    image_F = np.round(image_F * 255)

    h1 = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h3 = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # Step 1: Extract Edge Information
    pA = torch.from_numpy(image_A).float().unsqueeze(0).unsqueeze(0)
    pB = torch.from_numpy(image_B).float().unsqueeze(0).unsqueeze(0)

    img1X = conv2d(pA, h3, padding=1)
    img1Y = conv2d(pA, h1, padding=1)
    im1G = (img1X**2 + img1Y**2)**0.5

    img2X = conv2d(pB, h3, padding=1)
    img2Y = conv2d(pB, h1, padding=1)
    im2G = (img2X**2 + img2Y**2)**0.5

    M, N = image_A.shape
    ramda1 = conv2d(im1G**alpha, torch.ones(1, 1, window_size, window_size, dtype=im1G.dtype), stride=window_size)
    ramda2 = conv2d(im2G**alpha, torch.ones(1, 1, window_size, window_size, dtype=im1G.dtype), stride=window_size)

    # Similarity Measurement
    f1 = image_A - image_F
    f2 = image_B - image_F

    u, v = np.meshgrid(np.fft.fftfreq(N, 0.5), np.fft.fftfreq(M, 0.5))
    u *= N/8
    v *= M/8
    r = np.sqrt(u**2 + v**2)

    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-((0.144 * r) ** 1.1))

    Df1 = np.fft.ifft2(np.fft.fft2(f1) * theta_m).real
    Df2 = np.fft.ifft2(np.fft.fft2(f2) * theta_m).real

    Df1 = torch.from_numpy(Df1).float().unsqueeze(0).unsqueeze(0)
    Df2 = torch.from_numpy(Df2).float().unsqueeze(0).unsqueeze(0)

    D1 = conv2d(Df1**2, torch.ones(1, 1, window_size, window_size, dtype=Df1.dtype)/(window_size**2), stride=window_size)
    D2 = conv2d(Df2**2, torch.ones(1, 1, window_size, window_size, dtype=Df2.dtype)/(window_size**2), stride=window_size)

    # Overall Quality
    Q = torch.sum(ramda1 * D1 + ramda2 * D2) / torch.sum(ramda1 + ramda2)

    return Q.item()


def sd(image):
    return image.std()


def sf(image):
    return np.sqrt(
        np.mean((image[:, 1:] - image[:, :-1]) ** 2)
        + np.mean((image[1:, :] - image[:-1, :]) ** 2)
    )


def ssim(image_F, image_A, image_B):
    return _ssim(image_F, image_A) + _ssim(image_F, image_B)


def vif(image_F, image_A, image_B):
    return getvif(image_F, image_A) + getvif(image_F, image_B)


# def getArray(img):
#     SAx = conv2d(img, h3, padding=1)[0][0]
#     SAy = conv2d(img, h1, padding=1)[0][0]
#     gA = (SAx**2 + SAy**2) ** 0.5
#     aA = torch.atan(SAy / SAx).byte()
#     aA = torch.where(SAx == 0, torch.tensor(math.pi / 2, dtype=torch.uint8), aA).byte()
#     return gA, aA


# # the relative strength and orientation value of GAF,GBF and AAF,ABF;
# def computeQabf(aA, gA, aF, gF):
#     GAF = torch.zeros_like(aA)
#     mask = gA > gF
#     GAF[mask] = (gF[mask] / gA[mask]).byte()
#     mask = gA == gF
#     GAF[mask] = gF[mask].byte()
#     mask = gA < gF
#     GAF[mask] = (gA[mask] / gF[mask]).byte()
#     AAF = 1 - torch.abs(aA - aF) / math.pi * 2
#     QgAF = Tg / (1 + (kg * (GAF - Dg)).exp())
#     QaAF = Ta / (1 + (ka * (AAF - Da)).exp())
#     QAF = QgAF * QaAF

#     return QAF


# def qabf(image_F, image_A, image_B):
#     image_A = torch.from_numpy(image_A).unsqueeze(0).unsqueeze(0).float()
#     image_B = torch.from_numpy(image_B).unsqueeze(0).unsqueeze(0).float()
#     image_F = torch.from_numpy(image_F).unsqueeze(0).unsqueeze(0).float()

#     gA, aA = getArray(image_A)
#     gB, aB = getArray(image_B)
#     gF, aF = getArray(image_F)

#     QAF = computeQabf(aA, gA, aF, gF)
#     QBF = computeQabf(aB, gB, aF, gF)

#     wA = gA**L
#     wB = gB**L
#     output = torch.sum(QAF * wA + QBF * wB) / torch.sum(wA + wB)

#     return output
