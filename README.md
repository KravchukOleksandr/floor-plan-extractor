# Floor Plan Extractor

Extract room polygons and relative pixel areas from a single RGB image of a 3D floor plan.

<p align="center">
  <img src="examples/test_2/annotated.png" width="49%" />
  <img src="examples/test_4/annotated.png" width="49%" />
</p>

Additional inputs, intermediate masks, annotated images, and JSON outputs are available in [`examples/`](examples/).

## Approach

The RGB image is resized with aspect-ratio-preserving padding and normalized using ImageNet statistics. Depth Anything V2 provides a relative depth map, which is robustly normalized between its 1st and 99th percentiles. The resulting RGBD tensor is processed by a ConvNeXt-S backbone with a UPerNet segmentation head and a homography regression head.

$$
\begin{aligned}
D_n &= 2\,\mathrm{clip}\left(\frac{D-P_1(D)}{\max(P_{99}(D)-P_1(D),\varepsilon)},0,1\right)-1 \\
X &= \mathrm{concat}\left(\frac{I-\mu}{\sigma},D_n\right) \\
(F_1,F_2,F_3,F_4) &= \mathrm{ConvNeXtS}(X) \\
M_{top} &= \mathrm{UPerNet}(F_1,F_2,F_3,F_4) \\
\Delta h &= \mathrm{MLP}\left(\mathrm{Pool}(\mathrm{Conv}(F_4))\right) \in \mathbb{R}^8 \\
H &= \begin{bmatrix}
1+\Delta h_1 & \Delta h_2 & \Delta h_3 \\
\Delta h_4 & 1+\Delta h_5 & \Delta h_6 \\
\Delta h_7 & \Delta h_8 & 1
\end{bmatrix} \\
M_{floor} &= \mathrm{warp}(M_{top},H)
\end{aligned}
$$

The floor-projected wall probability is converted into room polygons with a deterministic post-processing pipeline:

1. **Wall reconstruction** — clear the image frame, apply hysteresis thresholding, then seal local cracks with elliptical closing and slight dilation.
2. **Gap completion** — connect aligned skeleton endpoints only when the bridge creates another enclosed region.
3. **Exterior removal** — flood-fill free space from the image borders and discard everything reachable from outside.
4. **Room detection** — label the remaining eight-connected components and reject regions that are too small or narrow.
5. **Polygon output** — trace and simplify contours with Douglas–Peucker, restore the original image coordinates, and compute relative pixel areas.

## Assumptions

These are **<u>TEMPORARY</u>** limitations of the current prototype: wall surfaces are expected to have a color close to white, and the visible wall-cut plane is assumed to be parallel to the floor plane. Inputs that strongly violate either assumption may produce incomplete wall masks or geometrically inconsistent room polygons.

## Quickstart

Git LFS is required for the model checkpoint.

```bash
git lfs pull
docker compose up --build -d
```

Wait until the service is ready:

```bash
curl http://localhost:8000/health
```

Extract rooms from an image:

```bash
curl -X POST "http://localhost:8000/extract?preset=weak" \
  -F "image=@examples/test_2/input.webp" \
  -o result.zip

mkdir -p outputs/test_2
unzip -o result.zip -d outputs/test_2
```

The archive contains:

```text
annotated.png
probability.png
wall_barrier.png
rooms.json
```

The conservative post-processing preset is available with `preset=conservative`.
