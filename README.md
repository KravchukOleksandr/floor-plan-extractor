# Floor Plan Extractor

Extract room polygons and relative pixel areas from a single RGB image of a 3D floor plan.

<p align="center">
  <img src="examples/test_2/annotated.png" width="49%" />
  <img src="examples/test_4/annotated.png" width="49%" />
</p>

Additional inputs, intermediate masks, annotated images, and JSON outputs are available in [`examples/`](examples/).

## Approach

Let $I$ denote the normalized RGB input, and let $D$ denote the normalized depth map estimated from the corresponding RGB image using Depth Anything V2. The trained custom wall-projection network performs the following forward pass:

$$
\begin{array}{rcl}
X & = & \mathrm{concat}(I,D) \\
(F_1,F_2,F_3,F_4)
  & = & \mathrm{ConvNeXtS}_{\mathrm{backbone}}(X) \\
M_{\mathrm{top}}
  & = & \mathrm{sigmoid}\left(
     \mathrm{UPerNet}_{\mathrm{head}}(F_1,F_2,F_3,F_4)
     \right) \\
H & = & \mathrm{HomographyHead}(F_4) \\
M_{\mathrm{floor}} & = & \mathrm{warp}(M_{\mathrm{top}},H)
\end{array}
$$

Here, $H$ is the predicted homography matrix, and $\mathrm{warp}$ applies its inverse-mapped perspective transformation to the top-view wall probability map using bilinear interpolation.

The floor-projected wall probability is converted into room polygons with a deterministic post-processing pipeline:

1. **Wall reconstruction** — clear the image frame, apply hysteresis thresholding by binary propagation, then seal local cracks with elliptical morphological closing and dilation.
2. **Gap completion** — skeletonize the wall mask, estimate endpoint tangents, and use cosine alignment to connect nearby endpoints only when the bridge creates another enclosed region; set its thickness from the Euclidean distance transform.
3. **Exterior removal** — perform border-seeded binary propagation (morphological reconstruction) through free space and discard everything reachable from outside.
4. **Room detection** — label the remaining eight-connected components and reject regions with insufficient area or Euclidean distance-transform radius.
5. **Polygon output** — trace external contours, simplify them with the Ramer–Douglas–Peucker algorithm, restore the original image coordinates, and compute relative pixel areas.

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
