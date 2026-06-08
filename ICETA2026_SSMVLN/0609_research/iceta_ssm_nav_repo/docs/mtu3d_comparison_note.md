# MTU3D Comparison Note

## Comparison Paper

- Paper: Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation
- Year / venue: 2025, ICCV 2025 Highlight / arXiv preprint
- Paper link: https://arxiv.org/abs/2507.04047
- Code: https://github.com/MTU3D/MTU3D
- Checkpoints: https://huggingface.co/bigai/MTU3D

## Why This Is A Better Comparison Than Pure Semantic Mapping

MTU3D is closer to this project because it studies 3D vision-language grounding plus active exploration. It also represents unexplored locations as frontier queries and learns where to explore next. This gives a direct comparison axis for language-guided unknown-space exploration, while classical semantic mapping papers mainly compare SLAM, frontier exploration, and object-map updating.

## Method Comparison

| Item | MTU3D | This Project |
| --- | --- | --- |
| Main task | 3D vision-language grounding and active embodied navigation | 2D/3D unknown-space exploration with VLN grounding, online frontier graph, and target/anomaly discovery |
| Input | RGB-D plus category, language description, or reference image | RGB-D/depth plus natural-language command grounded into target/action spec |
| Representation | Online query-based 3D spatial memory with object/frontier queries | Explicit online 2D grid and 3D voxel-frontier graph with nodes and edges |
| Frontier use | Unexplored locations are modeled as frontier queries inside the learned model | Observed free voxels/cells adjacent to unknown space become explicit frontier candidates |
| Model | Large 3D vision-language-exploration model | Local VLN/object-goal grounding plus lightweight SSM-style frontier scorer and graph planner |
| Training | Vision-language-exploration pretraining over large simulated and real RGB-D trajectories, then navigation fine-tuning | One-time SSM frontier scorer training on synthetic frontier candidates; unseen maps use inference only |
| Planning | Learned model selects/explores goals with 3D-VL memory | Highest-scoring frontier node is selected, connected through the observed graph, then BFS/path reconstruction is used |
| 2D/3D scope | Primarily 3D embodied navigation | 2D grid validation plus 3D voxel partial-observation exploration |
| Efficiency claim | Strong model, but onboard lightweight deployment is not the core validated claim | SSM plus explicit graph is designed for lower-cost onboard execution, with Jetson Orin as target after simulation validation |

## Current Project Evidence

- Frontier training rows: 168492
- Frontier validation rows: 44211
- Best validation top-1 frontier choice accuracy: 0.879

### Latest Local Paper-Proxy Aggregate

The table below is **not** an official MTU3D reproduction. It compares this project's proposed `ssm_utility` mode with an `mtu3d_proxy` mode in the local semantic-prior voxel simulator. The proxy approximates MTU3D's active frontier-query behavior using the signals available in this simulator, so it is suitable for ICETA prototype discussion but not for claiming official MTU3D benchmark superiority.

| mode | maps | success | avg successful target step | revisit | fallback | decision ms | frontiers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mtu3d_proxy | 12 | 0.917 | 38.27 | 0.1763 | 0.00 | 54.57 | 80.07 |
| ssm_utility | 12 | 1.000 | 22.08 | 0.0139 | 0.00 | 42.39 | 79.94 |
| hybrid_ssm | 12 | 1.000 | 33.67 | 0.0770 | 0.00 | 97.19 | 80.01 |

## Safe Claim For The Paper

MTU3D learns a powerful 3D vision-language-exploration model that unifies visual grounding and active exploration. Our work instead keeps the spatial memory explicit as a 2D/3D frontier graph and uses a one-time-trained target-conditioned SSM-style scorer for frontier selection. In the local semantic-prior voxel study, the proposed scorer improved target discovery success, successful target step, revisit ratio, and prototype decision time against the MTU3D-style proxy. `hybrid_ssm` is reported only as an ablation; the proposed `ssm_utility` result does not call the MTU3D-style proxy scorer during frontier selection. The key claim should still be local simulator superiority against the proxy, not that this prototype outperforms official MTU3D benchmarks.

## What Is Still Needed For Direct Quantitative Comparison

Direct numerical comparison with MTU3D requires one of the following:

1. Run the official MTU3D code/checkpoints and this project on the same official benchmark and report the same metrics.
2. Implement an MTU3D-compatible adapter inside this voxel simulator, including RGB-D input formatting, query memory, and frontier selection.
3. Use MTU3D's reported benchmark numbers only as literature context, while clearly stating that the datasets and protocols differ.

For ICETA, use MTU3D as the comparison paper for method-level positioning. The dashboard can show `MTU3D-style Proxy` vs `Proposed SSM`, but the caption must state that this is a local simulator proxy, not official MTU3D reproduction.

## Next Experiment To Strengthen The Comparison

1. Re-run this timing report after each SSM/frontier scoring change, using the same unseen maps.
2. Add an MTU3D reproduction/adaptation plan if a direct quantitative comparison is required.
3. Add Jetson Orin timing once the Gazebo/RGB-D pipeline is stable.
4. Validate and further optimize SSM decision latency on the intended onboard stack through batching, TorchScript/ONNX, or C++ runtime integration.
5. Keep ETPNav as the VLN/topological reference and use MTU3D as the 3D vision-language exploration comparison paper.
