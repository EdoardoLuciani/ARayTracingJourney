# ARayTracingJourney

ARayTracingJourney, like the title mentions is a repo where my main objective is to understand hardware Vulkan ray tracing. We all know that the theoretical answer to rendering is path tracing, but the hardware performance is not quite there. In this repo I am to explore how to incorporate ray-tracing by substituting some of the rasterization tecniques, while trying to keep the performance respectable.

My ultimate goal is to write the successor of my first renderer [TheVulkanTemple](https://github.com/EdoardoLuciani/TheVulkanTemple).

![Screenshot](docs/screenshot1.png)

## Features
- PBR pipeline
- Ray traced primitive exclusion
- Ray traced shadows
- Point, spot, directional and area lights
- XeGTAO ambient occlusion (https://github.com/GameTechDev/XeGTAO)
- High dynamic range with final tonemapping
- GLTF models as inputs


