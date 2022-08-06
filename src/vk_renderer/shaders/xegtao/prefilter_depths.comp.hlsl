#define XE_GTAO_USE_DEFAULT_CONSTANTS 1
#define XE_GTAO_USE_HALF_FLOAT_PRECISION 0
#define XE_GTAO_VIEWSPACE_DEPTH

#include "XeGTAO.h"
#include "XeGTAO.hlsli"

// input output textures for the first pass (XeGTAO_PrefilterDepths16x16)
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float>            g_srcRawDepth;   // source depth buffer data (in NDC space in DirectX)
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState g_samplerPointClamp;

[[vk::binding(1, 0)]]
RWTexture2D<lpfloat>        g_outWorkingDepthMIP[5];   // output viewspace depth MIP (these are views into g_srcWorkingDepth MIP levels)

[[vk::binding(0, 1)]]
cbuffer GTAOConstantBuffer {
        GTAOConstants g_GTAOConsts;
}

// Engine-specific entry point for the first pass
[numthreads(8, 8, 1)]   // <- hard coded to 8x8; each thread computes 2x2 blocks so processing 16x16 block: Dispatch needs to be called with (width + 16-1) / 16, (height + 16-1) / 16
void main( uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupThreadID : SV_GroupThreadID )
{
    XeGTAO_PrefilterDepths16x16( dispatchThreadID, groupThreadID, g_GTAOConsts, g_srcRawDepth, g_samplerPointClamp, g_outWorkingDepthMIP[0], g_outWorkingDepthMIP[1], g_outWorkingDepthMIP[2], g_outWorkingDepthMIP[3], g_outWorkingDepthMIP[4]);
}