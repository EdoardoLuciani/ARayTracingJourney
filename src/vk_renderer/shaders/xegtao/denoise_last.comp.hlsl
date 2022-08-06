#define XE_GTAO_USE_DEFAULT_CONSTANTS 1
#define XE_GTAO_USE_HALF_FLOAT_PRECISION 1

#include "XeGTAO.h"
#include "XeGTAO.hlsli"

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<uint>             g_srcWorkingAOTerm;   // coming from previous pass
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState g_samplerPointClamp0;

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<lpfloat>          g_srcWorkingEdges;   // coming from previous pass
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState g_samplerPointClamp1;

[[vk::binding(2, 0)]]
RWTexture2D<uint>           g_outFinalAOTerm;   // final AO term - just 'visibility' or 'visibility + bent normals'

[[vk::binding(0, 1)]]
cbuffer GTAOConstantBuffer {
        GTAOConstants g_GTAOConsts;
}

// Engine-specific entry point for the third pass
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void main( const uint2 dispatchThreadID : SV_DispatchThreadID ) {
    const uint2 pixCoordBase = dispatchThreadID * uint2( 2, 1 );    // we're computing 2 horizontal pixels at a time (performance optimization)
    XeGTAO_Denoise( pixCoordBase, g_GTAOConsts, g_srcWorkingAOTerm, g_srcWorkingEdges, g_samplerPointClamp0, g_samplerPointClamp1, g_outFinalAOTerm, true );
}