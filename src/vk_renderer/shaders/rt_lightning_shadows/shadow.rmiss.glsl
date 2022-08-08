#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#include "ray_payload.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload prd;

void main() {
    prd.is_shadowed = false;
}