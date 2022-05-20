#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#include "ray_payload.glsl"

layout(location = 0) rayPayloadInEXT HitPayload prd;

void main() {
    prd.hit_value = vec3(0.2, 0.5, 0.5);
}