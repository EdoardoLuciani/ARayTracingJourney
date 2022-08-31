#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#include "ray_payload.glsl"

layout(location = 0) rayPayloadInEXT HitPayload prd;

void main() {
    prd.instance_id = -1;
}