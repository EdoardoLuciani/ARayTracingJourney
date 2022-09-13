#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "ray_payload.glsl"

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT HitPayload prd;

void main() {
    prd.primitive_info_idx = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    prd.primitive_id = gl_PrimitiveID;
    prd.instance_id = gl_InstanceID;
    prd.hit_attribs = attribs;
}