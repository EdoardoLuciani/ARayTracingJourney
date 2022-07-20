#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_nonuniform_qualifier : require

#extension GL_EXT_debug_printf: require


#include "ray_payload.glsl"

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT HitPayload prd;

struct PrimitiveInfo {
    uint64_t vertices_address;
    uint64_t indices_address;
    uint texture_offset;
    uint single_index_size;
};
layout(binding = 1, set = 1, scalar) buffer PrimitiveInfos { PrimitiveInfo i[]; } primitive_infos;
layout(binding = 2, set = 1) uniform sampler2DArray textures[];

struct Vertex {
    vec3 pos;
    vec2 tex_coord;
    vec3 normal;
    vec4 tangent;
};
layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices16 {u16vec3 i[]; };
layout(buffer_reference, scalar) buffer Indices32 {uvec3 i[]; };

void main() {
    PrimitiveInfo primitive_info = primitive_infos.i[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT];

    Vertices vertices = Vertices(primitive_info.vertices_address);

    uvec3 indices;
    if (primitive_info.single_index_size == 2) {
        Indices16 indices16 = Indices16(primitive_info.indices_address);
        indices = indices16.i[gl_PrimitiveID];
    }
    else {
       Indices32 indices32 = Indices32(primitive_info.indices_address);
       indices = indices32.i[gl_PrimitiveID];
    }

    Vertex v0 = vertices.v[indices.x];
    Vertex v1 = vertices.v[indices.y];
    Vertex v2 = vertices.v[indices.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    const vec3 pos      = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
    const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

    // Computing the normal at hit position
    const vec3 normal      = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    const vec2 tex_coord = v0.tex_coord * barycentrics.x + v1.tex_coord * barycentrics.y + v2.tex_coord * barycentrics.z;
    vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));  // Transforming the normal to world space

    vec3 albedo = pow(texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 0)).rgb, vec3(2.2));

    prd.hit_value = albedo;
}