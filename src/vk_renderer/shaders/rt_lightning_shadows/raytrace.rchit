#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf: require

#include "ray_payload.glsl"
#include "../brdfs.glsl"
#include "../color_spaces.glsl"
#include "../tonemaps.glsl"

struct PrimitiveInfo {
    uint64_t vertices_address;
    uint64_t indices_address;
    uint texture_offset;
    uint single_index_size;
};
layout(binding = 1, set = 1, scalar) buffer PrimitiveInfos { PrimitiveInfo i[]; } primitive_infos;
layout(binding = 2, set = 1) uniform sampler2DArray textures[];

layout(binding = 0, set = 2) uniform camera {
    mat4 view;
    mat4 view_inv;
    mat4 proj;
    mat4 proj_inv;
    vec3 camera_pos;
};

struct Vertex {
    vec3 pos;
    vec2 tex_coord;
    vec3 normal;
    vec4 tangent;
};
struct VertexData {
    float data[12];
};
layout(buffer_reference, scalar) buffer VerticesData {VertexData v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices16 {u16vec3 i[]; };
layout(buffer_reference, scalar) buffer Indices32 {uvec3 i[]; };

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT HitPayload prd;


uvec3 get_indices(PrimitiveInfo primitive_info, uint primitive_id) {
    uvec3 indices;
    if (primitive_info.single_index_size == 2) {
        Indices16 indices16 = Indices16(primitive_info.indices_address);
        indices = indices16.i[primitive_id];
    }
    else {
       Indices32 indices32 = Indices32(primitive_info.indices_address);
       indices = indices32.i[primitive_id];
    }
    return indices;
}

Vertex vertex_data_to_vertex(VertexData vertex_data) {
    return Vertex(
        vec3(vertex_data.data[0], vertex_data.data[1], vertex_data.data[2]),
        vec2(vertex_data.data[3], vertex_data.data[4]),
        vec3(vertex_data.data[5], vertex_data.data[6], vertex_data.data[7]),
        vec4(vertex_data.data[8], vertex_data.data[9], vertex_data.data[10], vertex_data.data[11])
    );
}

void main() {
    PrimitiveInfo primitive_info = primitive_infos.i[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT];

    uvec3 indices = get_indices(primitive_info, gl_PrimitiveID);

    VerticesData vertices_data = VerticesData(primitive_info.vertices_address);
    Vertex v0 = vertex_data_to_vertex(vertices_data.v[indices.x]);
    Vertex v1 = vertex_data_to_vertex(vertices_data.v[indices.y]);
    Vertex v2 = vertex_data_to_vertex(vertices_data.v[indices.z]);

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
    vec3 world_pos = vec3(gl_ObjectToWorldEXT * vec4(pos,1.0f));

    vec2 tex_coord = v0.tex_coord * barycentrics.x + v1.tex_coord * barycentrics.y + v2.tex_coord * barycentrics.z;

    vec3 normal      = normalize(v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z);
    vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));

    vec3 tangent = normalize(v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z);
    vec3 world_tangent = normalize(vec3(mat4(gl_ObjectToWorldEXT) * vec4(tangent.xyz, 0)));
    world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
    vec3 world_binormal = cross(world_normal, world_tangent) * v0.tangent.w;

    mat3 tbn = mat3(world_tangent, world_binormal, world_normal);
    vec3 N = normalize(texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 2)).xyz * 2.0 - 1.0);
    N = normalize(tbn * N);

    vec3 albedo = pow(texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 0)).rgb, vec3(2.2));
    float roughness = texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 1)).g;
    float metallic = texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 1)).b;

    vec3 V = normalize(camera_pos - world_pos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float corrected_roughness = roughness * roughness;

    float nc_NdotV = dot(N,V);
    float NdotV = clamp(nc_NdotV, 1e-5, 1.0);
    vec3 rho = vec3(0.0);

    vec3 light_pos = vec3(0.0f, 0.66f, 0.0f);
    for(int i=0; i<1; i++) {
        vec3 L = normalize(light_pos - world_pos);
        vec3 H = normalize(V + L);

        float nc_NdotL = dot(N, L);
        float NdotL = clamp(nc_NdotL, 0.0, 1.0);
        float NdotH = clamp(dot(N, H), 0.0, 1.0);
        float LdotV = clamp(dot(L, V), 0.0, 1.0);
        float LdotH = clamp(dot(L, H), 0.0, 1.0);

        vec3 Ks = F_Schlick(F0, LdotH);
        vec3 Kd = (1.0 - metallic)*albedo;

        vec3 rho_s = CookTorrance_specular(NdotL, NdotV, NdotH, corrected_roughness, Ks);
        vec3 rho_d = Kd * Burley_diffuse_local_sss(corrected_roughness, NdotV, nc_NdotV, nc_NdotL, LdotH, 0.4);

        vec3 radiance = vec3(3.0f);
        rho += (rho_s + rho_d) * radiance * NdotL;
    }

    vec3 color = rho;

    vec3 xyY = rgb_to_xyY(color);
    xyY.z = Tonemap_Uchimura(xyY.z);
    color = rgb_to_srgb_approx(xyY_to_rgb(xyY));

    prd.hit_value = color;
}