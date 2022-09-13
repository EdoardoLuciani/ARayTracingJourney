#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_nonuniform_qualifier : require

#include "ray_payload.glsl"
#include "../brdfs.glsl"
#include "light.glsl"
#include "camera.glsl"
#include "model_uniform.glsl"

layout(binding = 0, set = 0, rgba32f) uniform image2D image;
layout(binding = 1, set = 0, r16f) uniform image2D depth_image;
layout(binding = 2, set = 0, r11f_g11f_b10f) uniform image2D normal_image;
layout(binding = 3, set = 0, rg16f) uniform image2D motion_vector_image;

layout(binding = 0, set = 1) uniform accelerationStructureEXT topLevelAS;
struct PrimitiveInfo {
    uint64_t vertices_address;
    uint64_t indices_address;
    uint texture_offset;
    uint single_index_size;
};
layout(binding = 1, set = 1, scalar) readonly buffer PrimitiveInfos { PrimitiveInfo primitive_infos[]; };
layout(binding = 2, set = 1) uniform sampler2DArray textures[];
layout(binding = 3, set = 1) uniform ModelUniforms { 
    ModelUniform model_uniform;
} model_uniforms[];

layout(binding = 0, set = 2) uniform CameraUniform {
    Camera camera;
};

layout(binding = 0, set = 3) readonly buffer Lights { Light lights[]; };

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

layout(location = 0) rayPayloadEXT HitPayload hit_payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadow_payload;

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
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5) + camera.jitter;
    const vec2 in_uv = pixel_center/vec2(gl_LaunchSizeEXT.xy);
    vec2 d = in_uv * 2.0 - 1.0;

    vec3 origin    = camera.camera_pos;
    vec4 target    = camera.proj_inv * vec4(d.x, d.y, 1, 1);
    vec4 direction = camera.view_inv * vec4(normalize(target.xyz), 0);

    uint  ray_flags = gl_RayFlagsOpaqueEXT;
    float t_min     = 0.001;
    float t_max     = 10000.0;

    traceRayEXT(topLevelAS, // acceleration structure
              ray_flags,       // rayFlags
              0xFF,           // cullMask
              0,              // sbtRecordOffset
              0,              // sbtRecordStride
              0,              // missIndex
              origin,     // ray origin
              t_min,           // ray min range
              direction.xyz,  // ray direction
              t_max,           // ray max range
              0               // payload (location = 0)
    );

    float out_depth = 10000.0;
    vec3 out_color = vec3(0.0);
    vec3 out_normal = vec3(0.5);
    vec2 out_motion_vector = vec2(0.0);
    if (hit_payload.instance_id >= 0) {
        PrimitiveInfo primitive_info = primitive_infos[hit_payload.primitive_info_idx];

        uvec3 indices = get_indices(primitive_info, hit_payload.primitive_id);

        VerticesData vertices_data = VerticesData(primitive_info.vertices_address);
        Vertex v0 = vertex_data_to_vertex(vertices_data.v[indices.x]);
        Vertex v1 = vertex_data_to_vertex(vertices_data.v[indices.y]);
        Vertex v2 = vertex_data_to_vertex(vertices_data.v[indices.z]);

        const vec3 barycentrics = vec3(1.0 - hit_payload.hit_attribs.x - hit_payload.hit_attribs.y, hit_payload.hit_attribs.x, hit_payload.hit_attribs.y);

        vec3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
        vec3 world_pos = vec3(model_uniforms[nonuniformEXT(hit_payload.instance_id)].model_uniform.matrix * vec4(pos,1.0f));

        vec2 tex_coord = v0.tex_coord * barycentrics.x + v1.tex_coord * barycentrics.y + v2.tex_coord * barycentrics.z;

        vec3 normal      = normalize(v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z);
        vec3 world_normal = normalize(mat3(model_uniforms[nonuniformEXT(hit_payload.instance_id)].model_uniform.trans_inv_matrix) * normal);

        vec3 tangent = normalize(v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z);
        vec3 world_tangent = normalize(mat3(model_uniforms[nonuniformEXT(hit_payload.instance_id)].model_uniform.trans_inv_matrix) * tangent);
        world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
        vec3 world_binormal = cross(world_normal, world_tangent) * v0.tangent.w;

        mat3 tbn = mat3(world_tangent, world_binormal, world_normal);
        vec3 N = normalize(texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 2)).xyz * 2.0 - 1.0);
        N = normalize(tbn * N);

        vec3 albedo = pow(texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 0)).rgb, vec3(2.2));
        float roughness = texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 1)).g;
        float metallic = texture(textures[nonuniformEXT(primitive_info.texture_offset)], vec3(tex_coord, 1)).b;

        vec3 V = normalize(camera.camera_pos - world_pos);
        vec3 F0 = mix(vec3(0.04), albedo, metallic);
        float corrected_roughness = roughness * roughness;

        float nc_NdotV = dot(N,V);
        float NdotV = clamp(nc_NdotV, 1e-5, 1.0);

        vec3 rho = vec3(0.0);
        for(int i=0; i<lights.length(); i++) {
            vec3 nn_L = get_unnormalized_L_vec(lights[i], world_pos);
            vec3 L = normalize(nn_L);
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

            if (casts_shadows(lights[i]) && nc_NdotL > 0) {
                shadow_payload.is_lit = false;
                traceRayEXT(topLevelAS,
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                    0xFF,
                    0,
                    0,
                    1,
                    world_pos,
                    0.01,
                    L,
                    length(nn_L),
                    1
                );
            }

            vec3 radiance = get_light_radiance(lights[i], world_pos, L);
            rho += (rho_s + rho_d) * radiance * float(shadow_payload.is_lit) * NdotL;
        }
        // direct lightning
        out_color = rho;

        // indirect lightning
        out_color += 0.01 * albedo;

        out_depth = -(camera.view * vec4(world_pos, 1.0f)).z;

        out_normal = mat3(transpose(camera.view_inv)) * N;
        out_normal.yz = -out_normal.yz;
        out_normal = normalize(out_normal) * 0.5 + 0.5;

        vec4 current_pos = camera.vp * vec4(world_pos, 1.0);
        vec4 prev_pos = camera.prev_vp * model_uniforms[nonuniformEXT(hit_payload.instance_id)].model_uniform.prev_matrix * vec4(pos, 1.0);

        vec2 cancel_jitter = camera.prev_jitter - camera.jitter;
        out_motion_vector = (prev_pos.xy / prev_pos.w) - (current_pos.xy / current_pos.w);
        out_motion_vector *= vec2(0.5, 0.5);
    }

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(out_color, 0.0));
    imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(out_depth));
    imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(out_normal, 0.0));
    imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(out_motion_vector, 0.0, 0.0));
}