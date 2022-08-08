#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 0, r11f_g11f_b10f) uniform readonly image2D input_color;
layout (set = 0, binding = 1, r32ui) uniform readonly uimage2D global_ao;

layout (set = 0, binding = 2) uniform restrict writeonly image2D out_color[];

layout(push_constant) uniform PushConstants {
    uint dst_index;
};

layout (local_size_x = 8, local_size_y = 8) in;
void main() {
    vec3 color = imageLoad(input_color, ivec2(gl_GlobalInvocationID.xy)).rgb;
    float ao = float(imageLoad(global_ao, ivec2(gl_GlobalInvocationID.xy)).r) / 255.0;

    color *= ao;
    imageStore(out_color[nonuniformEXT(dst_index)], ivec2(gl_GlobalInvocationID.xy), vec4(ao));
}