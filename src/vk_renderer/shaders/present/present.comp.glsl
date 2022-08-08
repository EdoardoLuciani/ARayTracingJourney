#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "../color_spaces.glsl"
#include "../tonemaps.glsl"

layout (set = 0, binding = 0, r11f_g11f_b10f) uniform readonly image2D input_color;
layout (set = 0, binding = 1, r32ui) uniform readonly uimage2D global_ao;

layout (set = 0, binding = 2) uniform writeonly image2D out_color[];

layout(push_constant) uniform PushConstants {
    uint dst_index;
};

layout (local_size_x = 8, local_size_y = 8) in;
void main() {
    ivec2 global_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 color = imageLoad(input_color, global_coords).rgb;
    float ao = float(imageLoad(global_ao, global_coords).r) / 255.0;
    color *= ao;

    vec3 xyY = rgb_to_xyY(color);
    xyY.z = Tonemap_Uchimura(xyY.z);
    color = rgb_to_srgb_approx(xyY_to_rgb(xyY));

    imageStore(out_color[nonuniformEXT(dst_index)], global_coords, vec4(color, 1.0f));
}