#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "../color_spaces.glsl"

#define A_GLSL 1
#define A_GPU 1
#include "ffx_a.h"

layout (set = 0, binding = 0, r11f_g11f_b10f) uniform readonly image2D input_color;

layout (set = 0, binding = 1) uniform writeonly image2D out_color[];

layout (set = 0, binding = 2) uniform uniform_buffer {
    uvec4 lpm_data[24];
};

layout(push_constant) uniform PushConstants {
    uint dst_index;
};

layout (local_size_x = 8, local_size_y = 8) in;

AU4 LpmFilterCtl(AU1 i){return lpm_data[i];}
#define LPM_NO_SETUP 1
#include "ffx_lpm.h"

void main() {
    ivec2 global_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 color = imageLoad(input_color, global_coords).rgb;
    LpmFilter(color.r,color.g,color.b,false,LPM_CONFIG_709_709);
    color = rgb_to_srgb_approx(color);

    imageStore(out_color[nonuniformEXT(dst_index)], global_coords, vec4(color, 1.0f));
}