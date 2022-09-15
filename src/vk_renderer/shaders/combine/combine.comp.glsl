#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 0, r11f_g11f_b10f) uniform image2D input_output_color;
layout (set = 0, binding = 1, r32ui) uniform readonly uimage2D global_ao;

layout (local_size_x = 8, local_size_y = 8) in;


void main() {
    ivec2 global_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 color = imageLoad(input_output_color, global_coords).rgb;
    float ao = float(imageLoad(global_ao, global_coords).r) / 255.0;
    color *= ao;

    imageStore(input_output_color, global_coords, vec4(color, 1.0f));
}