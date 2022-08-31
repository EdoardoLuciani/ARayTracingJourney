struct Camera {
    mat4 view;
    mat4 view_inv;
    mat4 prev_view;
    mat4 proj;
    mat4 proj_inv;
    mat4 prev_proj;
    mat4 vp;
    mat4 prev_vp;
    vec3 camera_pos;
    float pad;
    vec2 jitter;
    vec2 prev_jitter;
};