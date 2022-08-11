struct HitPayload {
    int primitive_info_idx;
    int primitive_id;
    vec2 hit_attribs;
    mat4x3 object_to_world;
    mat4x3 world_to_object;
};

struct ShadowPayload {
    bool is_lit;
};