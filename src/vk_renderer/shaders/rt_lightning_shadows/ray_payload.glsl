struct HitPayload {
    int primitive_info_idx;
    int primitive_id;
    int instance_id;
    vec2 hit_attribs;
};

struct ShadowPayload {
    bool is_lit;
};