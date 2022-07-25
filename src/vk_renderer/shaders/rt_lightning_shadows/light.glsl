struct Light {
    vec3 pos;
    uint type;
    vec3 dir;
    uint area_light_direction;
    vec3 color;
    float falloff_distance;
    vec2 penumbra_umbra_angles;
    uint shadowed;
    float dummy;
};

bool is_point(Light light) {
    return light.type == 0;
}

bool is_spot(Light light) {
    return light.type == 1;
}

bool is_directional(Light light) {
    return light.type == 2;
}

bool is_area(Light light) {
    return light.type == 3;
}

bool is_shadowed(Light light) {
    return bool(light.shadowed);
}

vec3 get_light_radiance(Light light, vec3 pos, vec3 L_vec) {
    vec3 radiance = light.color;
    if (is_spot(light)) {
        float theta_s = acos(dot(light.dir, -L_vec));
        float t = clamp((theta_s - light.penumbra_umbra_angles.y) / (light.penumbra_umbra_angles.x - light.penumbra_umbra_angles.y), 0.0, 1.0);
        radiance *= pow(t, 2.0);
    }

    if (light.falloff_distance > 0.0) {
        float dist = length(light.pos - pos);
        radiance *= pow(max(1-pow(dist/light.falloff_distance, 2.0f), 0.0f), 2.0f);
    }

    return radiance;
}

vec3 get_L_vec(Light light, vec3 pos) {
    if(is_point(light) || is_spot(light)) {
        return normalize(light.pos - pos);
    }
    else if (is_directional(light)) {
        return normalize(-light.dir);
    }
    return vec3(1.0);
}

