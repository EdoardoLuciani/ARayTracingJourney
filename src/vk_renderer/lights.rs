use nalgebra::*;

#[derive(Default)]
pub struct Lights {
    point_lights: Vec<PointLight>,
    spot_lights: Vec<SpotLight>,
    directional_lights: Vec<DirectionalLight>,
    area_lights: Vec<AreaLight>,
}

impl Lights {
    pub fn get_point_lights(&self) -> &[PointLight] {
        &self.point_lights
    }
    pub fn get_spot_lights(&self) -> &[SpotLight] {
        &self.spot_lights
    }
    pub fn get_directional_lights(&self) -> &[DirectionalLight] {
        &self.directional_lights
    }
    pub fn get_area_lights(&self) -> &[AreaLight] {
        &self.area_lights
    }
    pub fn copy_lights_shader_data(&self, lights_shader_data: &mut [LightShaderData]) {
        assert_eq!(lights_shader_data.len(), self.get_lights_count());

        let mut destination_idx = 0;

        self.point_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
        destination_idx += self.point_lights.len();

        self.spot_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
        destination_idx += self.spot_lights.len();

        self.directional_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
        destination_idx += self.directional_lights.len();

        self.area_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
    }
    pub fn get_lights_count(&self) -> usize {
        self.point_lights.len()
            + self.spot_lights.len()
            + self.directional_lights.len()
            + self.area_lights.len()
    }

    pub fn get_point_lights_mut(&mut self) -> &mut Vec<PointLight> {
        &mut self.point_lights
    }
    pub fn get_spot_lights_mut(&mut self) -> &mut Vec<SpotLight> {
        &mut self.spot_lights
    }
    pub fn get_directional_lights_mut(&mut self) -> &mut Vec<DirectionalLight> {
        &mut self.directional_lights
    }
    pub fn get_area_lights_mut(&mut self) -> &mut Vec<AreaLight> {
        &mut self.area_lights
    }
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct LightShaderData {
    pos: Vector3<f32>,
    light_type: u32,
    dir: Vector3<f32>,
    area_light_direction: u32,
    color: Vector3<f32>,
    falloff_distance: f32,
    penumbra_umbra_angles: Vector2<f32>,
    is_shadowed: u32,
    dummy: f32,
}

trait LightShaderSerializable {
    fn get_light_shader_data(&self) -> LightShaderData;
}

enum LightType {
    POINT = 0,
    SPOT = 1,
    DIRECTIONAL = 2,
    AREA = 3,
}

pub struct PointLight {
    pos: Vector3<f32>,
    color: Vector3<f32>,
    falloff_distance: f32,
    is_shadowed: bool,
}

impl PointLight {
    pub fn new(
        pos: Vector3<f32>,
        color: Vector3<f32>,
        falloff_distance: f32,
        is_shadowed: bool,
    ) -> Self {
        Self {
            pos,
            color,
            falloff_distance,
            is_shadowed,
        }
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn falloff_distance(&self) -> f32 {
        self.falloff_distance
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_falloff_distance(&mut self, falloff_distance: f32) {
        self.falloff_distance = falloff_distance;
    }
}

impl LightShaderSerializable for PointLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: self.pos,
            dir: Vector3::zeros(),
            color: self.color,
            falloff_distance: self.falloff_distance,
            penumbra_umbra_angles: Vector2::zeros(),
            light_type: LightType::POINT as u32,
            area_light_direction: 0,
            is_shadowed: self.is_shadowed as u32,
            dummy: 0.0f32,
        }
    }
}

pub struct SpotLight {
    pos: Vector3<f32>,
    dir: Vector3<f32>,
    color: Vector3<f32>,
    falloff_distance: f32,
    penumbra_umbra_angles: Vector2<f32>,
    is_shadowed: bool,
}

impl SpotLight {
    pub fn new(
        pos: Vector3<f32>,
        dir: Vector3<f32>,
        color: Vector3<f32>,
        falloff_distance: f32,
        penumbra_umbra_angles: Vector2<f32>,
        is_shadowed: bool,
    ) -> Self {
        Self {
            pos,
            dir,
            color,
            falloff_distance,
            penumbra_umbra_angles,
            is_shadowed,
        }
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }
    pub fn dir(&self) -> Vector3<f32> {
        self.dir
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn falloff_distance(&self) -> f32 {
        self.falloff_distance
    }
    pub fn penumbra_umbra_angles(&self) -> Vector2<f32> {
        self.penumbra_umbra_angles
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }
    pub fn set_dir(&mut self, dir: Vector3<f32>) {
        self.dir = dir;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_falloff_distance(&mut self, falloff_distance: f32) {
        self.falloff_distance = falloff_distance;
    }
    pub fn set_penumbra_umbra_angles(&mut self, penumbra_umbra_angles: Vector2<f32>) {
        self.penumbra_umbra_angles = penumbra_umbra_angles;
    }
}

impl LightShaderSerializable for SpotLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: self.pos,
            dir: self.dir,
            color: self.color,
            falloff_distance: self.falloff_distance,
            penumbra_umbra_angles: self.penumbra_umbra_angles,
            light_type: LightType::SPOT as u32,
            area_light_direction: 0,
            is_shadowed: self.is_shadowed as u32,
            dummy: 0.0f32,
        }
    }
}

pub struct DirectionalLight {
    dir: Vector3<f32>,
    color: Vector3<f32>,
    is_shadowed: bool,
}

impl DirectionalLight {
    pub fn new(dir: Vector3<f32>, color: Vector3<f32>, is_shadowed: bool) -> Self {
        Self {
            dir,
            color,
            is_shadowed,
        }
    }

    pub fn dir(&self) -> Vector3<f32> {
        self.dir
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }

    pub fn set_dir(&mut self, dir: Vector3<f32>) {
        self.dir = dir;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
}

impl LightShaderSerializable for DirectionalLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: Vector3::zeros(),
            dir: self.dir,
            color: self.color,
            falloff_distance: 0.0f32,
            penumbra_umbra_angles: Vector2::zeros(),
            light_type: LightType::DIRECTIONAL as u32,
            area_light_direction: 0,
            is_shadowed: self.is_shadowed as u32,
            dummy: 0.0f32,
        }
    }
}

pub struct AreaLight {
    pos: Vector3<f32>,
    pos2: Vector3<f32>,
    direction: bool,
    color: Vector3<f32>,
    falloff_distance: f32,
    penumbra_umbra_angles: Vector2<f32>,
    is_shadowed: bool,
}

impl AreaLight {
    pub fn new(
        pos: Vector3<f32>,
        pos2: Vector3<f32>,
        direction: bool,
        color: Vector3<f32>,
        falloff_distance: f32,
        penumbra_umbra_angles: Vector2<f32>,
        is_shadowed: bool,
    ) -> Self {
        Self {
            pos,
            pos2,
            direction,
            color,
            falloff_distance,
            penumbra_umbra_angles,
            is_shadowed,
        }
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }
    pub fn pos2(&self) -> Vector3<f32> {
        self.pos2
    }
    pub fn direction(&self) -> bool {
        self.direction
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn falloff_distance(&self) -> f32 {
        self.falloff_distance
    }
    pub fn penumbra_umbra_angles(&self) -> Vector2<f32> {
        self.penumbra_umbra_angles
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }
    pub fn set_pos2(&mut self, pos2: Vector3<f32>) {
        self.pos2 = pos2;
    }
    pub fn set_direction(&mut self, direction: bool) {
        self.direction = direction;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_falloff_distance(&mut self, falloff_distance: f32) {
        self.falloff_distance = falloff_distance;
    }
    pub fn set_penumbra_umbra_angles(&mut self, penumbra_umbra_angles: Vector2<f32>) {
        self.penumbra_umbra_angles = penumbra_umbra_angles;
    }
}

impl LightShaderSerializable for AreaLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: self.pos,
            dir: self.pos2,
            color: self.color,
            falloff_distance: self.falloff_distance,
            penumbra_umbra_angles: self.penumbra_umbra_angles,
            light_type: LightType::AREA as u32,
            area_light_direction: self.direction as u32,
            is_shadowed: self.is_shadowed as u32,
            dummy: 0.0f32,
        }
    }
}
