use bitflags::bitflags;
use nalgebra::*;
use std::path::Path;

bitflags! {
    pub struct MeshAttributeType: u8 {
        const VERTICES = 1;
        const TEX_COORDS = 2;
        const NORMALS = 4;
        const TANGENTS = 8;
        const INDICES = 16;
    }

    pub struct TextureType: u8 {
        const ALBEDO = 1;
        const ORM = 2;
        const NORMAL = 4;
        const EMISSIVE = 8;
    }
}

macro_rules! bitflag_vec {
    ($bitflag_type:ty, $bitflag:expr) => {{
        (0..<$bitflag_type>::all().bits().count_ones())
            .filter_map(|i| {
                let flag = <$bitflag_type>::from_bits(1 << i).unwrap();
                if $bitflag.contains(flag) {
                    return Some(flag);
                }
                None
            })
            .collect::<Vec<$bitflag_type>>()
    }};
}
pub(crate) use bitflag_vec;

pub trait ModelReader {
    fn open(
        file_path: &Path,
        normalize_vectors: bool,
        coerce_image_to_format: Option<ash::vk::Format>,
    ) -> Self;
    fn copy_model_data_to_ptr(
        &self,
        mesh_attributes_types_to_copy: MeshAttributeType,
        textures_to_copy: TextureType,
        dst_slice: Option<&mut [u8]>,
    ) -> ModelCopyInfo;
    fn get_primitives_bounding_sphere(&self) -> Sphere;
}

pub struct ModelCopyInfo {
    primitives_copy_data: Vec<PrimitiveCopyInfo>,
}

#[derive(Default)]
pub struct PrimitiveCopyInfo {
    pub mesh_buffer_offset: u64,
    pub mesh_size: u64,
    pub single_mesh_element_size: u32,

    pub indices_buffer_offset: u64,
    pub indices_size: u64,
    pub single_index_size: u32,

    pub image_buffer_offset: u64,
    pub image_size: u64,
    pub image_format: ash::vk::Format,
    pub image_extent: ash::vk::Extent3D,
    pub image_mip_levels: u32,
    pub image_layers: u32,
}

impl ModelCopyInfo {
    pub fn new(primitives_copy_data: Vec<PrimitiveCopyInfo>) -> Self {
        ModelCopyInfo {
            primitives_copy_data,
        }
    }

    pub fn get_primitive_data(&self) -> &[PrimitiveCopyInfo] {
        self.primitives_copy_data.as_slice()
    }

    pub fn compute_total_size(&self) -> usize {
        let mut size: u64 = 0;
        for primitive_copy_data in &self.primitives_copy_data {
            size += primitive_copy_data.mesh_size;
            size += primitive_copy_data.indices_size;
            size += primitive_copy_data.image_size
        }
        size as usize
    }

    pub fn compute_aligned_mesh_and_indices_size(&self) -> usize {
        let mut size = 0;
        for primitive_copy_data in &self.primitives_copy_data {
            size = align_offset(size, 12);
            size += primitive_copy_data.mesh_size;
            size += primitive_copy_data.indices_size;
        }
        size as usize
    }
}

pub struct Sphere {
    center: Vector3<f32>,
    radius: f32,
}

impl Sphere {
    pub fn new(center: Vector3<f32>, radius: f32) -> Self {
        Sphere { center, radius }
    }

    pub fn get_center(&self) -> Vector3<f32> {
        self.center
    }

    pub fn get_radius(&self) -> f32 {
        self.radius
    }

    pub fn get_distance_from_point(&self, point: Vector3<f32>) -> f32 {
        (self.center - point).magnitude() - self.radius
    }

    pub fn transform(&self, m_transform: Matrix3x4<f32>) -> Sphere {
        let center = Vector4::<f32>::new(self.center.x, self.center.y, self.center.z, 1.0f32);

        let vec_scale2 = Vector3::new(
            m_transform.column(0).magnitude(),
            m_transform.column(1).magnitude(),
            m_transform.column(2).magnitude(),
        );
        let max_scale = vec_scale2.max();
        Sphere {
            center: (m_transform * center),
            radius: max_scale * self.radius,
        }
    }
}

pub fn align_offset(offset: u64, alignment: u64) -> u64 {
    alignment * ((offset as f32 / alignment as f32).ceil() as u64)
}

#[cfg(test)]
mod tests {
    use super::MeshAttributeType;

    #[test]
    fn bitflag_vec_partial_filled() {
        let f = MeshAttributeType::INDICES | MeshAttributeType::NORMALS;
        assert_eq!(
            bitflag_vec!(MeshAttributeType, f),
            vec![MeshAttributeType::NORMALS, MeshAttributeType::INDICES]
        )
    }

    #[test]
    fn bitflag_vec_partial_full() {
        let f = MeshAttributeType::all();
        assert_eq!(
            bitflag_vec!(MeshAttributeType, f),
            vec![
                MeshAttributeType::VERTICES,
                MeshAttributeType::TEX_COORDS,
                MeshAttributeType::NORMALS,
                MeshAttributeType::TANGENTS,
                MeshAttributeType::INDICES
            ]
        )
    }
}
