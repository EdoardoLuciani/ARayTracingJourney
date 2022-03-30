use bitflags::bitflags;

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

pub fn get_aligned_memory_size(offset: u64, alignment: u64) -> u64 {
    alignment * ((offset as f32 / alignment as f32).ceil() as u64)
}

#[derive(Default)]
pub struct PrimitiveCopyInfo {
    pub mesh_buffer_offset: u64,
    pub mesh_size: u64,
    pub single_mesh_element_size: u32,

    pub indices_buffer_offset: u64,
    pub indices_size: u64,
    pub single_index_size: u32,

    pub textures_buffer_offset: u64,
    pub textures_extent: (u32, u32),
    pub textures_format: ash::vk::Format,
    pub textures_size: u64,
}

pub struct ModelCopyInfo {
    primitives_copy_data: Vec<PrimitiveCopyInfo>,
}

impl ModelCopyInfo {
    pub fn new(primitives_copy_data: Vec<PrimitiveCopyInfo>) -> Self {
        ModelCopyInfo {
            primitives_copy_data,
        }
    }

    pub fn access_primitive_data(&self) -> &[PrimitiveCopyInfo] {
        self.primitives_copy_data.as_slice()
    }

    pub fn compute_total_required_size(&self) -> usize {
        let mut size = 0;
        for primitive_copy_data in &self.primitives_copy_data {
            size += primitive_copy_data.mesh_size;
            size += primitive_copy_data.indices_size;
            size += primitive_copy_data.textures_size;
        }
        size as usize
    }
}

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
        dst_ptr: *mut u8,
    ) -> ModelCopyInfo;
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
