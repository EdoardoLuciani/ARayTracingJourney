use bitflags::bitflags;
use std::ffi::c_void;
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

struct PrimitiveCopyData {
    mesh_buffer_offset: Option<u64>,
    mesh_size: Option<u64>,
    single_mesh_element_size: Option<u32>,

    indices_buffer_offset: Option<u64>,
    indices_size: Option<u64>,
    single_index_size: Option<u32>,

    textures_buffer_offset: Option<u64>,
    textures_extent: Option<(u32, u32)>,
    textures_format: ash::vk::Format,
    textures_count: Option<u32>,
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
        dst_ptr: *mut c_void,
    ) -> u64;
}
