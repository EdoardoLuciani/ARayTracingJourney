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

pub trait ModelReader {
    fn open(file_path: &Path) -> Self;
}
