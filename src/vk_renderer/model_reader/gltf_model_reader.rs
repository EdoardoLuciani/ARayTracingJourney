use super::model_reader::*;
use crate::vk_renderer::get_binary_shader_data;
use gltf;
use gltf::json::accessor::ComponentType;
use gltf::{Material, Semantic};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::path::Path;
use winit::window::CursorIcon::Default;

struct GltfPrimitiveMeshAttribute {
    buffer_data_start: u64,
    buffer_data_len: u64,
    element_size: u32,
}

struct GltfPrimitive {
    // Mesh attribute | mesh attribute data
    mesh_attributes: HashMap<MeshAttributeType, GltfPrimitiveMeshAttribute>,
    // Texture type | texture index in gltf images
    textures: HashMap<TextureType, *const gltf::image::Data>,
}

pub struct GltfModelReader {
    buffer_data: Vec<u8>,
    images: Vec<gltf::image::Data>,
    primitives: Vec<GltfPrimitive>,
}

impl ModelReader for GltfModelReader {
    fn open(file_path: &Path) -> Self {
        let (document, mut buffers, images) = gltf::import(file_path)
            .expect(format!("Could not read file {:?}", file_path.as_os_str()).as_str());
        assert_eq!(document.meshes().count(), 1);
        assert_eq!(document.buffers().count(), 1);

        let primitives = document
            .meshes()
            .nth(0)
            .unwrap()
            .primitives()
            .map(|primitive_data| {
                let mut mesh_attributes = HashMap::new();
                if let Some(indices_accessor) = primitive_data.indices() {
                    mesh_attributes.insert(
                        MeshAttributeType::INDICES,
                        GltfModelReader::get_mesh_attribute_from_accessor(indices_accessor),
                    );
                }
                primitive_data
                    .attributes()
                    .filter_map(|primitive_attribute| match primitive_attribute.0 {
                        Semantic::Positions => {
                            Some((MeshAttributeType::VERTICES, primitive_attribute.1))
                        }
                        Semantic::Normals => {
                            Some((MeshAttributeType::NORMALS, primitive_attribute.1))
                        }
                        Semantic::Tangents => {
                            Some((MeshAttributeType::TANGENTS, primitive_attribute.1))
                        }
                        Semantic::TexCoords(0) => {
                            Some((MeshAttributeType::TEX_COORDS, primitive_attribute.1))
                        }
                        _ => None,
                    })
                    .for_each(|elem| {
                        mesh_attributes.insert(
                            elem.0,
                            GltfModelReader::get_mesh_attribute_from_accessor(elem.1),
                        );
                    });

                let mut textures = HashMap::new();
                let primitive_material = primitive_data.material();
                let get_image_data_ptr = |texture: gltf::Texture| -> *const gltf::image::Data {
                    images.get(texture.index()).unwrap()
                };

                macro_rules! check_txt_existence_and_append {
                    ( $($texture_type:ty, $texture:expr), *) => {
                        $(
                            if let Some(v) = $texture {
                                textures.insert($texture_type, get_image_data_ptr($texture.texture()))
                            }
                        )*
                    };
                }

                if let Some(base_color) = primitive_material
                    .pbr_metallic_roughness()
                    .base_color_texture()
                {
                    textures.insert(
                        TextureType::ALBEDO,
                        get_image_data_ptr(base_color.texture()),
                    );
                }
                if let Some(pbr_metallic_roughness) = primitive_material
                    .pbr_metallic_roughness()
                    .metallic_roughness_texture()
                {
                    textures.insert(
                        TextureType::ORM,
                        get_image_data_ptr(pbr_metallic_roughness.texture()),
                    );
                }
                if let Some(normal_texture) = primitive_material.normal_texture() {
                    textures.insert(
                        TextureType::NORMAL,
                        get_image_data_ptr(normal_texture.texture()),
                    );
                }
                if let Some(emissive_texture) = primitive_material.emissive_texture() {
                    textures.insert(
                        TextureType::EMISSIVE,
                        get_image_data_ptr(emissive_texture.texture()),
                    );
                }

                GltfPrimitive {
                    mesh_attributes,
                    textures,
                }
            })
            .collect::<Vec<GltfPrimitive>>();

        GltfModelReader {
            primitives,
            images,
            buffer_data: buffers.remove(0).0,
        }
    }
}

impl GltfModelReader {
    fn get_mesh_attribute_from_accessor(accessor: gltf::Accessor) -> GltfPrimitiveMeshAttribute {
        let accessor_view = accessor.view().unwrap();
        GltfPrimitiveMeshAttribute {
            buffer_data_start: (accessor.offset() + accessor_view.offset()) as u64,
            buffer_data_len: accessor_view.length() as u64,
            element_size: accessor.size() as u32,
        }
    }
}
