use super::model_reader::*;

use gltf::json::accessor::ComponentType;
use gltf::{Material, Semantic};
use nalgebra;
use nalgebra::Vector3;
use std::any::{Any, TypeId};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of_val;
use std::ops::{BitAnd, Deref, DivAssign};
use std::path::Path;

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
    /*
    Given a path, open a gltf file and perform little validation on it:
    one mesh and one buffer
    every vertex has size 12
    every normal has size 12
    every tangent has size 16
    every tex_coord has size 8
    */
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
                            assert_eq!(primitive_attribute.1.size(), 12);
                            Some((MeshAttributeType::VERTICES, primitive_attribute.1))
                        }
                        Semantic::Normals => {
                            assert_eq!(primitive_attribute.1.size(), 12);
                            Some((MeshAttributeType::NORMALS, primitive_attribute.1))
                        }
                        Semantic::Tangents => {
                            assert_eq!(primitive_attribute.1.size(), 16);
                            Some((MeshAttributeType::TANGENTS, primitive_attribute.1))
                        }
                        Semantic::TexCoords(0) => {
                            assert_eq!(primitive_attribute.1.size(), 8);
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
                macro_rules! check_txt_existence_and_append {
                    ( $($texture_type:expr, $texture:expr), *) => {
                        $(
                            if let Some(v) = $texture {
                                textures.insert($texture_type, images.get(v.texture().index()).unwrap() as *const gltf::image::Data);
                            }
                        )*
                    };
                }
                let primitive_material = primitive_data.material();
                check_txt_existence_and_append!(
                    TextureType::ALBEDO,
                    primitive_material
                        .pbr_metallic_roughness()
                        .base_color_texture(),
                    TextureType::ORM,
                    primitive_material
                        .pbr_metallic_roughness()
                        .metallic_roughness_texture(),
                    TextureType::NORMAL,
                    primitive_material.normal_texture(),
                    TextureType::EMISSIVE,
                    primitive_material.emissive_texture()
                );

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

    // normalize each primitive mesh individually
    fn normalize_vectors(&mut self) {
        self.primitives.iter().for_each(|primitive| unsafe {
            if let Some(vertex_attribute) =
                primitive.mesh_attributes.get(&MeshAttributeType::VERTICES)
            {
                let vertex_data_slice = std::slice::from_raw_parts_mut(
                    self.buffer_data
                        .as_ptr()
                        .add(vertex_attribute.buffer_data_start as usize)
                        as *mut nalgebra::Vector3<f32>,
                    vertex_attribute.buffer_data_len as usize
                        / std::mem::size_of::<nalgebra::Vector3<f32>>(),
                );

                let max_val = vertex_data_slice.iter().fold(0f32, |max_len: f32, elem| {
                    let elem_magnitude = elem.magnitude();
                    match elem_magnitude.partial_cmp(&max_len) {
                        Some(Ordering::Greater) => elem_magnitude,
                        _ => max_len,
                    }
                });
                vertex_data_slice.iter_mut().for_each(|mut elem| {
                    elem.div_assign(max_val);
                });
            }
        })
    }

    fn copy_model_data_to_ptr(
        &self,
        mesh_attributes_types_to_copy: MeshAttributeType,
        textures_to_copy: TextureType,
        dst_ptr: *mut c_void,
    ) -> u64 {
        if dst_ptr.is_null() {
            return self
                .compute_and_validate_copy_size(mesh_attributes_types_to_copy, textures_to_copy);
        }
        0
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

    fn compute_and_validate_copy_size(
        &self,
        requested_mesh_attributes: MeshAttributeType,
        requested_textures: TextureType,
    ) -> u64 {
        self.primitives
            .iter()
            .fold(0, |mut progressive_size, primitive| {
                (0..8).for_each(|i| {
                    let current_attribute = MeshAttributeType::from_bits(1 << i).unwrap();
                    if requested_mesh_attributes.contains(current_attribute) {
                        match primitive.mesh_attributes.get(&current_attribute) {
                            Some(v) => progressive_size += v.buffer_data_len,
                            None => panic!("Mesh attribute {:?} not found", current_attribute),
                        }
                    }

                    let current_attribute = TextureType::from_bits(1 << i).unwrap();
                    if requested_textures.contains(current_attribute) {
                        match primitive.textures.get(&current_attribute) {
                            Some(v) => progressive_size += unsafe { (*(*v)).pixels.len() as u64 },
                            None => panic!("Texture type {:?} not found", current_attribute),
                        }
                    }
                });
                progressive_size
            })
    }
}
