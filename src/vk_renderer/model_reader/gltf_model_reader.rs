use super::model_reader::*;

use ash::vk::Format;
use gltf::{Gltf, Semantic};
use nalgebra;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::DivAssign;
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
    /* Given a path, open a gltf file and perform little validation on it:
    one mesh and one buffer
    every vertex has size 12
    every normal has size 12
    every tangent has size 16
    every tex_coord has size 8 */
    fn open(
        file_path: &Path,
        normalize_vectors: bool,
        coerce_image_to_format: Option<ash::vk::Format>,
    ) -> Self {
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
                macro_rules! check_txt_existence_and_append {
                    ( $($texture_type:expr, $texture:expr), *) => {
                        $(
                            if let Some(v) = $texture {
                                let texture_idx = v.texture().index();
                                textures.insert($texture_type, images.get(texture_idx).expect(&format!("Cannot open texture idx {}", texture_idx)) as *const gltf::image::Data);
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

        let mut gltf_model = GltfModelReader {
            primitives,
            images,
            buffer_data: buffers.remove(0).0,
        };
        if normalize_vectors {
            gltf_model.normalize_vectors();
        }

        gltf_model.validate_model();
        gltf_model
    }

    /* if dst_ptr is null, requested fields are validated by checking if they are available in the model,
    if not, the function panics.
    Return value is the size (in bytes) written if dst_ptr would not be null
    if dst_ptr is not null a copy to dst_ptr is performed. */
    fn copy_model_data_to_ptr(
        &self,
        mesh_attributes_types_to_copy: MeshAttributeType,
        textures_to_copy: TextureType,
        dst_ptr: *mut c_void,
    ) -> u64 {
        if dst_ptr.is_null() {
            return self.validate_copy(mesh_attributes_types_to_copy, textures_to_copy);
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

    fn coerce_images_to_format(&mut self, format: ash::vk::Format) {
        let (desired_bytes_per_pixel, desired_component_count) : (u32, u32) = match format {
            Format::R8G8B8A8_UNORM => (1, 4),
            _ => {panic!("Unsupported format requested during format coercion")}
        };

        for primitive in self.primitives {
            for texture in primitive.textures {
                let old_texture_format = unsafe { (*texture.1).format };
                let (current_bytes_per_pixel, current_component_count) : (u32, u32) = match old_texture_format {
                    gltf::image::Format::
                    _ => {panic!("Unsupported current format during format coercion")}
                }
            }
        }

    }

    /* validates the model given the following conditions,
    if one of the following is not valid, the function panic
    every vertex has size 12
    every normal has size 12
    every tangent has size 16
    every tex_coord has size 8
    each primitive has same element count for all available mesh attributes
    each primitive has same extent and format for all textures */
    fn validate_model(&self) {
        self.primitives.iter().for_each(|primitive| {
            let mut common_element_count = None;
            for mesh_attribute in &primitive.mesh_attributes {
                match mesh_attribute.0 {
                    &MeshAttributeType::VERTICES => assert_eq!(mesh_attribute.1.element_size, 12),
                    &MeshAttributeType::TEX_COORDS => assert_eq!(mesh_attribute.1.element_size, 8),
                    &MeshAttributeType::NORMALS => assert_eq!(mesh_attribute.1.element_size, 12),
                    &MeshAttributeType::TANGENTS => assert_eq!(mesh_attribute.1.element_size, 16),
                    _ => continue,
                }

                let mesh_attribute_element_count =
                    mesh_attribute.1.buffer_data_len / mesh_attribute.1.element_size as u64;
                if common_element_count.is_none() {
                    common_element_count = Some(mesh_attribute_element_count);
                } else {
                    assert_eq!(common_element_count.unwrap(), mesh_attribute_element_count);
                }
            }

            let mut common_image_format = None;
            let mut common_image_extent = None;
            for texture in &primitive.textures {
                let texture_data = unsafe { (texture.1).as_ref().unwrap() };
                if common_image_extent.is_none() {
                    common_image_format = Some(texture_data.format);
                    common_image_extent = Some((texture_data.width, texture_data.height));
                } else {
                    assert_eq!(
                        common_image_extent.unwrap(),
                        (texture_data.width, texture_data.height)
                    );
                    assert_eq!(common_image_format.unwrap(), texture_data.format);
                }
            }
        });
    }

    // check if the requested fields are present, panic otherwise
    fn validate_copy(
        &self,
        requested_mesh_attributes: MeshAttributeType,
        requested_textures: TextureType,
    ) -> u64 {
        self.primitives
            .iter()
            .fold(0, |mut progressive_size, primitive| {
                (0..MeshAttributeType::all().bits().count_ones()).for_each(|i| {
                    let current_attribute = MeshAttributeType::from_bits(1 << i).unwrap();
                    if requested_mesh_attributes.contains(current_attribute) {
                        match primitive.mesh_attributes.get(&current_attribute) {
                            Some(v) => progressive_size += v.buffer_data_len,
                            None => panic!("Mesh attribute {:?} not found", current_attribute),
                        }
                    }
                });

                (0..TextureType::all().bits().count_ones()).for_each(|i| {
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
