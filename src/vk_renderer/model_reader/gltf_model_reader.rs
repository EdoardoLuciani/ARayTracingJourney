use super::model_reader::*;

use ash::vk::Format;
use gltf::{Gltf, Semantic};
use nalgebra;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::Hash;
use std::io::Read;
use std::ops::{Deref, DivAssign};
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
        if let Some(new_format) = coerce_image_to_format {
            gltf_model.coerce_images_to_format(new_format);
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

    // Given a format, convert all images to that one
    fn coerce_images_to_format(&mut self, format: ash::vk::Format) {
        let (dst_map, d_t_size): (_, u8) = match format {
            Format::R8G8B8A8_UNORM => (HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]), 4),
            Format::B8G8R8A8_UNORM => (HashMap::from([('b', 0), ('g', 1), ('r', 2), ('a', 3)]), 4),
            _ => {
                panic!("Unsupported destination format during format coercion")
            }
        };

        for primitive in &mut self.primitives {
            for texture in &mut primitive.textures {
                let (src_map, s_t_size): (_, u8) = match unsafe { (*(*texture.1)).format } {
                    gltf::image::Format::R8G8B8 => {
                        (HashMap::from([('r', 0), ('g', 1), ('b', 2)]), 3)
                    }
                    gltf::image::Format::R8G8B8A8 => {
                        (HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]), 4)
                    }
                    _ => {
                        panic!("Unsupported source format during format coercion")
                    }
                };

                let mut conversion_map = Self::generate_src_to_dst_map(&src_map, &dst_map);
                if s_t_size != d_t_size || conversion_map.iter().any(|v| v.0 != v.1) {
                    let new_data = GltfModelReader::permute_pixels(
                        unsafe { (*(*texture.1)).pixels.as_slice() },
                        s_t_size as usize,
                        &conversion_map,
                        d_t_size as usize,
                    );
                    unsafe {
                        let image_data = ((*texture.1) as *mut gltf::image::Data);
                        (*image_data).pixels = new_data;
                        (*image_data).format = match format {
                            Format::R8G8B8A8_UNORM => gltf::image::Format::R8G8B8A8,
                            Format::B8G8R8A8_UNORM => gltf::image::Format::B8G8R8A8,
                            _ => panic!("Unsupported destination format conversion"),
                        };
                    }
                }
            }
        }
    }

    fn generate_src_to_dst_map(
        src_map: &HashMap<char, u8>,
        dst_map: &HashMap<char, u8>,
    ) -> HashMap<usize, usize> {
        let mut conversion_map = HashMap::<usize, usize>::new();
        for (s_c, s_i) in src_map.iter() {
            if let Some(d_i) = dst_map.get(s_c) {
                conversion_map.insert(*s_i as usize, *d_i as usize);
            }
        }
        conversion_map
    }

    fn permute_pixels(
        src_data: &[u8],
        src_texel_size: usize,
        source_to_destination_map: &HashMap<usize, usize>,
        dst_texel_size: usize,
    ) -> Vec<u8> {
        let mut out_data = Vec::<u8>::new();
        out_data.resize((src_data.len() / src_texel_size) * dst_texel_size, 0);
        let mut written_out_data: usize = 0;

        for src_texel_idx in 0..src_data.len() / src_texel_size {
            for src_byte_idx in 0..src_texel_size {
                if let Some(dst_byte_idx) = source_to_destination_map.get(&src_byte_idx) {
                    out_data[written_out_data + dst_byte_idx] =
                        src_data[src_texel_idx * src_texel_size + src_byte_idx];
                }
            }
            written_out_data += dst_texel_size;
        }
        out_data
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

#[cfg(test)]
mod tests {
    use crate::GltfModelReader;
    use std::collections::HashMap;

    #[test]
    fn wide_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];

        let conversion_map = HashMap::from([(0, 0), (1, 1), (2, 2)]);
        let res = GltfModelReader::permute_pixels(&src_data, 3, &conversion_map, 4);

        assert_eq!(res, vec![0, 1, 2, 0, 3, 4, 5, 0])
    }

    #[test]
    fn narrow_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];

        let conversion_map = HashMap::from([(0, 0), (1, 1), (2, 2)]);
        let res = GltfModelReader::permute_pixels(&src_data, 4, &conversion_map, 3);

        assert_eq!(res, vec![0, 1, 2, 4, 5, 6])
    }

    #[test]
    fn mix_and_narrow_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];

        let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1)]);
        let res = GltfModelReader::permute_pixels(&src_data, 4, &conversion_map, 3);

        assert_eq!(res, vec![1, 2, 0, 5, 6, 4])
    }

    #[test]
    fn mix_and_wide_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];

        let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1)]);
        let res = GltfModelReader::permute_pixels(&src_data, 3, &conversion_map, 4);

        assert_eq!(res, vec![1, 2, 0, 0, 4, 5, 3, 0])
    }

    #[test]
    fn wide_src_to_dst_map() {
        let src_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2)]);
        let dst_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]);

        let res = GltfModelReader::generate_src_to_dst_map(&src_map, &dst_map);

        let out_map = HashMap::from([(0, 0), (1, 1), (2, 2)]);
        assert_eq!(res, out_map)
    }

    #[test]
    fn narrow_src_to_dst_map() {
        let src_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]);
        let dst_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2)]);

        let res = GltfModelReader::generate_src_to_dst_map(&src_map, &dst_map);

        let out_map = HashMap::from([(0, 0), (1, 1), (2, 2)]);
        assert_eq!(res, out_map)
    }

    #[test]
    fn wide_mix_src_to_dst_map() {
        let src_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]);
        let dst_map: HashMap<char, u8> = HashMap::from([('b', 0), ('g', 1), ('r', 2)]);

        let res = GltfModelReader::generate_src_to_dst_map(&src_map, &dst_map);

        let out_map = HashMap::from([(0, 2), (1, 1), (2, 0)]);
        assert_eq!(res, out_map)
    }
}
