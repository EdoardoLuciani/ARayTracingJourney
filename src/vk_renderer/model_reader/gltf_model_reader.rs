use super::model_reader::*;
use ash::vk::Format;
use gltf::Semantic;
use nalgebra::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::DivAssign;
use std::path::Path;

struct GltfPrimitiveMeshAttribute {
    buffer_data_start: u64,
    buffer_data_len: u64,
    element_size: u32,
    element_stride: u32,
}

impl GltfPrimitiveMeshAttribute {
    fn get_element_count(&self) -> u64 {
        self.buffer_data_len / self.element_stride as u64
    }

    fn copy_ith_element_to_ptr(&self, buffer_data: &[u8], element_idx: u64, dst_ptr: *mut u8) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer_data.as_ptr().add(
                    (self.buffer_data_start + element_idx * self.element_stride as u64) as usize,
                ),
                dst_ptr,
                self.element_size as usize,
            );
        }
    }
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
        coerce_image_to_format: Option<Format>,
    ) -> Self {
        let (document, mut buffers, images) = gltf::import(file_path)
            .unwrap_or_else(|_| panic!("Could not read file {:?}", file_path.as_os_str()));
        assert_eq!(document.meshes().count(), 1);
        assert_eq!(document.buffers().count(), 1);

        let primitives = document
            .meshes()
            .next()
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
                                let texture_idx = v.texture().source().index();
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

    /* if dst_slice is None, requested fields are validated by checking if they are available in the model,
    if not, the function panics.
    Return value is the metadata info if dst_slice would not be null
    if dst_slice is not None a copy to dst_slice is performed. */
    fn copy_model_data_to_ptr(
        &self,
        mesh_attributes_types_to_copy: MeshAttributeType,
        textures_to_copy: TextureType,
        mut dst_slice: Option<&mut [u8]>,
    ) -> ModelCopyInfo {
        let mut mesh_flags: Vec<MeshAttributeType> =
            bitflag_vec!(MeshAttributeType, mesh_attributes_types_to_copy);
        if mesh_attributes_types_to_copy.contains(MeshAttributeType::INDICES) {
            mesh_flags.pop();
        }
        let texture_flags: Vec<TextureType> = bitflag_vec!(TextureType, textures_to_copy);

        let mut written_bytes: usize = 0;
        let primitives_copy_data = self
            .primitives
            .iter()
            .map(|primitive| {
                let mut copy_data = PrimitiveCopyInfo::default();

                if let Some(first_mesh_flag) = mesh_flags.first() {
                    copy_data.mesh_buffer_offset = written_bytes as u64;
                    let first_mesh_attribute = &primitive.mesh_attributes[first_mesh_flag];
                    let element_count = first_mesh_attribute.get_element_count();
                    for i in 0..element_count {
                        for mesh_flag in &mesh_flags {
                            let attribute_to_copy =
                                primitive.mesh_attributes.get(mesh_flag).unwrap_or_else(|| {
                                    panic!("Mesh attribute {:?} not found", mesh_flag)
                                });
                            if let Some(data) = &mut dst_slice {
                                attribute_to_copy.copy_ith_element_to_ptr(
                                    &self.buffer_data,
                                    i,
                                    unsafe { data.as_mut_ptr().add(written_bytes) },
                                );
                            }
                            written_bytes += attribute_to_copy.element_size as usize;
                        }
                    }
                    copy_data.mesh_size = written_bytes as u64 - copy_data.mesh_buffer_offset;
                    copy_data.single_mesh_element_size =
                        (copy_data.mesh_size / element_count) as u32;
                }

                if mesh_attributes_types_to_copy.contains(MeshAttributeType::INDICES) {
                    copy_data.indices_buffer_offset = written_bytes as u64;
                    let indices_data = primitive
                        .mesh_attributes
                        .get(&MeshAttributeType::INDICES)
                        .unwrap_or_else(|| {
                            panic!(
                                "Attribute {:?} not found in model",
                                MeshAttributeType::INDICES
                            )
                        });
                    copy_data.indices_size =
                        indices_data.get_element_count() * indices_data.element_size as u64;
                    copy_data.single_index_size = indices_data.element_size;

                    for i in 0..indices_data.get_element_count() {
                        if let Some(dst_slice) = &mut dst_slice {
                            indices_data.copy_ith_element_to_ptr(&self.buffer_data, i, unsafe {
                                dst_slice.as_mut_ptr().add(written_bytes)
                            });
                        }
                        written_bytes += indices_data.element_size as usize;
                    }
                }

                if let Some(first_texture_type) = texture_flags.first() {
                    let first_texture = unsafe {
                        (*primitive.textures.get(first_texture_type).unwrap())
                            .as_ref()
                            .unwrap()
                    };
                    copy_data.image_extent = ash::vk::Extent3D {
                        width: first_texture.width,
                        height: first_texture.height,
                        depth: 1,
                    };
                    let component_size = first_texture.pixels.len()
                        / (copy_data.image_extent.width
                            * copy_data.image_extent.height
                            * copy_data.image_extent.depth) as usize;
                    written_bytes =
                        align_offset(written_bytes as u64, component_size as u64) as usize;
                    copy_data.image_buffer_offset = written_bytes as u64;
                    copy_data.image_mip_levels = 1;
                    copy_data.image_layers = texture_flags.len() as u32;
                    copy_data.image_format = match first_texture.format {
                        gltf::image::Format::R8 => Format::R8_UNORM,
                        gltf::image::Format::R8G8 => Format::R8G8_UNORM,
                        gltf::image::Format::R8G8B8 => Format::R8G8B8_UNORM,
                        gltf::image::Format::R8G8B8A8 => Format::R8G8B8A8_UNORM,
                        gltf::image::Format::B8G8R8 => Format::B8G8R8_UNORM,
                        gltf::image::Format::B8G8R8A8 => Format::B8G8R8A8_UNORM,
                        gltf::image::Format::R16 => Format::R16_UNORM,
                        gltf::image::Format::R16G16 => Format::R16G16_UNORM,
                        gltf::image::Format::R16G16B16 => Format::R16G16B16_UNORM,
                        gltf::image::Format::R16G16B16A16 => Format::R16G16B16A16_UNORM,
                    };
                    copy_data.image_size = 0;
                    for texture_type in &texture_flags {
                        let texture_to_copy =
                            *primitive.textures.get(texture_type).unwrap_or_else(|| {
                                panic!("Texture type {:?} not found in model", texture_type)
                            });
                        unsafe {
                            if let Some(dst_slice) = &mut dst_slice {
                                std::ptr::copy_nonoverlapping(
                                    (*texture_to_copy).pixels.as_ptr(),
                                    dst_slice.as_mut_ptr().add(written_bytes),
                                    (*texture_to_copy).pixels.len(),
                                );
                            }
                            written_bytes += (*texture_to_copy).pixels.len();
                        }
                    }
                    copy_data.image_size = written_bytes as u64 - copy_data.image_buffer_offset;
                }
                copy_data
            })
            .collect::<Vec<PrimitiveCopyInfo>>();
        ModelCopyInfo::new(primitives_copy_data)
    }

    fn get_primitives_bounding_sphere(&self) -> Sphere {
        let mut m_radius: f32;
        let mut m_radius2: f32;

        let mut xmax: Vector3<f32> = Vector3::from_element(f32::MIN);
        let mut xmin: Vector3<f32> = Vector3::from_element(f32::MAX);
        let mut ymin: Vector3<f32> = Vector3::from_element(f32::MAX);
        let mut ymax: Vector3<f32> = Vector3::from_element(f32::MIN);
        let mut zmin: Vector3<f32> = Vector3::from_element(f32::MAX);
        let mut zmax: Vector3<f32> = Vector3::from_element(f32::MIN);
        let mut dia1: Vector3<f32>;
        let mut dia2: Vector3<f32>;

        for primitive in &self.primitives {
            let vertex_attribute = &primitive.mesh_attributes[&MeshAttributeType::VERTICES];
            let vertex_data_slice = unsafe {
                std::slice::from_raw_parts(
                    self.buffer_data
                        .as_ptr()
                        .add(vertex_attribute.buffer_data_start as usize),
                    vertex_attribute.buffer_data_len as usize,
                )
            };

            vertex_data_slice
                .windows(vertex_attribute.element_size as usize)
                .step_by(vertex_attribute.element_stride as usize)
                .for_each(|elem| {
                    let vertex = unsafe { *(elem.as_ptr() as *const Vector3<f32>) };
                    if vertex[0] < xmin[0] {
                        xmin = vertex;
                    }
                    if vertex[0] > xmax[0] {
                        xmax = vertex;
                    }
                    if vertex[1] < ymin[1] {
                        ymin = vertex;
                    }
                    if vertex[1] > ymax[1] {
                        ymax = vertex;
                    }
                    if vertex[2] < zmin[2] {
                        zmin = vertex;
                    }
                    if vertex[2] > zmax[2] {
                        zmax = vertex;
                    }
                });
        }

        /* Set *span = distance between the 2 points *min & *max (squared) */
        let xspan = (xmax - xmin).magnitude_squared();
        let yspan = (ymax - ymin).magnitude_squared();
        let zspan = (zmax - zmin).magnitude_squared();

        /* Set points dia1 & dia2 to the maximally separated pair */
        dia1 = xmin;
        dia2 = xmax;

        /* assume xspan biggest */
        let mut maxspan = xspan;

        if yspan > maxspan {
            maxspan = yspan;
            dia1 = ymin;
            dia2 = ymax;
        }

        if zspan > maxspan {
            dia1 = zmin;
            dia2 = zmax;
        }

        /* dia1,dia2 is a diameter of initial sphere */
        /* calc initial center */
        let mut center: Vector3<f32> = (dia1 + dia2) * 0.5f32;

        /* calculate initial radius**2 and radius */
        m_radius2 = (dia2 - center).magnitude_squared();
        m_radius = m_radius2.sqrt();

        /* SECOND PASS: increment current sphere */
        for primitive in &self.primitives {
            let vertex_attribute = &primitive.mesh_attributes[&MeshAttributeType::VERTICES];
            let vertex_data_slice = unsafe {
                std::slice::from_raw_parts(
                    self.buffer_data
                        .as_ptr()
                        .add(vertex_attribute.buffer_data_start as usize),
                    vertex_attribute.buffer_data_len as usize,
                )
            };

            vertex_data_slice
                .windows(vertex_attribute.element_size as usize)
                .step_by(vertex_attribute.element_stride as usize)
                .for_each(|elem| {
                    let vertex = unsafe { *(elem.as_ptr() as *const Vector3<f32>) };

                    let delta: Vector3<f32> = vertex - center;
                    let old_to_p_sq = delta.magnitude_squared();
                    /* do r**2 test first */
                    if old_to_p_sq > m_radius2 {
                        /* this point is outside of current sphere */
                        let old_to_p = old_to_p_sq.sqrt();
                        /* calc radius of new sphere */
                        m_radius = (m_radius + old_to_p) * 0.5f32;
                        m_radius2 = m_radius * m_radius; /* for next r**2 compare */
                        let old_to_new = old_to_p - m_radius;
                        /* calc center of new sphere */
                        let recip = 1.0f32 / old_to_p;
                        center = (m_radius * center + old_to_new * vertex) * recip;
                    }
                });
        }
        Sphere::new(center, m_radius)
    }
}

impl GltfModelReader {
    fn get_mesh_attribute_from_accessor(accessor: gltf::Accessor) -> GltfPrimitiveMeshAttribute {
        let buffer_view = accessor.view().unwrap();
        let element_stride = buffer_view.stride().unwrap_or(accessor.size());
        GltfPrimitiveMeshAttribute {
            buffer_data_start: (accessor.offset() + buffer_view.offset()) as u64,
            buffer_data_len: (accessor.count() * element_stride) as u64,
            element_size: accessor.size() as u32,
            element_stride: element_stride as u32,
        }
    }

    // normalize each primitive mesh individually
    fn normalize_vectors(&mut self) {
        self.primitives.iter().for_each(|primitive| unsafe {
            if let Some(vertex_attribute) =
                primitive.mesh_attributes.get(&MeshAttributeType::VERTICES)
            {
                let vertex_data_slice = std::slice::from_raw_parts_mut(
                    self.buffer_data.as_mut_ptr(),
                    vertex_attribute.buffer_data_len as usize,
                );

                let max_val = vertex_data_slice
                    .windows(vertex_attribute.element_size as usize)
                    .step_by(vertex_attribute.element_stride as usize)
                    .fold(0f32, |max_len: f32, elem| {
                        let position = *(elem.as_ptr() as *const Vector3<f32>);
                        let magnitude = position.magnitude();
                        match magnitude.partial_cmp(&max_len) {
                            Some(Ordering::Greater) => magnitude,
                            _ => max_len,
                        }
                    });

                vertex_data_slice
                    .windows(vertex_attribute.element_size as usize)
                    .step_by(vertex_attribute.element_stride as usize)
                    .for_each(|elem| {
                        let mut position = *(elem.as_ptr() as *const Vector3<f32>);
                        position.div_assign(max_val);
                    });
            }
        })
    }

    // Given a format, convert all images to that one
    fn coerce_images_to_format(&mut self, format: Format) {
        let (dst_map, d_t_size): (_, u8) = match format {
            Format::R8G8B8A8_UNORM => (HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]), 4),
            Format::B8G8R8A8_UNORM => (HashMap::from([('b', 0), ('g', 1), ('r', 2), ('a', 3)]), 4),
            Format::B8G8R8_UNORM => (HashMap::from([('b', 0), ('g', 1), ('r', 2)]), 3),
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
                    gltf::image::Format::B8G8R8A8 => {
                        (HashMap::from([('r', 2), ('g', 1), ('b', 0), ('a', 3)]), 4)
                    }
                    _ => {
                        panic!("Unsupported source format during format coercion")
                    }
                };

                let conversion_map = Self::generate_src_to_dst_map(&src_map, &dst_map);
                if s_t_size != d_t_size || conversion_map.iter().any(|(src, dst)| src != dst) {
                    if s_t_size == d_t_size && s_t_size % 4 == 0 {
                        let src_data = unsafe {
                            (*((*texture.1) as *mut gltf::image::Data))
                                .pixels
                                .as_mut_slice()
                        };
                        // For some weird reason the ssse is 2x faster on my processor than the avx2
                        GltfModelReader::permute_pixels_same_size_x86_ssse(
                            src_data,
                            s_t_size,
                            &conversion_map,
                        );
                    } else {
                        let new_data = GltfModelReader::permute_pixels(
                            unsafe { (*(*texture.1)).pixels.as_slice() },
                            s_t_size,
                            &conversion_map,
                            d_t_size,
                        );
                        unsafe {
                            (*((*texture.1) as *mut gltf::image::Data)).pixels = new_data;
                        }
                    }

                    unsafe {
                        let image_data = (*texture.1) as *mut gltf::image::Data;
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
    ) -> HashMap<u8, u8> {
        let mut conversion_map = HashMap::<u8, u8>::new();
        for (s_c, s_i) in src_map.iter() {
            if let Some(d_i) = dst_map.get(s_c) {
                conversion_map.insert(*s_i, *d_i);
            }
        }
        conversion_map
    }

    fn permute_pixels(
        src_data: &[u8],
        src_texel_size: u8,
        src_to_dst_map: &HashMap<u8, u8>,
        dst_texel_size: u8,
    ) -> Vec<u8> {
        // The conversion from hashmap to vec is done because it grants a 4x speedup
        // while looking up the keys
        let mut src_to_dst_map_vec: Vec<Option<u8>> =
            vec![None; (*src_to_dst_map.keys().max().unwrap() + 1) as usize];
        src_to_dst_map.iter().for_each(|(k, v)| {
            src_to_dst_map_vec[*k as usize] = Some(*v as u8);
        });

        let mut out_data = Vec::<u8>::new();
        out_data.resize(
            (src_data.len() / src_texel_size as usize) * dst_texel_size as usize,
            0,
        );
        let mut written_out_data: usize = 0;

        for src_texel_idx in 0..src_data.len() / src_texel_size as usize {
            for src_byte_idx in 0..std::cmp::min(src_texel_size, src_to_dst_map_vec.len() as u8) {
                if let Some(dst_byte_idx) = src_to_dst_map_vec[src_byte_idx as usize] {
                    out_data[written_out_data + dst_byte_idx as usize] =
                        src_data[src_texel_idx * src_texel_size as usize + src_byte_idx as usize];
                }
            }
            written_out_data += dst_texel_size as usize;
        }
        out_data
    }

    // Permutes pixels with ssse instructions (faster than avx2 version on my processor)
    fn permute_pixels_same_size_x86_ssse(
        src_data: &mut [u8],
        texel_size: u8,
        src_to_dst_map: &HashMap<u8, u8>,
    ) {
        use ::core::arch::x86_64::*;
        unsafe {
            let mut mask_arr: [i8; std::mem::size_of::<__m128i>()] = [0; 16];
            let simd_block_iter = (0..std::mem::size_of::<__m128i>()).step_by(texel_size as usize);
            for texel_starting_byte in simd_block_iter {
                for texel_component_idx in 0..texel_size {
                    if let Some(dst_idx) = src_to_dst_map.get(&texel_component_idx) {
                        mask_arr[texel_starting_byte + *dst_idx as usize] =
                            (texel_starting_byte as u8 + texel_component_idx) as i8;
                    }
                }
            }

            let mask: __m128i = _mm_loadu_si128(mask_arr.as_ptr() as *const __m128i);
            for i in (0..src_data.len()).step_by(std::mem::size_of::<__m128i>()) {
                let mem_address = src_data.as_ptr().add(i) as *mut __m128i;

                let data: __m128i = _mm_loadu_si128(mem_address);
                let out_data = _mm_shuffle_epi8(data, mask);
                _mm_storeu_si128(mem_address, out_data);
            }
        }
    }

    // Permutes pixels with avx2 instructions (slower than ssse version on my processor)
    fn permute_pixels_same_size_x86_avx2(
        src_data: &mut [u8],
        texel_size: u8,
        src_to_dst_map: &HashMap<u8, u8>,
    ) {
        use ::core::arch::x86_64::*;
        unsafe {
            let mut mask_arr: [i8; std::mem::size_of::<__m256i>()] = [0; 32];
            let simd_block_iter = (0..std::mem::size_of::<__m256i>()).step_by(texel_size as usize);
            for texel_starting_byte in simd_block_iter {
                for texel_component_idx in 0..texel_size {
                    if let Some(dst_idx) = src_to_dst_map.get(&texel_component_idx) {
                        mask_arr[texel_starting_byte + *dst_idx as usize] =
                            (texel_starting_byte as u8 + texel_component_idx) as i8;
                    }
                }
            }

            let mask: __m256i = _mm256_loadu_si256(mask_arr.as_ptr() as *const __m256i);
            for i in (0..src_data.len()).step_by(std::mem::size_of::<__m256i>()) {
                let mem_address = src_data.as_ptr().add(i) as *mut __m256i;

                let data: __m256i = _mm256_loadu_si256(mem_address);
                let out_data = _mm256_shuffle_epi8(data, mask);
                _mm256_storeu_si256(mem_address, out_data);
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
                match *mesh_attribute.0 {
                    MeshAttributeType::VERTICES => assert_eq!(mesh_attribute.1.element_size, 12),
                    MeshAttributeType::TEX_COORDS => assert_eq!(mesh_attribute.1.element_size, 8),
                    MeshAttributeType::NORMALS => assert_eq!(mesh_attribute.1.element_size, 12),
                    MeshAttributeType::TANGENTS => assert_eq!(mesh_attribute.1.element_size, 16),
                    _ => continue,
                }

                if common_element_count.is_none() {
                    common_element_count = Some(mesh_attribute.1.get_element_count());
                } else {
                    assert_eq!(
                        common_element_count.unwrap(),
                        mesh_attribute.1.get_element_count()
                    );
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::iter::zip;

    #[test]
    fn wide_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];

        //(0 -> 0), (1 -> 1), (2 -> 2)
        let conversion_map = HashMap::from([(0, 0), (1, 1), (2, 2)]);
        let res = GltfModelReader::permute_pixels(&src_data, 3, &conversion_map, 4);

        assert_eq!(res, vec![0, 1, 2, 0, 3, 4, 5, 0])
    }

    #[test]
    fn narrow_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];

        //(0 -> 0), (1 -> 1), (2 -> 2)
        let conversion_map = HashMap::from([(0, 0), (1, 1), (2, 2)]);
        let res = GltfModelReader::permute_pixels(&src_data, 4, &conversion_map, 3);

        assert_eq!(res, vec![0, 1, 2, 4, 5, 6])
    }

    #[test]
    fn mix_and_narrow_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // (0 -> 2), (1 -> 0), (2 -> 1)
        let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1)]);
        let res = GltfModelReader::permute_pixels(&src_data, 4, &conversion_map, 3);

        assert_eq!(res, vec![1, 2, 0, 5, 6, 4])
    }

    #[test]
    fn mix_and_wide_permute_pixel() {
        let src_data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];

        // (0 -> 2), (1 -> 0), (2 -> 1)
        let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1)]);
        let res = GltfModelReader::permute_pixels(&src_data, 3, &conversion_map, 4);

        assert_eq!(res, vec![1, 2, 0, 0, 4, 5, 3, 0])
    }

    #[test]
    fn mix_and_wide_permute_pixel_x86_simd() {
        let src_data = (0..128).map(|e| e).collect::<Vec<u8>>();

        let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1), (3, 3)]);

        let normal_result = GltfModelReader::permute_pixels(&src_data, 4, &conversion_map, 4);

        let mut sse_result = src_data.clone();
        GltfModelReader::permute_pixels_same_size_x86_ssse(&mut sse_result, 4, &conversion_map);

        let mut avx2_result = src_data.clone();
        GltfModelReader::permute_pixels_same_size_x86_avx2(&mut avx2_result, 4, &conversion_map);

        assert_eq!(normal_result, sse_result);
        assert_eq!(normal_result, avx2_result);
    }

    #[test]
    fn wide_src_to_dst_map() {
        let src_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2)]);
        let dst_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]);

        let res = GltfModelReader::generate_src_to_dst_map(&src_map, &dst_map);

        // (0 -> 0), (1 -> 1), (2 -> 2)
        assert_eq!(res, HashMap::from([(0, 0), (1, 1), (2, 2)]));
    }

    #[test]
    fn narrow_src_to_dst_map() {
        let src_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]);
        let dst_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2)]);

        let res = GltfModelReader::generate_src_to_dst_map(&src_map, &dst_map);

        assert_eq!(res, HashMap::from([(0, 0), (1, 1), (2, 2)]));
    }

    #[test]
    fn wide_mix_src_to_dst_map() {
        let src_map: HashMap<char, u8> = HashMap::from([('r', 0), ('g', 1), ('b', 2), ('a', 3)]);
        let dst_map: HashMap<char, u8> = HashMap::from([('b', 0), ('g', 1), ('r', 2)]);

        let res = GltfModelReader::generate_src_to_dst_map(&src_map, &dst_map);

        // (0 -> 2), (1 -> 1), (2 -> 0)
        assert_eq!(res, HashMap::from([(0, 2), (1, 1), (2, 0)]));
    }

    #[test]
    fn test_textured_cube() {
        let sponza = GltfModelReader::open(
            "assets/models/BoxTextured.glb".as_ref(),
            true,
            Some(ash::vk::Format::B8G8R8A8_UNORM),
        );
        let sphere = sponza.get_primitives_bounding_sphere();
        assert!((sphere.get_radius() - 1.0f32) < 1e-5);
        assert!(
            (sphere.get_center() - Vector3::<f32>::from_element(1.0))
                < Vector3::<f32>::from_element(1e-5)
        );

        let res = sponza.copy_model_data_to_ptr(
            MeshAttributeType::VERTICES
                | MeshAttributeType::NORMALS
                | MeshAttributeType::TEX_COORDS
                | MeshAttributeType::INDICES,
            TextureType::ALBEDO,
            None,
        );
        let model_size = res.compute_total_size();

        let mut vec_data = vec![0u8; model_size];
        let res = sponza.copy_model_data_to_ptr(
            MeshAttributeType::VERTICES
                | MeshAttributeType::NORMALS
                | MeshAttributeType::TEX_COORDS
                | MeshAttributeType::INDICES,
            TextureType::ALBEDO,
            Some(&mut vec_data),
        );

        let first_vertex_view = unsafe {
            std::slice::from_raw_parts(
                vec_data
                    .as_ptr()
                    .add(res.get_primitive_data().get(0).unwrap().mesh_buffer_offset as usize)
                    as *const f32,
                12,
            )
        };
        let reference_first_vertex = vec![-0.5f32, -0.5, 0.5, 6.0, 0.0, 0.0, 0.0, 1.0];
        for (e1, e2) in zip(first_vertex_view, reference_first_vertex) {
            assert!(e1 - e2 < 1e-7);
        }

        let first_indices = unsafe {
            std::slice::from_raw_parts(
                vec_data.as_ptr().add(
                    res.get_primitive_data()
                        .get(0)
                        .unwrap()
                        .indices_buffer_offset as usize,
                ) as *const u16,
                4,
            )
        };
        assert_eq!(first_indices, vec![0, 1, 2, 3]);

        let first_texture_pixels = unsafe {
            std::slice::from_raw_parts(
                vec_data
                    .as_ptr()
                    .add(res.get_primitive_data().get(0).unwrap().image_buffer_offset as usize)
                    as *const u8,
                4,
            )
        };
        assert_eq!(first_texture_pixels, vec![220, 220, 220, 0]);
    }
}
