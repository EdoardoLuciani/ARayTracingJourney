use super::super::vk_allocator::vk_buffers_suballocator::SubAllocationData;
use super::super::vk_allocator::vk_descriptor_sets_allocator::*;
use super::super::vk_allocator::vk_memory_resource_allocator::*;
use super::super::vk_allocator::VkAllocator;
use super::super::vk_boot::helper::vk_create_shader_stage;
use ash::vk;
use gpu_allocator::MemoryLocation;
use itertools::Itertools;
use nalgebra::*;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

const XE_GTAO_DEPTH_MIP_LEVELS: u32 = 5;

// xegtao settings
const XE_GTAO_RADIUS: f32 = 0.5f32;
const XE_GTAO_DEFAULT_RADIUS_MULTIPLIER: f32 = 1.457f32;
const XE_GTAO_DEFAULT_SAMPLE_DISTRIBUTION_POWER: f32 = 2.0f32;
const XE_GTAO_DEFAULT_FALLOFF_RANGE: f32 = 0.615f32;
const XE_GTAO_DEFAULT_THIN_OCCLUDER_COMPENSATION: f32 = 0.0f32;
const XE_GTAO_DEFAULT_FINAL_VALUE_POWER: f32 = 2.2f32;
const XE_GTAO_DEFAULT_DEPTH_MIP_SAMPLING_OFFSET: f32 = 3.30f32;

const XE_GTAO_NUMTHREADS_X: u32 = 8;
const XE_GTAO_NUMTHREADS_Y: u32 = 8;

#[repr(C, packed)]
struct GTAOConstants {
    viewport_size: Vector2<i32>,
    viewport_pixel_size: Vector2<f32>,

    depth_unpack_consts: Vector2<f32>,
    camera_tan_half_fov: Vector2<f32>,

    ndc_to_view_mul: Vector2<f32>,
    ndc_to_view_add: Vector2<f32>,

    ndc_to_view_mul_x_pixel_size: Vector2<f32>,
    effect_radius: f32,
    effect_falloff_range: f32,

    radius_multiplier: f32,
    padding0: f32,
    final_value_power: f32,
    denoise_blur_beta: f32,

    sample_distribution_power: f32,
    thin_occluder_compensation: f32,
    depth_mipsampling_offset: f32,
    noise_index: i32,
}

struct Stage {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: DescriptorSetAllocation,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl Stage {
    fn dispatch(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        image_memory_barriers: &[vk::ImageMemoryBarrier2],
        additional_descriptor_set: vk::DescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
    ) {
        let dependency_info =
            vk::DependencyInfo::builder().image_memory_barriers(image_memory_barriers);
        unsafe {
            device.cmd_pipeline_barrier2(cb, &dependency_info);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[
                    self.descriptor_set.get_descriptor_sets()[0],
                    additional_descriptor_set,
                ],
                &[],
            );
            device.cmd_dispatch(cb, group_count_x, group_count_y, 1);
        }
    }
}

#[derive(Copy, Clone)]
pub enum DenoiseLevel {
    Disabled = 0,
    Sharp = 1,
    Medium = 2,
    Soft = 3,
}
#[non_exhaustive]
pub struct QualityLevel;
impl QualityLevel {
    pub const LOW: (f32, f32) = (1f32, 2f32);
    pub const MEDIUM: (f32, f32) = (2f32, 2f32);
    pub const HIGH: (f32, f32) = (3f32, 3f32);
    pub const ULTRA: (f32, f32) = (9f32, 3f32);
}
pub struct GtaoSettings {
    pub denoise: DenoiseLevel,
    pub quality: (f32, f32),
}

pub struct VkXeGtao {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    rendering_resolution: vk::Extent2D,
    gtao_settings: GtaoSettings,
    xe_gtao_constants_descriptor_set_layout: vk::DescriptorSetLayout,
    xe_gtao_constants_descriptor_set: DescriptorSetAllocation,
    xe_gtao_host_allocation: SubAllocationData,
    input_depth_image: vk::Image,
    input_depth_image_view: vk::ImageView,
    input_normal_image: vk::Image,
    input_normal_image_view: vk::ImageView,
    filter_depth_image: ImageAllocation,
    filter_depth_image_views: Vec<vk::ImageView>,
    filter_depth_single_image_view: vk::ImageView,
    ao_image: ImageAllocation,
    ao_image_view: vk::ImageView,
    edges_image: ImageAllocation,
    edges_image_view: vk::ImageView,
    #[cfg(debug_assertions)]
    debug_image: ImageAllocation,
    #[cfg(debug_assertions)]
    debug_image_view: vk::ImageView,
    out_ao_image: ImageAllocation,
    out_ao_image_view: vk::ImageView,
    filter_depth_image_sampler: vk::Sampler,
    shader_stages: Vec<Stage>,
    frame_idx: u64,
}

impl VkXeGtao {
    pub fn new(
        device: Rc<ash::Device>,
        allocator: Rc<RefCell<VkAllocator>>,
        rendering_resolution: vk::Extent2D,
        gtao_settings: GtaoSettings,
        shader_spirv_location: &Path,
        input_depth_image: vk::Image,
        input_depth_image_view: vk::ImageView,
        input_normal_image: vk::Image,
        input_normal_image_view: vk::ImageView,
    ) -> Self {
        let xe_gtao_constants_descriptor_set_layout = {
            let xe_gtao_constants_layout_bindings = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);
            Self::create_descriptor_set_layout(
                &device,
                std::slice::from_ref(&xe_gtao_constants_layout_bindings),
            )
        };
        let xe_gtao_constants_descriptor_set = allocator
            .as_ref()
            .borrow_mut()
            .get_descriptor_set_allocator_mut()
            .allocate_descriptor_sets(&[xe_gtao_constants_descriptor_set_layout]);
        let xe_gtao_host_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_host_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<GTAOConstants>(), 128);
        unsafe {
            let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(xe_gtao_host_allocation.get_buffer())
                .offset(xe_gtao_host_allocation.get_buffer_offset() as u64)
                .range(std::mem::size_of::<GTAOConstants>() as u64);
            let write_descriptor_set = vk::WriteDescriptorSet::builder()
                .dst_set(xe_gtao_constants_descriptor_set.get_descriptor_sets()[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&descriptor_buffer_info));
            device.update_descriptor_sets(&[write_descriptor_set.build()], &[]);
        }

        let filter_depth_image_sampler = unsafe {
            let sampler_ci = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mip_lod_bias(0.0f32)
                .anisotropy_enable(false)
                .compare_op(vk::CompareOp::NEVER)
                .min_lod(0.0f32)
                .max_lod(vk::LOD_CLAMP_NONE)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .unnormalized_coordinates(false);
            device.create_sampler(&sampler_ci, None).unwrap()
        };

        let shader_stages = Self::create_shader_stages(
            &device,
            &[xe_gtao_constants_descriptor_set_layout],
            allocator
                .as_ref()
                .borrow_mut()
                .get_descriptor_set_allocator_mut(),
            shader_spirv_location,
            filter_depth_image_sampler,
            &gtao_settings,
        );

        let mut ret = Self {
            device,
            allocator,
            rendering_resolution,
            gtao_settings,
            xe_gtao_constants_descriptor_set_layout,
            xe_gtao_constants_descriptor_set,
            xe_gtao_host_allocation,
            input_depth_image,
            input_depth_image_view,
            input_normal_image,
            input_normal_image_view,
            filter_depth_image: unsafe { std::mem::zeroed() },
            filter_depth_image_views: vec![],
            filter_depth_single_image_view: vk::ImageView::null(),
            ao_image: unsafe { std::mem::zeroed() },
            ao_image_view: vk::ImageView::null(),
            edges_image: unsafe { std::mem::zeroed() },
            edges_image_view: vk::ImageView::null(),
            #[cfg(debug_assertions)]
            debug_image: unsafe { std::mem::zeroed() },
            #[cfg(debug_assertions)]
            debug_image_view: vk::ImageView::null(),
            out_ao_image: unsafe { std::mem::zeroed() },
            out_ao_image_view: vk::ImageView::null(),
            filter_depth_image_sampler,
            shader_stages,
            frame_idx: 0,
        };
        ret.resize(
            rendering_resolution,
            input_depth_image,
            input_depth_image_view,
            input_normal_image,
            input_normal_image_view,
        );

        let constants = unsafe {
            &mut *(ret.xe_gtao_host_allocation.get_host_ptr().unwrap().as_ptr()
                as *mut GTAOConstants)
        };

        constants.effect_radius = XE_GTAO_RADIUS;
        constants.effect_falloff_range = XE_GTAO_DEFAULT_FALLOFF_RANGE;
        constants.radius_multiplier = XE_GTAO_DEFAULT_RADIUS_MULTIPLIER;
        constants.sample_distribution_power = XE_GTAO_DEFAULT_SAMPLE_DISTRIBUTION_POWER;
        constants.thin_occluder_compensation = XE_GTAO_DEFAULT_THIN_OCCLUDER_COMPENSATION;
        constants.final_value_power = XE_GTAO_DEFAULT_FINAL_VALUE_POWER;
        constants.depth_mipsampling_offset = XE_GTAO_DEFAULT_DEPTH_MIP_SAMPLING_OFFSET;
        constants.padding0 = 0.0f32;
        constants.denoise_blur_beta = match ret.gtao_settings.denoise as u8 {
            0 => 1e4f32,
            _ => 1.2f32,
        };

        ret
    }

    pub fn resize(
        &mut self,
        rendering_resolution: vk::Extent2D,
        input_depth_image: vk::Image,
        input_depth_image_view: vk::ImageView,
        input_normal_image: vk::Image,
        input_normal_image_view: vk::ImageView,
    ) {
        self.rendering_resolution = rendering_resolution;

        self.input_depth_image = input_depth_image;
        self.input_depth_image_view = input_depth_image_view;

        self.input_normal_image = input_normal_image;
        self.input_normal_image_view = input_normal_image_view;

        self.recreate_filter_depth_image(self.rendering_resolution);

        Self::replace_output_image(
            &self.device,
            &mut self.allocator.as_ref().borrow_mut().get_allocator_mut(),
            self.rendering_resolution,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            &mut self.ao_image,
            &mut self.ao_image_view,
        );
        Self::replace_output_image(
            &self.device,
            &mut self.allocator.as_ref().borrow_mut().get_allocator_mut(),
            self.rendering_resolution,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            &mut self.edges_image,
            &mut self.edges_image_view,
        );

        #[cfg(debug_assertions)]
        Self::replace_output_image(
            &self.device,
            &mut self.allocator.as_ref().borrow_mut().get_allocator_mut(),
            self.rendering_resolution,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE,
            &mut self.debug_image,
            &mut self.debug_image_view,
        );

        Self::replace_output_image(
            &self.device,
            &mut self.allocator.as_ref().borrow_mut().get_allocator_mut(),
            self.rendering_resolution,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            &mut self.out_ao_image,
            &mut self.out_ao_image_view,
        );

        self.write_shader_stages_descriptor_sets();

        let constants = unsafe {
            &mut *(self
                .xe_gtao_host_allocation
                .get_host_ptr()
                .unwrap()
                .as_ptr() as *mut GTAOConstants)
        };
        constants.viewport_size = Vector2::new(
            rendering_resolution.width as i32,
            rendering_resolution.height as i32,
        );
        constants.viewport_pixel_size = Vector2::new(
            1.0f32 / rendering_resolution.width as f32,
            1.0f32 / rendering_resolution.height as f32,
        );
    }

    pub fn set_constants(&mut self, clip_near: f32, clip_far: f32, fovy: f32, aspect: f32) {
        let constants = unsafe {
            &mut *(self
                .xe_gtao_host_allocation
                .get_host_ptr()
                .unwrap()
                .as_ptr() as *mut GTAOConstants)
        };

        constants.depth_unpack_consts = {
            let depth_linearize_mul = (clip_far * clip_near) / (clip_far - clip_near);

            let mut depth_linearize_add = clip_far / (clip_far - clip_near);

            if depth_linearize_mul * depth_linearize_add < 0.0f32 {
                depth_linearize_add = -depth_linearize_add;
            }
            Vector2::new(depth_linearize_mul, depth_linearize_add)
        };

        let camera_tan_half_fov = {
            let tan_half_fovy = (fovy * 0.5f32).tan();

            let tan_half_fovx = tan_half_fovy * aspect;

            Vector2::new(tan_half_fovx, tan_half_fovy)
        };

        constants.camera_tan_half_fov = camera_tan_half_fov;
        constants.ndc_to_view_mul = Vector2::new(
            camera_tan_half_fov.x * 2.0f32,
            camera_tan_half_fov.y * -2.0f32,
        );
        constants.ndc_to_view_add = Vector2::new(
            camera_tan_half_fov.x * -1.0f32,
            camera_tan_half_fov.y * 1.0f32,
        );

        constants.ndc_to_view_mul_x_pixel_size = {
            let ndc_to_view_mul = constants.ndc_to_view_mul;
            let viewport_pixel_size = constants.viewport_pixel_size;
            Vector2::new(
                ndc_to_view_mul.x * viewport_pixel_size.x,
                ndc_to_view_mul.y * viewport_pixel_size.y,
            )
        };
    }

    pub fn output_ao_image(&self) -> vk::Image {
        match self.gtao_settings.denoise as u8 {
            2 => self.out_ao_image.get_image(),
            _ => self.ao_image.get_image(),
        }
    }

    pub fn compute_ao(&mut self, cb: vk::CommandBuffer) {
        let constants = unsafe {
            &mut *(self
                .xe_gtao_host_allocation
                .get_host_ptr()
                .unwrap()
                .as_ptr() as *mut GTAOConstants)
        };
        constants.noise_index = (self.frame_idx % 64) as i32;
        self.frame_idx += 1;

        unsafe {
            // prefilter pass
            let image_memory_barriers = [
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.input_depth_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.filter_depth_image.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
            ];
            self.shader_stages[0].dispatch(
                &self.device,
                cb,
                &image_memory_barriers,
                self.xe_gtao_constants_descriptor_set.get_descriptor_sets()[0],
                (self.rendering_resolution.width + 16 - 1) / 16,
                (self.rendering_resolution.height + 16 - 1) / 16,
            );

            // main pass
            let image_memory_barriers = [
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.filter_depth_image.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.input_normal_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.ao_image.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.edges_image.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
                #[cfg(debug_assertions)]
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.debug_image.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build(),
            ];
            self.shader_stages[1].dispatch(
                &self.device,
                cb,
                &image_memory_barriers,
                self.xe_gtao_constants_descriptor_set.get_descriptor_sets()[0],
                (self.rendering_resolution.width + XE_GTAO_NUMTHREADS_X - 1) / XE_GTAO_NUMTHREADS_X,
                (self.rendering_resolution.height + XE_GTAO_NUMTHREADS_Y - 1)
                    / XE_GTAO_NUMTHREADS_Y,
            );

            // denoise passes
            let image_memory_barriers = [vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.edges_image.get_image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                })
                .build()];
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);

            let mut denoise_image_src = self.ao_image.get_image();
            let mut denoise_image_dst = self.out_ao_image.get_image();
            for shader_stage in self.shader_stages.iter().skip(2) {
                let image_memory_barriers = [
                    vk::ImageMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(denoise_image_src)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        })
                        .build(),
                    vk::ImageMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(denoise_image_dst)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        })
                        .build(),
                ];
                let dependency_info =
                    vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
                self.device.cmd_pipeline_barrier2(cb, &dependency_info);

                std::mem::swap(&mut denoise_image_src, &mut denoise_image_dst);

                shader_stage.dispatch(
                    &self.device,
                    cb,
                    &image_memory_barriers,
                    self.xe_gtao_constants_descriptor_set.get_descriptor_sets()[0],
                    (self.rendering_resolution.width + (XE_GTAO_NUMTHREADS_X * 2) - 1)
                        / (XE_GTAO_NUMTHREADS_X * 2),
                    (self.rendering_resolution.height + XE_GTAO_NUMTHREADS_Y - 1)
                        / XE_GTAO_NUMTHREADS_Y,
                );
            }
        }
    }

    fn write_shader_stages_descriptor_sets(&self) {
        let prefilter_depths_input_image = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.input_depth_image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let prefilter_depths_output_image = self
            .filter_depth_image_views
            .iter()
            .map(|filter_depth_image_view| {
                vk::DescriptorImageInfo::builder()
                    .sampler(vk::Sampler::null())
                    .image_view(*filter_depth_image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .build()
            })
            .collect_vec();

        let main_pass_prefilter_depth = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.filter_depth_single_image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let main_pass_normal = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.input_normal_image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let main_pass_ao = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.ao_image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let main_pass_edges = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.edges_image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let mut write_descriptor_sets = vec![
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[0].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&prefilter_depths_input_image))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[0].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&prefilter_depths_output_image)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[1].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&main_pass_prefilter_depth))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[1].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&main_pass_normal))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[1].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&main_pass_ao))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[1].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&main_pass_edges))
                .build(),
        ];

        #[cfg(debug_assertions)]
        let main_pass_debug = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.debug_image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        #[cfg(debug_assertions)]
        write_descriptor_sets.push(
            vk::WriteDescriptorSet::builder()
                .dst_set(self.shader_stages[1].descriptor_set.get_descriptor_sets()[0])
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&main_pass_debug))
                .build(),
        );

        let mut denoise_image_view_src = self.ao_image_view;
        let mut denoise_image_view_dst = self.out_ao_image_view;
        let mut denoise_image_infos = Vec::<vk::DescriptorImageInfo>::new();
        for _ in 0..(self.shader_stages.len() - 2) {
            denoise_image_infos.push(vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: denoise_image_view_src,
                image_layout: vk::ImageLayout::GENERAL,
            });
            denoise_image_infos.push(vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: self.edges_image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            });
            denoise_image_infos.push(vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: denoise_image_view_dst,
                image_layout: vk::ImageLayout::GENERAL,
            });
            std::mem::swap(&mut denoise_image_view_src, &mut denoise_image_view_dst);
        }

        self.shader_stages
            .iter()
            .skip(2)
            .enumerate()
            .for_each(|(i, shader_stage)| {
                write_descriptor_sets.push(
                    vk::WriteDescriptorSet::builder()
                        .dst_set(shader_stage.descriptor_set.get_descriptor_sets()[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&denoise_image_infos[i * 3]))
                        .build(),
                );
                write_descriptor_sets.push(
                    vk::WriteDescriptorSet::builder()
                        .dst_set(shader_stage.descriptor_set.get_descriptor_sets()[0])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&denoise_image_infos[i * 3 + 1]))
                        .build(),
                );
                write_descriptor_sets.push(
                    vk::WriteDescriptorSet::builder()
                        .dst_set(shader_stage.descriptor_set.get_descriptor_sets()[0])
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(&denoise_image_infos[i * 3 + 2]))
                        .build(),
                );
            });

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
    }

    fn recreate_filter_depth_image(&mut self, image_extent: vk::Extent2D) {
        let mut val = self.allocator.as_ref().borrow_mut();
        let memory_resource_allocator: &mut VkMemoryResourceAllocator =
            &mut val.get_allocator_mut();

        unsafe {
            self.filter_depth_image_views
                .drain(..)
                .for_each(|image_view| {
                    self.device.destroy_image_view(image_view, None);
                });

            self.device
                .destroy_image_view(self.filter_depth_single_image_view, None);

            take_mut::take(&mut self.filter_depth_image, |image| {
                memory_resource_allocator.destroy_image(image);

                let image_ci = vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16_SFLOAT)
                    .extent(vk::Extent3D {
                        width: image_extent.width,
                        height: image_extent.height,
                        depth: 1,
                    })
                    .mip_levels(XE_GTAO_DEPTH_MIP_LEVELS)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED);
                memory_resource_allocator.allocate_image(&image_ci, MemoryLocation::GpuOnly)
            });
        }

        self.filter_depth_image_views = unsafe {
            (0..XE_GTAO_DEPTH_MIP_LEVELS)
                .map(|i| {
                    let image_view_ci = vk::ImageViewCreateInfo::builder()
                        .image(self.filter_depth_image.get_image())
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R16_SFLOAT)
                        .components(vk::ComponentMapping::default())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: i,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    self.device.create_image_view(&image_view_ci, None).unwrap()
                })
                .collect::<Vec<_>>()
        };

        self.filter_depth_single_image_view = unsafe {
            let image_view_ci = vk::ImageViewCreateInfo::builder()
                .image(self.filter_depth_image.get_image())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16_SFLOAT)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                });
            self.device.create_image_view(&image_view_ci, None).unwrap()
        };
    }

    fn replace_output_image(
        device: &ash::Device,
        allocator: &mut VkMemoryResourceAllocator,
        resolution: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        image: &mut ImageAllocation,
        image_view: &mut vk::ImageView,
    ) {
        unsafe {
            device.destroy_image_view(*image_view, None);
        }
        take_mut::take(image, |image| {
            allocator.destroy_image(image);

            let image_ci = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: resolution.width,
                    height: resolution.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            allocator.allocate_image(&image_ci, MemoryLocation::GpuOnly)
        });

        let image_view_ci = vk::ImageViewCreateInfo::builder()
            .image(image.get_image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        *image_view = unsafe { device.create_image_view(&image_view_ci, None).unwrap() };
    }

    fn create_shader_stages(
        device: &ash::Device,
        additional_global_set_layouts: &[vk::DescriptorSetLayout],
        descriptor_set_allocator: &mut VkDescriptorSetsAllocator,
        shaders_spirv_location: &Path,
        input_depth_sampler: vk::Sampler,
        settings: &GtaoSettings,
    ) -> Vec<Stage> {
        let mut shader_stages = Vec::<Stage>::new();

        // prefilter depth pass
        shader_stages.push({
            let layout_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&[input_depth_sampler])
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(5)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];
            let descriptor_set_layout =
                Self::create_descriptor_set_layout(device, &layout_bindings);
            let descriptor_set =
                descriptor_set_allocator.allocate_descriptor_sets(&[descriptor_set_layout]);

            let mut descriptor_set_layouts = Vec::<vk::DescriptorSetLayout>::new();
            descriptor_set_layouts.push(descriptor_set_layout);
            descriptor_set_layouts.extend_from_slice(additional_global_set_layouts);
            let pipeline_layout = Self::create_pipeline_layout(device, &descriptor_set_layouts);

            let pipeline = Self::create_compute_pipeline(
                device,
                format!(
                    "{}//{}",
                    shaders_spirv_location.to_str().unwrap(),
                    "prefilter_depths.comp.spirv"
                ),
                std::ptr::null(),
                pipeline_layout,
            );

            Stage {
                descriptor_set_layout,
                descriptor_set,
                pipeline_layout,
                pipeline,
            }
        });

        // main pass
        shader_stages.push({
            let layout_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&[input_depth_sampler])
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&[input_depth_sampler])
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];
            let descriptor_set_layout =
                Self::create_descriptor_set_layout(device, &layout_bindings);
            let descriptor_set =
                descriptor_set_allocator.allocate_descriptor_sets(&[descriptor_set_layout]);

            let mut descriptor_set_layouts = Vec::<vk::DescriptorSetLayout>::new();
            descriptor_set_layouts.push(descriptor_set_layout);
            descriptor_set_layouts.extend_from_slice(additional_global_set_layouts);
            let pipeline_layout = Self::create_pipeline_layout(device, &descriptor_set_layouts);

            let entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
            ];
            let binary_data = [
                settings.quality.0.to_ne_bytes(),
                settings.quality.1.to_ne_bytes(),
            ]
            .concat();
            let spec_info = vk::SpecializationInfo::builder()
                .map_entries(&entries)
                .data(&binary_data);

            let pipeline = Self::create_compute_pipeline(
                device,
                format!(
                    "{}//{}",
                    shaders_spirv_location.to_str().unwrap(),
                    "main_pass.comp.spirv"
                ),
                &spec_info.build() as *const _,
                pipeline_layout,
            );

            Stage {
                descriptor_set_layout,
                descriptor_set,
                pipeline_layout,
                pipeline,
            }
        });

        // denoise passes
        let denoise_layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&[input_depth_sampler])
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&[input_depth_sampler])
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];
        let denoise_descriptor_set_layout =
            Self::create_descriptor_set_layout(device, &denoise_layout_bindings);

        let mut descriptor_set_layouts = Vec::<vk::DescriptorSetLayout>::new();
        descriptor_set_layouts.push(denoise_descriptor_set_layout);
        descriptor_set_layouts.extend_from_slice(additional_global_set_layouts);
        let denoise_pipeline_layout = Self::create_pipeline_layout(device, &descriptor_set_layouts);

        if settings.denoise as u8 > 1 {
            let denoise_pipeline = Self::create_compute_pipeline(
                device,
                format!(
                    "{}//{}",
                    shaders_spirv_location.to_str().unwrap(),
                    "denoise.comp.spirv"
                ),
                std::ptr::null(),
                denoise_pipeline_layout,
            );

            (0..settings.denoise as u8 - 1).for_each(|_| {
                shader_stages.push(Stage {
                    descriptor_set_layout: denoise_descriptor_set_layout,
                    descriptor_set: descriptor_set_allocator
                        .allocate_descriptor_sets(&[denoise_descriptor_set_layout]),
                    pipeline_layout: denoise_pipeline_layout,
                    pipeline: denoise_pipeline,
                });
            });
        }

        let denoise_last_pipeline = Self::create_compute_pipeline(
            device,
            format!(
                "{}//{}",
                shaders_spirv_location.to_str().unwrap(),
                "denoise_last.comp.spirv"
            ),
            std::ptr::null(),
            denoise_pipeline_layout,
        );

        shader_stages.push(Stage {
            descriptor_set_layout: denoise_descriptor_set_layout,
            descriptor_set: descriptor_set_allocator
                .allocate_descriptor_sets(&[denoise_descriptor_set_layout]),
            pipeline_layout: denoise_pipeline_layout,
            pipeline: denoise_last_pipeline,
        });

        shader_stages
    }

    fn create_descriptor_set_layout(
        device: &ash::Device,
        descriptor_set_layout_binding: &[vk::DescriptorSetLayoutBinding],
    ) -> vk::DescriptorSetLayout {
        let descriptor_set_layout_ci =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(descriptor_set_layout_binding);
        unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None)
                .unwrap()
        }
    }

    fn create_pipeline_layout(
        device: &ash::Device,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> vk::PipelineLayout {
        let pipeline_layout_ci =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
        unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .unwrap()
        }
    }

    fn create_compute_pipeline<T: AsRef<Path>>(
        device: &ash::Device,
        shader_spirv_file: T,
        cs_shader_spec_info: *const vk::SpecializationInfo,
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let mut shader_stage = vk_create_shader_stage(shader_spirv_file, device);
        shader_stage.p_specialization_info = cs_shader_spec_info;

        let compute_pipelines_ci = vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage)
            .layout(pipeline_layout)
            .build();
        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipelines_ci], None)
                .unwrap()[0]
        };
        unsafe {
            device.destroy_shader_module(compute_pipelines_ci.stage.module, None);
        }
        pipeline
    }
}

impl Drop for VkXeGtao {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.xe_gtao_constants_descriptor_set_layout, None);

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_descriptor_set_allocator_mut()
                .free_descriptor_sets(std::mem::replace(
                    &mut self.xe_gtao_constants_descriptor_set,
                    DescriptorSetAllocation::null(),
                ));
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_host_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.xe_gtao_host_allocation,
                    std::mem::zeroed(),
                ));

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(
                    &mut self.filter_depth_image,
                    std::mem::zeroed(),
                ));
            self.device
                .destroy_image_view(self.filter_depth_single_image_view, None);
            self.filter_depth_image_views.iter().for_each(|image_view| {
                self.device.destroy_image_view(*image_view, None);
            });

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(&mut self.ao_image, std::mem::zeroed()));
            self.device.destroy_image_view(self.ao_image_view, None);

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(&mut self.edges_image, std::mem::zeroed()));
            self.device.destroy_image_view(self.edges_image_view, None);

            #[cfg(debug_assertions)]
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(&mut self.debug_image, std::mem::zeroed()));
            #[cfg(debug_assertions)]
            self.device.destroy_image_view(self.debug_image_view, None);

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(
                    &mut self.out_ao_image,
                    std::mem::zeroed(),
                ));
            self.device.destroy_image_view(self.out_ao_image_view, None);

            self.device
                .destroy_sampler(self.filter_depth_image_sampler, None);

            let shader_stages_last_idx = self.shader_stages.len() - 1;
            self.shader_stages
                .iter_mut()
                .enumerate()
                .for_each(|(i, shader_stage)| {
                    if i < 3 {
                        self.device.destroy_descriptor_set_layout(
                            shader_stage.descriptor_set_layout,
                            None,
                        );
                        self.device
                            .destroy_pipeline_layout(shader_stage.pipeline_layout, None);
                    }

                    if i < 3 || i == shader_stages_last_idx {
                        self.device.destroy_pipeline(shader_stage.pipeline, None);
                    }

                    self.allocator
                        .as_ref()
                        .borrow_mut()
                        .get_descriptor_set_allocator_mut()
                        .free_descriptor_sets(std::mem::replace(
                            &mut shader_stage.descriptor_set,
                            DescriptorSetAllocation::null(),
                        ));
                });
        }
    }
}
