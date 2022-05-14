use super::super::model_reader::model_reader::align_offset;
use super::super::vk_allocator::vk_descriptor_sets_allocator::*;
use super::super::vk_allocator::vk_memory_resource_allocator::*;
use super::super::vk_allocator::VkAllocator;
use super::super::vk_boot::helper::vk_create_shader_stage;
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use std::cell::RefCell;
use std::ops::DerefMut;
use std::path::Path;
use std::rc::Rc;

const DESCRIPTOR_SET_TLAS_BINDING: u32 = 0;
const DESCRIPTOR_SET_IMAGE_BINDING: u32 = 1;

pub struct VkRTLightningShadows {
    device: Rc<ash::Device>,
    ray_tracing_pipeline_fp: Rc<khr::RayTracingPipeline>,
    allocator: Rc<RefCell<VkAllocator>>,
    rendering_resolution: vk::Extent2D,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,
    output_image: ImageAllocation,
    output_image_view: vk::ImageView,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sbt_buffer: BufferAllocation,
    sbt_regions: [vk::StridedDeviceAddressRegionKHR; 3],
}

impl VkRTLightningShadows {
    pub fn new(
        device: Rc<ash::Device>,
        ray_tracing_pipeline_fp: Rc<khr::RayTracingPipeline>,
        ray_tracing_pipeline_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        allocator: Rc<RefCell<VkAllocator>>,
        rendering_resolution: vk::Extent2D,
        shader_spirv_location: &Path,
        output_format: vk::Format,
    ) -> Self {
        let descriptor_set_layout_tlas_image = unsafe {
            let descriptor_set_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_TLAS_BINDING)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_IMAGE_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .build(),
            ];
            let binding_flags = [
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::empty(),
            ];
            let mut descriptor_set_layout_binding_flags_ci =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                    .binding_flags(&binding_flags);
            let descriptor_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder()
                .push_next(&mut descriptor_set_layout_binding_flags_ci)
                .bindings(&descriptor_set_bindings);
            device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None)
                .unwrap()
        };
        let descriptor_set_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_descriptor_set_allocator_mut()
            .allocate_descriptor_sets(&[descriptor_set_layout_tlas_image]);

        let image_data = Self::create_output_image(
            device.as_ref(),
            allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .deref_mut(),
            rendering_resolution,
            output_format,
        );

        let pipeline_data = Self::create_ray_tracing_pipeline(
            device.as_ref(),
            ray_tracing_pipeline_fp.as_ref(),
            shader_spirv_location,
            &[descriptor_set_layout_tlas_image],
        );

        let sbt_data = Self::create_shader_binding_table(
            ray_tracing_pipeline_fp.as_ref(),
            &ray_tracing_pipeline_properties,
            pipeline_data.1,
            allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .deref_mut(),
        );

        let vk_ray_traced_lightning_shadows = VkRTLightningShadows {
            device,
            ray_tracing_pipeline_fp,
            allocator,
            rendering_resolution,
            descriptor_set_layout: descriptor_set_layout_tlas_image,
            descriptor_set_allocation,
            output_image: image_data.0,
            output_image_view: image_data.1,
            pipeline_layout: pipeline_data.0,
            pipeline: pipeline_data.1,
            sbt_buffer: sbt_data.0,
            sbt_regions: sbt_data.1,
        };
        vk_ray_traced_lightning_shadows.update_output_image_descriptor_set();
        vk_ray_traced_lightning_shadows
    }

    pub fn resize(rendering_resolution: vk::Extent2D) {
        todo!()
    }

    pub fn set_tlas(&self, tlas: vk::AccelerationStructureKHR) {
        self.update_tlas_descriptor_set(tlas);
    }

    pub fn get_output_image(&self) -> vk::Image {
        self.output_image.get_image()
    }

    pub fn trace_rays(&self, cb: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self
                    .descriptor_set_allocation
                    .get_descriptor_sets()
                    .first()
                    .copied()
                    .unwrap()],
                &[],
            );
            self.ray_tracing_pipeline_fp.cmd_trace_rays(
                cb,
                &self.sbt_regions[0],
                &self.sbt_regions[1],
                &self.sbt_regions[2],
                &vk::StridedDeviceAddressRegionKHR {
                    device_address: 0,
                    stride: 0,
                    size: 0,
                },
                self.rendering_resolution.width,
                self.rendering_resolution.height,
                1,
            );
        }
    }

    fn update_tlas_descriptor_set(&self, tlas: vk::AccelerationStructureKHR) {
        let mut write_descriptor_set_acceleration_structure =
            vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                .acceleration_structures(std::slice::from_ref(&tlas));
        let descriptor_set_write = vk::WriteDescriptorSet::builder()
            .push_next(&mut write_descriptor_set_acceleration_structure)
            .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
            .dst_binding(DESCRIPTOR_SET_TLAS_BINDING)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .build();
        unsafe {
            self.device
                .update_descriptor_sets(&[descriptor_set_write], &[]);
        }
    }

    fn update_output_image_descriptor_set(&self) {
        let descriptor_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
            .dst_binding(DESCRIPTOR_SET_IMAGE_BINDING)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(
                &vk::DescriptorImageInfo::builder()
                    .sampler(vk::Sampler::null())
                    .image_view(self.output_image_view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ))
            .build();
        unsafe {
            self.device
                .update_descriptor_sets(&[descriptor_set_write], &[]);
        }
    }

    fn create_output_image(
        device: &ash::Device,
        allocator: &mut VkMemoryResourceAllocator,
        resolution: vk::Extent2D,
        format: vk::Format,
    ) -> (ImageAllocation, vk::ImageView) {
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
            .usage(vk::ImageUsageFlags::STORAGE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image_allocation = allocator.allocate_image(&image_ci, MemoryLocation::GpuOnly);

        let image_view_ci = vk::ImageViewCreateInfo::builder()
            .image(image_allocation.get_image())
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
        let image_view = unsafe { device.create_image_view(&image_view_ci, None).unwrap() };
        (image_allocation, image_view)
    }

    fn create_ray_tracing_pipeline(
        device: &ash::Device,
        ray_tracing_pipeline_fp: &khr::RayTracingPipeline,
        path: &Path,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let shader_stages = [
            vk_create_shader_stage(
                [path.to_str().unwrap(), "raytrace.rgen.spirv"]
                    .iter()
                    .collect::<std::path::PathBuf>(),
                device,
            ),
            vk_create_shader_stage(
                [path.to_str().unwrap(), "raytrace.rmiss.spirv"]
                    .iter()
                    .collect::<std::path::PathBuf>(),
                device,
            ),
            vk_create_shader_stage(
                [path.to_str().unwrap(), "raytrace.rchit.spirv"]
                    .iter()
                    .collect::<std::path::PathBuf>(),
                device,
            ),
        ];

        const RGEN_IDX: u32 = 0;
        const RMISS_IDX: u32 = 1;
        const RCHIT_IDX: u32 = 2;

        let rt_shader_groups_ci = [
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(RGEN_IDX)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(RMISS_IDX)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .closest_hit_shader(RCHIT_IDX)
                .build(),
        ];

        let pipeline_layout = unsafe {
            let pipeline_layout_ci =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .unwrap()
        };

        let pipeline = unsafe {
            let rt_pipeline_ci = vk::RayTracingPipelineCreateInfoKHR::builder()
                .stages(&shader_stages)
                .groups(&rt_shader_groups_ci)
                .max_pipeline_ray_recursion_depth(1)
                .layout(pipeline_layout);
            ray_tracing_pipeline_fp
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&rt_pipeline_ci),
                    None,
                )
                .unwrap()
                .first()
                .copied()
                .unwrap()
        };

        for shader_stage in shader_stages {
            unsafe {
                device.destroy_shader_module(shader_stage.module, None);
            }
        }

        (pipeline_layout, pipeline)
    }

    fn create_shader_binding_table(
        ray_tracing_pipeline_fp: &khr::RayTracingPipeline,
        ray_tracing_pipeline_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        ray_tracing_pipeline: vk::Pipeline,
        buffer_allocator: &mut VkMemoryResourceAllocator,
    ) -> (BufferAllocation, [vk::StridedDeviceAddressRegionKHR; 3]) {
        const RGEN_SHADERS_COUNT: usize = 1;
        const RMISS_SHADERS_COUNT: usize = 1;
        const CHIT_SHADERS_COUNT: usize = 1;

        const GROUP_COUNT: usize = RGEN_SHADERS_COUNT + RMISS_SHADERS_COUNT + CHIT_SHADERS_COUNT;

        let group_handle_size = ray_tracing_pipeline_properties.shader_group_handle_size as usize;
        let group_handle_size_aligned = align_offset(
            ray_tracing_pipeline_properties.shader_group_handle_size as u64,
            ray_tracing_pipeline_properties.shader_group_base_alignment as u64,
        ) as usize;

        let shader_group_handles = unsafe {
            ray_tracing_pipeline_fp
                .get_ray_tracing_shader_group_handles(
                    ray_tracing_pipeline,
                    0,
                    GROUP_COUNT as u32,
                    GROUP_COUNT * group_handle_size,
                )
                .unwrap()
        };

        let buffer_ci = vk::BufferCreateInfo::builder()
            .size((GROUP_COUNT * group_handle_size_aligned) as u64)
            .usage(vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR);
        let mut sbt_buffer = buffer_allocator.allocate_buffer(&buffer_ci, MemoryLocation::CpuToGpu);

        for (group_handle, sbt_group) in std::iter::zip(
            shader_group_handles.chunks_exact(group_handle_size),
            sbt_buffer
                .get_allocation_mut()
                .mapped_slice_mut()
                .unwrap()
                .chunks_exact_mut(group_handle_size_aligned),
        ) {
            sbt_group[..group_handle_size].copy_from_slice(group_handle);
        }

        let sbt_regions = [
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_buffer.get_device_address().unwrap(),
                stride: group_handle_size_aligned as u64,
                size: (group_handle_size_aligned * RGEN_SHADERS_COUNT) as u64,
            },
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_buffer.get_device_address().unwrap()
                    + (RGEN_SHADERS_COUNT * group_handle_size_aligned) as u64,
                stride: group_handle_size_aligned as u64,
                size: (group_handle_size_aligned * RMISS_SHADERS_COUNT) as u64,
            },
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_buffer.get_device_address().unwrap()
                    + ((RGEN_SHADERS_COUNT + RMISS_SHADERS_COUNT) * group_handle_size_aligned)
                        as u64,
                stride: group_handle_size_aligned as u64,
                size: (group_handle_size_aligned * CHIT_SHADERS_COUNT) as u64,
            },
        ];
        (sbt_buffer, sbt_regions)
    }
}

impl Drop for VkRTLightningShadows {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_descriptor_set_allocator_mut()
                .free_descriptor_sets(std::mem::replace(
                    &mut self.descriptor_set_allocation,
                    std::mem::zeroed(),
                ));
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(
                    &mut self.output_image,
                    std::mem::zeroed(),
                ));
            self.device.destroy_image_view(self.output_image_view, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_buffer(std::mem::replace(&mut self.sbt_buffer, std::mem::zeroed()));
        }
    }
}
