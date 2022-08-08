use super::super::vk_allocator::vk_descriptor_sets_allocator::*;
use super::super::vk_allocator::VkAllocator;
use super::super::vk_boot::helper::vk_create_shader_stage;
use ash::vk;
use itertools::Itertools;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

pub struct VkTonemap {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    presentation_resolution: vk::Extent2D,
    input_color_image: vk::Image,
    input_color_image_view: vk::ImageView,
    input_ao_image: vk::Image,
    input_ao_image_view: vk::ImageView,
    output_images: Vec<vk::Image>,
    output_image_views: Vec<vk::ImageView>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl VkTonemap {
    pub fn new(
        device: Rc<ash::Device>,
        allocator: Rc<RefCell<VkAllocator>>,
        presentation_resolution: vk::Extent2D,
        shader_spirv_location: &Path,
        input_color_image: vk::Image,
        input_color_image_view: vk::ImageView,
        input_ao_image: vk::Image,
        input_ao_image_view: vk::ImageView,
        output_images: Vec<vk::Image>,
        output_image_views: Vec<vk::ImageView>,
    ) -> Self {
        assert_eq!(output_images.len(), output_image_views.len());

        let descriptor_set_layout = unsafe {
            let descriptor_set_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(output_images.len() as u32)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];
            let flags = [
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            ];
            let mut descriptor_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);
            let descriptor_set_ci = vk::DescriptorSetLayoutCreateInfo::builder()
                .push_next(&mut descriptor_flags)
                .bindings(&descriptor_set_bindings);
            device
                .create_descriptor_set_layout(&descriptor_set_ci, None)
                .unwrap()
        };
        let descriptor_set_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_descriptor_set_allocator_mut()
            .allocate_descriptor_sets(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_info =
            Self::create_pipeline(&device, shader_spirv_location, &[descriptor_set_layout]);
        let ret = Self {
            device,
            allocator,
            presentation_resolution,
            input_color_image,
            input_color_image_view,
            input_ao_image,
            input_ao_image_view,
            output_images,
            output_image_views,
            descriptor_set_layout,
            descriptor_set_allocation,
            pipeline_layout: pipeline_info.0,
            pipeline: pipeline_info.1,
        };
        ret.update_descriptor_set();
        ret
    }

    pub fn resize(
        &mut self,
        presentation_resolution: vk::Extent2D,
        input_color_image: vk::Image,
        input_color_image_view: vk::ImageView,
        input_ao_image: vk::Image,
        input_ao_image_view: vk::ImageView,
        output_images: Vec<vk::Image>,
        output_image_views: Vec<vk::ImageView>,
    ) {
        self.presentation_resolution = presentation_resolution;
        self.input_color_image = input_color_image;
        self.input_color_image_view = input_color_image_view;
        self.input_ao_image = input_ao_image;
        self.input_ao_image_view = input_ao_image_view;
        self.output_images = output_images;
        self.output_image_views = output_image_views;
        self.update_descriptor_set();
    }

    pub fn present(&self, cb: vk::CommandBuffer, dst_image_idx: u32) {
        unsafe {
            let image_memory_barriers = [
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ_KHR)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.input_color_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ_KHR)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.input_ao_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.output_images[dst_image_idx as usize])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build(),
            ];
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);

            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            self.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &dst_image_idx.to_ne_bytes(),
            );

            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set_allocation.get_descriptor_sets()[0]),
                &[],
            );

            self.device.cmd_dispatch(
                cb,
                self.presentation_resolution.width / 8,
                self.presentation_resolution.height / 8,
                1,
            );
        }
    }

    fn update_descriptor_set(&self) {
        unsafe {
            let input_color = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(self.input_color_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let input_ao = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(self.input_ao_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let output_images = self
                .output_image_views
                .iter()
                .copied()
                .map(|image_view| {
                    vk::DescriptorImageInfo::builder()
                        .sampler(vk::Sampler::null())
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()
                })
                .collect_vec();

            let descriptor_set_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&input_color))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&input_ao))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&output_images)
                    .build(),
            ];
            self.device
                .update_descriptor_sets(&descriptor_set_writes, &[]);
        }
    }

    fn create_pipeline(
        device: &ash::Device,
        path: &Path,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let shader_stage = vk_create_shader_stage(
            format!("{}//{}", path.to_str().unwrap(), "present.comp.spirv"),
            device,
        );

        let pipeline_layout = unsafe {
            let push_constant = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(4);
            let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push_constant));
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .unwrap()
        };

        let pipeline_ci = vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_ci),
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(shader_stage.module, None);
        }

        (pipeline_layout, pipeline)
    }
}

impl Drop for VkTonemap {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_descriptor_set_allocator_mut()
                .free_descriptor_sets(std::mem::replace(
                    &mut self.descriptor_set_allocation,
                    DescriptorSetAllocation::null(),
                ));
        }
    }
}
