#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use super::super::vk_allocator::vk_descriptor_sets_allocator::*;
use super::super::vk_allocator::vk_memory_resource_allocator::*;
use super::super::vk_allocator::VkAllocator;
use super::super::vk_boot::helper::vk_create_shader_stage;
use super::VkImagePrevState;
use ash::vk;
use gpu_allocator::MemoryLocation;
use itertools::Itertools;
use nalgebra::*;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

pub struct VkCombine {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    resolution: vk::Extent2D,
    input_output_color_image: vk::Image,
    input_output_color_image_view: vk::ImageView,
    input_ao_image: vk::Image,
    input_ao_image_view: vk::ImageView,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl VkCombine {
    pub fn new(
        device: Rc<ash::Device>,
        allocator: Rc<RefCell<VkAllocator>>,
        shader_spirv_location: &Path,
        resolution: vk::Extent2D,
        input_output_color_image: vk::Image,
        input_output_color_image_view: vk::ImageView,
        input_ao_image: vk::Image,
        input_ao_image_view: vk::ImageView,
    ) -> Self {
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
            ];
            let descriptor_set_ci =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_bindings);
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

        let mut ret = Self {
            device,
            allocator,
            resolution,
            input_output_color_image,
            input_output_color_image_view,
            input_ao_image,
            input_ao_image_view,
            descriptor_set_layout,
            descriptor_set_allocation,
            pipeline_layout: pipeline_info.0,
            pipeline: pipeline_info.1,
        };
        ret.resize(
            resolution,
            input_output_color_image,
            input_output_color_image_view,
            input_ao_image,
            input_ao_image_view,
        );
        ret
    }

    pub fn resize(
        &mut self,
        resolution: vk::Extent2D,
        input_output_color_image: vk::Image,
        input_output_color_image_view: vk::ImageView,
        input_ao_image: vk::Image,
        input_ao_image_view: vk::ImageView,
    ) {
        self.resolution = resolution;
        self.input_output_color_image = input_output_color_image;
        self.input_output_color_image_view = input_output_color_image_view;
        self.input_ao_image = input_ao_image;
        self.input_ao_image_view = input_ao_image_view;

        self.update_descriptor_set();
    }

    pub fn get_output_image(&self) -> vk::Image {
        self.input_output_color_image
    }

    pub fn get_output_image_view(&self) -> vk::ImageView {
        self.input_output_color_image_view
    }

    pub fn combine(
        &self,
        cb: vk::CommandBuffer,
        input_output_color_prev_state: VkImagePrevState,
        input_ao_prev_state: VkImagePrevState,
    ) {
        unsafe {
            let image_memory_barriers = [
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(input_output_color_prev_state.src_stage)
                    .src_access_mask(input_output_color_prev_state.src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(input_output_color_prev_state.src_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.input_output_color_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(input_ao_prev_state.src_stage)
                    .src_access_mask(input_ao_prev_state.src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .old_layout(input_ao_prev_state.src_layout)
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
            ];
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);

            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set_allocation.get_descriptor_sets()[0]),
                &[],
            );

            self.device
                .cmd_dispatch(cb, self.resolution.width / 8, self.resolution.height / 8, 1);
        }
    }

    fn update_descriptor_set(&self) {
        unsafe {
            let input_color = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(self.input_output_color_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let input_ao = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(self.input_ao_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

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
            format!("{}//{}", path.to_str().unwrap(), "combine.comp.spirv"),
            device,
        );

        let pipeline_layout = unsafe {
            let pipeline_layout_ci =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(set_layouts);
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

impl Drop for VkCombine {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            let mut al = self.allocator.as_ref().borrow_mut();
            al.get_descriptor_set_allocator_mut()
                .free_descriptor_sets(std::mem::replace(
                    &mut self.descriptor_set_allocation,
                    DescriptorSetAllocation::null(),
                ));
        }
    }
}
