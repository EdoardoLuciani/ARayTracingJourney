use super::lights::{Lights, SpotLight};
use super::vk_allocator::VkAllocator;
use crate::vk_renderer::lights::LightShaderData;
use crate::vk_renderer::vk_allocator::vk_buffers_suballocator::SubAllocationData;
use crate::vk_renderer::vk_allocator::vk_descriptor_sets_allocator::DescriptorSetAllocation;
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub struct VkLights {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    lights: Lights,
    host_suballocation: SubAllocationData,
    device_suballocation: SubAllocationData,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,
    needs_update: bool,
}

impl VkLights {
    pub fn new(device: Rc<ash::Device>, allocator: Rc<RefCell<VkAllocator>>) -> Self {
        let host_suballocation = allocator
            .as_ref()
            .borrow_mut()
            .get_host_uniform_sub_allocator_mut()
            .allocate(1024, 128);
        let device_suballocation = allocator
            .as_ref()
            .borrow_mut()
            .get_device_uniform_sub_allocator_mut()
            .allocate(1024, 128);

        let descriptor_set_layout = unsafe {
            let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR);
            let descriptor_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(std::slice::from_ref(&descriptor_set_layout_binding));
            device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None)
                .unwrap()
        };
        let descriptor_set_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_descriptor_set_allocator_mut()
            .allocate_descriptor_sets(std::slice::from_ref(&descriptor_set_layout));

        Self {
            device,
            allocator,
            lights: Lights::default(),
            host_suballocation,
            device_suballocation,
            descriptor_set_layout,
            descriptor_set_allocation,
            needs_update: true,
        }
    }

    pub fn lights(&self) -> &Lights {
        &self.lights
    }

    pub fn lights_mut(&mut self) -> &mut Lights {
        self.needs_update = true;
        &mut self.lights
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set_allocation.get_descriptor_sets()[0]
    }

    pub fn update_host_and_device_buffer(&mut self, cb: vk::CommandBuffer) {
        if self.needs_update {
            let required_buffer_size =
                self.lights.get_lights_count() * std::mem::size_of::<LightShaderData>();
            if required_buffer_size > self.host_suballocation.get_size() {
                self.recreate_buffers(required_buffer_size);
            }

            self.update_descriptor_set(
                (self.lights.get_lights_count() * std::mem::size_of::<LightShaderData>()) as u64,
            );

            let host_shader_data_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    self.host_suballocation.get_host_ptr().unwrap().as_ptr()
                        as *mut LightShaderData,
                    self.lights.get_lights_count(),
                )
            };
            self.lights.copy_lights_shader_data(host_shader_data_slice);

            unsafe {
                let buffer_memory_barrier = vk::BufferMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .buffer(self.device_suballocation.get_buffer())
                    .offset(self.device_suballocation.get_buffer_offset() as u64)
                    .size(self.device_suballocation.get_size() as u64);
                let dependancy_info = vk::DependencyInfo::builder()
                    .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier));
                self.device.cmd_pipeline_barrier2(cb, &dependancy_info);

                let copy_regions = vk::BufferCopy2::builder()
                    .src_offset(self.host_suballocation.get_buffer_offset() as u64)
                    .dst_offset(self.device_suballocation.get_buffer_offset() as u64)
                    .size(self.host_suballocation.get_size() as u64);
                let copy_buffer_info = vk::CopyBufferInfo2::builder()
                    .src_buffer(self.host_suballocation.get_buffer())
                    .dst_buffer(self.device_suballocation.get_buffer())
                    .regions(std::slice::from_ref(&copy_regions));
                self.device.cmd_copy_buffer2(cb, &copy_buffer_info);

                let buffer_memory_barrier = vk::BufferMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(self.device_suballocation.get_buffer())
                    .offset(self.device_suballocation.get_buffer_offset() as u64)
                    .size(self.device_suballocation.get_size() as u64);
                let dependancy_info = vk::DependencyInfo::builder()
                    .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier));
                self.device.cmd_pipeline_barrier2(cb, &dependancy_info);
            }
            self.needs_update = false;
        }
    }

    fn recreate_buffers(&mut self, new_size: usize) {
        let mut al = self.allocator.as_ref().borrow_mut();
        take_mut::take(&mut self.host_suballocation, |allocation| {
            al.get_host_uniform_sub_allocator_mut().free(allocation);
            al.get_host_uniform_sub_allocator_mut()
                .allocate(new_size, 128)
        });
        take_mut::take(&mut self.device_suballocation, |allocation| {
            al.get_device_uniform_sub_allocator_mut().free(allocation);
            al.get_device_uniform_sub_allocator_mut()
                .allocate(new_size, 128)
        });
    }

    fn update_descriptor_set(&self, lights_byte_size: u64) {
        unsafe {
            let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.device_suballocation.get_buffer())
                .offset(self.device_suballocation.get_buffer_offset() as u64)
                .range(lights_byte_size);
            let descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set())
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&descriptor_buffer_info));
            self.device
                .update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
        }
    }
}

impl Drop for VkLights {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_host_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.host_suballocation,
                    std::mem::zeroed(),
                ));
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_device_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.device_suballocation,
                    std::mem::zeroed(),
                ));
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
