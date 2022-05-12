pub mod vk_buffers_suballocator;
pub mod vk_descriptor_sets_allocator;
pub mod vk_memory_resource_allocator;

use ash::vk;
use gpu_allocator::{vulkan as vkalloc, MemoryLocation};
use std::cell::RefCell;
use std::rc::Rc;
use vk_buffers_suballocator::VkBuffersSubAllocator;
use vk_descriptor_sets_allocator::VkDescriptorSetsAllocator;
use vk_memory_resource_allocator::VkMemoryResourceAllocator;

pub struct VkAllocator<'a> {
    allocator: Rc<RefCell<VkMemoryResourceAllocator<'a>>>,
    host_uniforms_sub_allocator: VkBuffersSubAllocator<'a>,
    device_uniforms_sub_allocator: VkBuffersSubAllocator<'a>,
    descriptor_sets_allocator: VkDescriptorSetsAllocator<'a>,
}

impl<'a> VkAllocator<'a> {
    pub fn new(
        instance: ash::Instance,
        device: &'a ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let allocator = Rc::new(RefCell::new(VkMemoryResourceAllocator::new(
            instance,
            device,
            physical_device,
        )));

        let host_uniforms_sub_allocator = VkBuffersSubAllocator::new(
            allocator.clone(),
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            524_288,
            256,
        );

        let device_uniforms_sub_allocator = VkBuffersSubAllocator::new(
            allocator.clone(),
            vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
            524_288,
            256,
        );

        let descriptor_pool_sizes = vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 10,
        }];
        let descriptor_sets_allocator = VkDescriptorSetsAllocator::new(
            device,
            vk::DescriptorPoolCreateFlags::empty(),
            1000,
            descriptor_pool_sizes,
        );

        VkAllocator {
            allocator,
            host_uniforms_sub_allocator,
            device_uniforms_sub_allocator,
            descriptor_sets_allocator,
        }
    }

    pub fn get_allocator_mut(&mut self) -> std::cell::RefMut<VkMemoryResourceAllocator<'a>> {
        self.allocator.as_ref().borrow_mut()
    }

    pub fn get_host_uniform_sub_allocator_mut(&mut self) -> &mut VkBuffersSubAllocator<'a> {
        &mut self.host_uniforms_sub_allocator
    }

    pub fn get_device_uniform_sub_allocator_mut(&mut self) -> &mut VkBuffersSubAllocator<'a> {
        &mut self.device_uniforms_sub_allocator
    }
    
    pub fn get_descriptor_set_allocator_mut(&mut self) -> &mut VkDescriptorSetsAllocator<'a> {
        &mut self.descriptor_sets_allocator
    }
}
