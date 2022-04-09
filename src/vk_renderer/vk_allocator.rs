use super::vk_buffers_suballocator::VkBuffersSubAllocator;
use ash::vk;
use gpu_allocator::{vulkan as vkalloc, MemoryLocation};
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

pub struct VkAllocator<'a> {
    allocator: Rc<RefCell<VkMemoryResourceAllocator<'a>>>,
    device_mesh_indices_sub_allocator: VkBuffersSubAllocator<'a>,
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
        let mesh_suballocator = VkBuffersSubAllocator::new(
            allocator.clone(),
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::GpuOnly,
            100_000_000,
            512,
        );
        VkAllocator {
            allocator,
            device_mesh_indices_sub_allocator: mesh_suballocator,
        }
    }

    pub fn get_allocator_mut(&mut self) -> &mut VkMemoryResourceAllocator {
        todo!()
    }

    pub fn get_device_mesh_indices_sub_allocator_mut(&mut self) -> &mut VkBuffersSubAllocator<'a> {
        &mut self.device_mesh_indices_sub_allocator
    }
}

pub struct VkMemoryResourceAllocator<'a> {
    device: &'a ash::Device,
    allocator: vkalloc::Allocator,
}

impl<'a> VkMemoryResourceAllocator<'a> {
    pub fn new(
        instance: ash::Instance,
        device: &'a ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance,
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
            })
            .expect("Could not create Allocator");

        VkMemoryResourceAllocator { device, allocator }
    }

    pub fn allocate_buffer(
        &mut self,
        buffer_create_info: &vk::BufferCreateInfo,
        memory_location: MemoryLocation,
    ) -> BufferAllocation {
        let buffer = unsafe { self.device.create_buffer(buffer_create_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .allocate(&vkalloc::AllocationCreateDesc {
                name: "",
                requirements,
                location: memory_location,
                linear: true, // buffers are always linear
            })
            .unwrap();

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap()
        };

        let device_address = match buffer_create_info
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        {
            true => {
                let buffer_device_address_info =
                    vk::BufferDeviceAddressInfo::builder().buffer(buffer);
                Some(unsafe {
                    self.device
                        .get_buffer_device_address(&buffer_device_address_info)
                })
            }
            false => None,
        };

        BufferAllocation {
            buffer,
            allocation,
            device_address,
        }
    }

    pub fn destroy_buffer(&mut self, buffer: BufferAllocation) {
        self.allocator.free(buffer.allocation).unwrap();
        unsafe { self.device.destroy_buffer(buffer.buffer, None) };
    }
}

pub struct BufferAllocation {
    buffer: vk::Buffer,
    allocation: vkalloc::Allocation,
    device_address: Option<vk::DeviceAddress>,
}

impl BufferAllocation {
    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_allocation(&self) -> &vkalloc::Allocation {
        &self.allocation
    }

    pub fn get_device_address(&self) -> Option<vk::DeviceAddress> {
        self.device_address
    }
}
