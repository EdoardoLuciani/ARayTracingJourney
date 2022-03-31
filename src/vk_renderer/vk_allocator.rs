use super::vk_buffers_suballocator::VkBuffersSubAllocator;
use ash::vk;
use gpu_allocator::{vulkan as vkalloc, MemoryLocation};
use std::cell::RefCell;
use std::rc::Rc;

pub struct VkAllocator {
    pub allocator: Rc<RefCell<VkMemoryResourceAllocator>>,
    pub device_mesh_indices_suballocator: VkBuffersSubAllocator,
}

impl VkAllocator {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
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
            device_mesh_indices_suballocator: mesh_suballocator,
        }
    }
}

pub struct BufferAllocation {
    pub buffer: vk::Buffer,
    pub allocation: vkalloc::Allocation,
    pub device_address: vk::DeviceAddress,
}

pub struct VkMemoryResourceAllocator {
    device: ash::Device,
    allocator: vkalloc::Allocator,
}

impl VkMemoryResourceAllocator {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
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

        let device_address = unsafe {
            let buffer_device_address_info = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
            self.device
                .get_buffer_device_address(&buffer_device_address_info)
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
