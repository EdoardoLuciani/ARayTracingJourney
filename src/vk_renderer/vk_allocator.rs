use super::vk_buffers_suballocator::VkBuffersSubAllocator;
use ash::vk;
use gpu_allocator::{vulkan as vkalloc, MemoryLocation};
use std::cell::RefCell;
use std::rc::Rc;

pub struct VkAllocator<'a> {
    allocator: Rc<RefCell<VkMemoryResourceAllocator<'a>>>,
    host_uniforms_sub_allocator: VkBuffersSubAllocator<'a>,
    device_uniforms_sub_allocator: VkBuffersSubAllocator<'a>,
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
        VkAllocator {
            allocator,
            host_uniforms_sub_allocator,
            device_uniforms_sub_allocator,
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

    pub fn allocate_image(
        &mut self,
        image_create_info: &vk::ImageCreateInfo,
        memory_location: MemoryLocation,
    ) -> ImageAllocation {
        let image = unsafe { self.device.create_image(image_create_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

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
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap()
        };

        ImageAllocation {
            image,
            allocation,
            device_address: None,
        }
    }

    pub fn destroy_image(&mut self, image: ImageAllocation) {
        self.allocator.free(image.allocation).unwrap();
        unsafe { self.device.destroy_image(image.image, None) };
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

    pub fn get_allocation_mut(&mut self) -> &mut vkalloc::Allocation {
        &mut self.allocation
    }

    pub fn get_device_address(&self) -> Option<vk::DeviceAddress> {
        self.device_address
    }
}

pub struct ImageAllocation {
    image: vk::Image,
    allocation: vkalloc::Allocation,
    device_address: Option<vk::DeviceAddress>,
}

impl ImageAllocation {
    pub fn get_image(&self) -> vk::Image {
        self.image
    }

    pub fn get_allocation(&self) -> &vkalloc::Allocation {
        &self.allocation
    }

    pub fn get_device_address(&self) -> Option<vk::DeviceAddress> {
        self.device_address
    }
}
