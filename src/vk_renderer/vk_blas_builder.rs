use super::vk_allocator::VkAllocator;
use crate::vk_renderer::vk_allocator::vk_memory_resource_allocator::BufferAllocation;
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use std::cell::RefCell;
use std::rc::Rc;

pub struct VkBlasBuilder {
    device: Rc<ash::Device>,
    acceleration_structure_fp: Rc<khr::AccelerationStructure>,
    allocator: Rc<RefCell<VkAllocator>>,
}

pub struct Blas {
    blas: vk::AccelerationStructureKHR,
    device_blas_allocation: BufferAllocation,
    scratch_buffer: Option<BufferAllocation>,
    allocator: Rc<RefCell<VkAllocator>>,
    acceleration_structure_fp: Rc<khr::AccelerationStructure>,
}

impl Blas {
    pub fn post_submit_cleanup(&mut self) {
        if let Some(buffer) = self.scratch_buffer.take() {
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_buffer(buffer);
        }
    }

    pub fn release_scratch_buffer(&mut self) -> BufferAllocation {
        self.scratch_buffer
            .take()
            .expect("Scratch buffer has already been released")
    }

    pub fn get_blas(&self) -> vk::AccelerationStructureKHR {
        self.blas
    }

    pub fn get_blas_address(&self) -> vk::DeviceAddress {
        let info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
            .acceleration_structure(self.blas);
        unsafe {
            self.acceleration_structure_fp
                .get_acceleration_structure_device_address(&info)
        }
    }

    pub fn get_blas_allocation(&self) -> &BufferAllocation {
        &self.device_blas_allocation
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        self.post_submit_cleanup();
        self.allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .destroy_buffer(std::mem::replace(
                &mut self.device_blas_allocation,
                unsafe { std::mem::zeroed() },
            ));
        unsafe {
            self.acceleration_structure_fp
                .destroy_acceleration_structure(self.blas, None);
        }
    }
}

impl VkBlasBuilder {
    pub fn new(
        device: Rc<ash::Device>,
        acceleration_structure_fp: Rc<khr::AccelerationStructure>,
        allocator: Rc<RefCell<VkAllocator>>,
    ) -> Self {
        Self {
            device,
            acceleration_structure_fp,
            allocator,
        }
    }

    pub fn build_blas_from_geometry(
        &self,
        cb: vk::CommandBuffer,
        as_geom_infos: &[vk::AccelerationStructureGeometryKHR],
        as_build_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        as_flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> Blas {
        assert_eq!(as_geom_infos.len(), as_build_ranges.len());

        let mut as_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(as_flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geom_infos)
            .build();

        let as_size_info = unsafe {
            self.acceleration_structure_fp
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &as_build_info,
                    &as_build_ranges
                        .iter()
                        .map(|e| e.primitive_count)
                        .collect::<Vec<_>>(),
                )
        };

        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(as_size_info.acceleration_structure_size)
            .usage(
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            );
        let device_blas_buffer = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::GpuOnly);

        as_build_info.dst_acceleration_structure = unsafe {
            let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(device_blas_buffer.get_buffer())
                .offset(0)
                .size(as_size_info.acceleration_structure_size)
                .ty(as_build_info.ty);
            self.acceleration_structure_fp
                .create_acceleration_structure(&as_create_info, None)
                .unwrap()
        };

        buffer_create_info.size = as_size_info.build_scratch_size;
        buffer_create_info.usage =
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let scratch_buffer = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::GpuOnly);
        as_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.get_device_address().unwrap(),
        };

        unsafe {
            self.acceleration_structure_fp
                .cmd_build_acceleration_structures(
                    cb,
                    std::slice::from_ref(&as_build_info),
                    &[&as_build_ranges],
                );
        };

        Blas {
            blas: as_build_info.dst_acceleration_structure,
            device_blas_allocation: device_blas_buffer,
            scratch_buffer: Some(scratch_buffer),
            allocator: self.allocator.clone(),
            acceleration_structure_fp: self.acceleration_structure_fp.clone(),
        }
    }
}
