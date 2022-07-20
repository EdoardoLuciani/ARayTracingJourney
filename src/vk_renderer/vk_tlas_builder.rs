use super::vk_allocator::vk_memory_resource_allocator::*;
use super::vk_allocator::VkAllocator;
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use std::cell::RefCell;
use std::ops::DerefMut;
use std::rc::Rc;

pub struct VkTlasBuilder {
    device: Rc<ash::Device>,
    acceleration_structure_fp: Rc<khr::AccelerationStructure>,
    allocator: Rc<RefCell<VkAllocator>>,
    host_as_instance_struct_buffer: Option<BufferAllocation>,
    device_as_instance_struct_buffer: Option<BufferAllocation>,
    scratch_buffer: Option<BufferAllocation>,
    tlas_buffer: Option<BufferAllocation>,
    tlas: vk::AccelerationStructureKHR,
}

impl VkTlasBuilder {
    pub fn new(
        device: Rc<ash::Device>,
        acceleration_structure_fp: Rc<khr::AccelerationStructure>,
        allocator: Rc<RefCell<VkAllocator>>,
    ) -> Self {
        Self {
            device,
            acceleration_structure_fp,
            allocator,
            host_as_instance_struct_buffer: None,
            device_as_instance_struct_buffer: None,
            scratch_buffer: None,
            tlas_buffer: None,
            tlas: vk::AccelerationStructureKHR::null(),
        }
    }

    pub fn recreate_tlas(
        &mut self,
        cb: vk::CommandBuffer,
        acceleration_structure_references: &[vk::AccelerationStructureInstanceKHR],
    ) -> vk::AccelerationStructureKHR {
        // delete and recreate tlas as it is more recommended than update
        unsafe {
            self.acceleration_structure_fp
                .destroy_acceleration_structure(self.tlas, None);
        }

        let mut allocator = self.allocator.as_ref().borrow_mut();

        Self::update_buffer(
            allocator.deref_mut(),
            &mut self.host_as_instance_struct_buffer,
            (acceleration_structure_references.len()
                * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                acceleration_structure_references.as_ptr(),
                self.host_as_instance_struct_buffer
                    .as_ref()
                    .unwrap()
                    .get_allocation()
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut vk::AccelerationStructureInstanceKHR,
                acceleration_structure_references.len(),
            )
        }

        Self::update_buffer(
            allocator.deref_mut(),
            &mut self.device_as_instance_struct_buffer,
            (acceleration_structure_references.len()
                * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as u64,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
        );

        // host to device buffer copy for the struct information
        unsafe {
            let region = vk::BufferCopy2::builder().src_offset(0).dst_offset(0).size(
                (acceleration_structure_references.len()
                    * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>())
                    as u64,
            );
            let copy_buffer_info = vk::CopyBufferInfo2::builder()
                .src_buffer(
                    self.host_as_instance_struct_buffer
                        .as_ref()
                        .unwrap()
                        .get_buffer(),
                )
                .dst_buffer(
                    self.device_as_instance_struct_buffer
                        .as_ref()
                        .unwrap()
                        .get_buffer(),
                )
                .regions(std::slice::from_ref(&region));
            self.device.cmd_copy_buffer2(cb, &copy_buffer_info);
        }

        let as_geometry = {
            let geometry_instances = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: self
                        .device_as_instance_struct_buffer
                        .as_ref()
                        .unwrap()
                        .get_device_address()
                        .unwrap(),
                })
                .build();
            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                instances: geometry_instances,
            };
            vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(geometry_data)
                .build()
        };

        let as_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(acceleration_structure_references.len() as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        let mut as_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&as_geometry));

        let as_size_info = unsafe {
            self.acceleration_structure_fp
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &as_build_info,
                    std::slice::from_ref(&as_range_info.primitive_count),
                )
        };

        Self::update_buffer(
            allocator.deref_mut(),
            &mut self.tlas_buffer,
            as_size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        );

        as_build_info.dst_acceleration_structure = unsafe {
            let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(self.tlas_buffer.as_ref().unwrap().get_buffer())
                .offset(0)
                .size(as_size_info.acceleration_structure_size)
                .ty(as_build_info.ty);
            self.acceleration_structure_fp
                .create_acceleration_structure(&as_create_info, None)
                .unwrap()
        };

        Self::update_buffer(
            allocator.deref_mut(),
            &mut self.scratch_buffer,
            as_size_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        );

        as_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: self
                .scratch_buffer
                .as_ref()
                .unwrap()
                .get_device_address()
                .unwrap(),
        };

        unsafe {
            let buffer_memory_barrier2 = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(
                    self.device_as_instance_struct_buffer
                        .as_ref()
                        .unwrap()
                        .get_buffer(),
                )
                .offset(0)
                .size(vk::WHOLE_SIZE);
            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier2));
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);

            self.acceleration_structure_fp
                .cmd_build_acceleration_structures(
                    cb,
                    std::slice::from_ref(&as_build_info),
                    &[std::slice::from_ref(&as_range_info)],
                )
        }
        self.tlas = as_build_info.dst_acceleration_structure;
        self.tlas
    }

    pub fn get_tlas(&self) -> Option<vk::AccelerationStructureKHR> {
        if self.tlas == vk::AccelerationStructureKHR::null() {
            None
        } else {
            Some(self.tlas)
        }
    }

    pub fn get_tlas_buffer(&self) -> Option<vk::Buffer> {
        self.tlas_buffer.as_ref().map(|b| b.get_buffer())
    }

    fn update_buffer(
        allocator: &mut VkAllocator,
        buffer: &mut Option<BufferAllocation>,
        buffer_size: u64,
        buffer_usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
    ) {
        if buffer.is_none() || buffer.as_ref().unwrap().get_allocation().size() < buffer_size {
            if let Some(old_scratch_buffer) = buffer.take() {
                allocator
                    .get_allocator_mut()
                    .destroy_buffer(old_scratch_buffer);
            }

            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(buffer_usage);
            let new_buffer = allocator
                .get_allocator_mut()
                .allocate_buffer(&buffer_create_info, memory_location);
            *buffer = Some(new_buffer)
        }
    }
}

impl Drop for VkTlasBuilder {
    fn drop(&mut self) {
        unsafe {
            self.acceleration_structure_fp
                .destroy_acceleration_structure(self.tlas, None);

            let mut allocator = self.allocator.as_ref().borrow_mut();
            if let Some(buffer) = self.tlas_buffer.take() {
                allocator.get_allocator_mut().destroy_buffer(buffer);
            }
            if let Some(buffer) = self.scratch_buffer.take() {
                allocator.get_allocator_mut().destroy_buffer(buffer);
            }
            if let Some(buffer) = self.host_as_instance_struct_buffer.take() {
                allocator.get_allocator_mut().destroy_buffer(buffer);
            }
            if let Some(buffer) = self.device_as_instance_struct_buffer.take() {
                allocator.get_allocator_mut().destroy_buffer(buffer);
            }
        }
    }
}

mod tests {
    use super::*;
    use crate::vk_renderer::vk_boot::vk_base::VkBase;
    use crate::vk_renderer::vk_model::VkModel;
    use nalgebra::*;
    use std::path::PathBuf;

    #[test]
    fn tlas_build() {
        let mut vulkan_ray_tracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);
        let mut vulkan_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true);
        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);
        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::builder().synchronization2(true);
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut vulkan_13_features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_acceleration_structure)
            .push_next(&mut vulkan_ray_tracing_pipeline);
        let bvk = VkBase::new(
            "",
            &[],
            &[
                "VK_KHR_acceleration_structure",
                "VK_KHR_deferred_host_operations",
            ],
            &physical_device_features2,
            &[(vk::QueueFlags::GRAPHICS, 1.0f32)],
            None,
        );
        let device = Rc::new(bvk.get_device().clone());
        let acceleration_structure_fp = Rc::new(khr::AccelerationStructure::new(
            bvk.get_instance(),
            bvk.get_device(),
        ));

        let allocator = Rc::new(RefCell::new(VkAllocator::new(
            bvk.get_instance().clone(),
            device.clone(),
            bvk.get_physical_device().clone(),
        )));

        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(bvk.get_queue_family_index());
        let command_pool = unsafe {
            bvk.get_device()
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };
        let command_buffer_create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            bvk.get_device()
                .allocate_command_buffers(&command_buffer_create_info)
                .unwrap()[0]
        };
        unsafe {
            let begin_command_buffer = vk::CommandBufferBeginInfo::default();
            bvk.get_device()
                .begin_command_buffer(command_buffer, &begin_command_buffer)
                .unwrap();
        };

        let mut water_bottle = VkModel::new(
            device.clone(),
            Some(acceleration_structure_fp.clone()),
            allocator.clone(),
            PathBuf::from("assets/models/WaterBottle.glb"),
            Matrix4::<f32>::new_translation(&Vector3::<f32>::from_element(0.0f32)).remove_row(3),
        );
        water_bottle.update_model_status(&Vector3::from_element(0.0f32), command_buffer);

        let mut tlas_builder = VkTlasBuilder::new(
            device.clone(),
            acceleration_structure_fp.clone(),
            allocator.clone(),
        );

        unsafe {
            let buffer_memory_barrier2 = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
                .buffer(water_bottle.get_blases_buffers().unwrap())
                .offset(0)
                .size(vk::WHOLE_SIZE);
            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier2));
            bvk.get_device()
                .cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        tlas_builder.recreate_tlas(
            command_buffer,
            std::slice::from_ref(&water_bottle.get_acceleration_structure_instances().unwrap()),
        );

        unsafe {
            bvk.get_device().end_command_buffer(command_buffer).unwrap();
        }

        let command_buffer_submit_info =
            vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);
        let queue_submit2 = vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&command_buffer_submit_info));
        unsafe {
            bvk.get_device()
                .queue_submit2(
                    *bvk.get_queues().first().unwrap(),
                    std::slice::from_ref(&queue_submit2),
                    vk::Fence::null(),
                )
                .unwrap();
            bvk.get_device().device_wait_idle().unwrap();
            bvk.get_device()
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        water_bottle.reset_command_buffer_submission_status();
        unsafe {
            bvk.get_device()
                .free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));
            bvk.get_device().destroy_command_pool(command_pool, None);
        }
    }
}
