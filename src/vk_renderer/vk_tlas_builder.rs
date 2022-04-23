use crate::vk_renderer::vk_allocator::{BufferAllocation, ImageAllocation, VkAllocator};
use ash::vk::{AccelerationStructureReferenceKHR, Packed24_8, TransformMatrixKHR};
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use nalgebra::*;
use std::cell::RefCell;
use std::rc::Rc;

trait PostSubmissionCleanup {
    fn cleanup(self, allocator: Rc<RefCell<VkAllocator>>);
}

struct TlasBuild {
    scratch_buffer: BufferAllocation
}
impl PostSubmissionCleanup for TlasBuild {
    fn cleanup(self, allocator: Rc<RefCell<VkAllocator>>) {
        allocator.as_ref().borrow_mut().get_allocator_mut().destroy_buffer(self.scratch_buffer);
    }
}

struct VkTlasBuilder<'a> {
    acceleration_structure_fp: &'a khr::AccelerationStructure,
    allocator: Rc<RefCell<VkAllocator<'a>>>,
    post_cb_submit_cleanup: Option<Box<dyn PostSubmissionCleanup>>
}

impl<'a> VkTlasBuilder<'a> {
    pub fn new(
        acceleration_structure_fp: &'a khr::AccelerationStructure,
        allocator: Rc<RefCell<VkAllocator<'a>>>,
    ) -> Self {
        Self {
            acceleration_structure_fp,
            allocator,
            post_cb_submit_cleanup: None
        }
    }

    pub fn create_tlas(&mut self, cb: vk::CommandBuffer, blases: &[vk::AccelerationStructureKHR]) ->  {
        // identity matrix
        let transform_matrix = TransformMatrixKHR {
            matrix: [
                1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                1.0f32, 0.0f32,
            ],
        };

        let as_instances = blases
            .iter()
            .copied()
            .map(|blas| {
                let blas_address = unsafe {
                    let as_device_address_info =
                        vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                            .acceleration_structure(blas);
                    self.acceleration_structure_fp
                        .get_acceleration_structure_device_address(&as_device_address_info)
                };

                vk::AccelerationStructureInstanceKHR {
                    transform: transform_matrix,
                    instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
                    instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                        0,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                    ),
                    acceleration_structure_reference: AccelerationStructureReferenceKHR {
                        device_handle: blas_address,
                    },
                }
            })
            .collect::<Vec<_>>();

        let as_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(blases.len() as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        let geometry_instances = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                host_address: as_instances.as_ptr() as *const std::ffi::c_void,
            })
            .build();
        let geometry_data = vk::AccelerationStructureGeometryDataKHR {
            instances: geometry_instances,
        };
        let as_geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(geometry_data)
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let mut as_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
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

        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(as_size_info.acceleration_structure_size)
            .usage(
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            );
        let device_tlas_buffer = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::GpuOnly);

        let tlas = unsafe {
            let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(device_tlas_buffer.get_buffer())
                .offset(0)
                .size(as_size_info.build_scratch_size)
                .ty(as_build_info.ty);
            self.acceleration_structure_fp
                .create_acceleration_structure(&as_create_info, None)
                .unwrap()
        };

        as_build_info.dst_acceleration_structure = tlas;

        buffer_create_info.size = as_size_info.build_scratch_size;
        buffer_create_info.usage =
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let scratch_buffer = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::GpuOnly);

        unsafe {
            self.acceleration_structure_fp
                .cmd_build_acceleration_structures(
                    cb,
                    std::slice::from_ref(&as_build_info),
                    &[std::slice::from_ref(&as_range_info)],
                )
        }
        self.post_cb_submit_cleanup = Some(Box::new(TlasBuild {
            scratch_buffer
        }))
    }

    pub fn post_submit_cleanup(&mut self) {
        if let Some(cleanup_box) = self.post_cb_submit_cleanup.take() {
            cleanup_box.cleanup(self.allocator.clone())
        }
    }
}
