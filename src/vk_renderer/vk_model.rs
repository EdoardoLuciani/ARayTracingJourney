use std::any::Any;
use std::boxed::Box;
use std::cell::RefCell;
use std::rc::Rc;

use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use nalgebra::*;

use crate::vk_renderer::model_reader::model_reader::{
    align_offset, ModelCopyInfo, ModelReader, PrimitiveCopyInfo, Sphere,
};
use crate::vk_renderer::vk_allocator::{BufferAllocation, ImageAllocation, VkAllocator};
use crate::vk_renderer::vk_buffers_suballocator::SubAllocationData;
use crate::{GltfModelReader, MeshAttributeType, TextureType};

// Trait for managing the state of the model from disk <-> host <-> device
trait VkModelTransferLocation {
    fn to_disk(self: Box<Self>, _vk_model: &mut VkModel) {}
    fn to_host(self: Box<Self>, _vk_model: &mut VkModel, cb: vk::CommandBuffer) {}
    fn to_device(self: Box<Self>, _vk_model: &mut VkModel, cb: vk::CommandBuffer) {}
    fn as_any(&self) -> &dyn Any;
}

struct Storage;
impl VkModelTransferLocation for Storage {
    fn to_host(self: Box<Storage>, vk_model: &mut VkModel, cb: vk::CommandBuffer) {
        let (buffer_allocation, model_copy_info) = vk_model.transfer_from_disk_to_host();
        vk_model.state = Some(Box::new(Host {
            host_buffer_allocation: buffer_allocation,
            host_model_copy_info: model_copy_info,
        }));
    }

    fn to_device(self: Box<Storage>, vk_model: &mut VkModel, cb: vk::CommandBuffer) {
        self.to_host(vk_model, cb);
        vk_model.state.take().unwrap().to_device(vk_model, cb);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct Host {
    host_buffer_allocation: BufferAllocation,
    host_model_copy_info: ModelCopyInfo,
}

impl VkModelTransferLocation for Host {
    fn to_disk(self: Box<Host>, vk_model: &mut VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .destroy_buffer(self.host_buffer_allocation);
        vk_model.state = Some(Box::new(Storage {}));
    }

    fn to_device(self: Box<Host>, vk_model: &mut VkModel, cb: vk::CommandBuffer) {
        let (device_mesh_indices_buffer, primitives_info) = vk_model.transfer_from_host_to_device(
            cb,
            self.host_buffer_allocation,
            self.host_model_copy_info,
        );

        let host_uniform_sub_allocation = vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_host_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<VkModelUniform>(), 1);

        let device_uniform_sub_allocation = vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_device_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<VkModelUniform>(), 1);

        vk_model.copy_uniform(
            &host_uniform_sub_allocation,
            cb,
            &device_uniform_sub_allocation,
        );

        let (acceleration_structure, device_acceleration_structure_buffer) =
            match vk_model.acceleration_structure_fp.is_some() {
                true => {
                    let res = vk_model.create_blas(
                        cb,
                        &device_mesh_indices_buffer,
                        &primitives_info,
                        &device_uniform_sub_allocation,
                    );
                    (Some(res.0), Some(res.1))
                }
                _ => (None, None),
            };

        vk_model.state = Some(Box::new(Device {
            device_mesh_indices_buffer,
            host_uniform_sub_allocation,
            device_uniform_sub_allocation,
            device_primitives_info: primitives_info,
            acceleration_structure,
            device_acceleration_structure_buffer,
        }));
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct DevicePrimitiveInfo {
    mesh_buffer_offset: u64,
    mesh_size: u64,
    single_mesh_element_size: u32,

    indices_buffer_offset: u64,
    indices_size: u64,
    single_index_size: u32,

    image: ImageAllocation,
    image_format: vk::Format,
    image_extent: vk::Extent3D,
    image_mip_levels: u32,
    image_layers: u32,
}

impl DevicePrimitiveInfo {
    fn get_indices_type(&self) -> vk::IndexType {
        match self.single_index_size {
            2 => vk::IndexType::UINT16,
            4 => vk::IndexType::UINT32,
            _ => {
                panic!("Non standard index type")
            }
        }
    }

    fn get_triangles_count(&self) -> u64 {
        (self.indices_size / self.single_index_size as u64) / 3
    }
}

struct Device {
    device_mesh_indices_buffer: BufferAllocation,
    host_uniform_sub_allocation: SubAllocationData,
    device_uniform_sub_allocation: SubAllocationData,
    acceleration_structure: Option<vk::AccelerationStructureKHR>,
    device_acceleration_structure_buffer: Option<BufferAllocation>,
    device_primitives_info: Vec<DevicePrimitiveInfo>,
}

impl VkModelTransferLocation for Device {
    fn to_disk(mut self: Box<Device>, vk_model: &mut VkModel) {
        let mut allocator = vk_model.allocator.as_ref().borrow_mut();

        allocator
            .get_allocator_mut()
            .destroy_buffer(self.device_mesh_indices_buffer);
        allocator
            .get_host_uniform_sub_allocator_mut()
            .free(self.host_uniform_sub_allocation);
        allocator
            .get_device_uniform_sub_allocator_mut()
            .free(self.device_uniform_sub_allocation);

        if let Some(buffer) = self.device_acceleration_structure_buffer {
            allocator.get_allocator_mut().destroy_buffer(buffer);
        }
        if let Some(acceleration_structure) = self.acceleration_structure {
            unsafe {
                vk_model
                    .acceleration_structure_fp
                    .unwrap()
                    .destroy_acceleration_structure(acceleration_structure, None);
            }
        }
        self.device_primitives_info
            .drain(0..)
            .for_each(|primitive_info| {
                allocator
                    .get_allocator_mut()
                    .destroy_image(primitive_info.image);
            });
        vk_model.state = Some(Box::new(Storage {}))
    }

    fn to_host(self: Box<Device>, vk_model: &mut VkModel, cb: vk::CommandBuffer) {
        {
            let mut allocator = vk_model.allocator.as_ref().borrow_mut();
            allocator
                .get_host_uniform_sub_allocator_mut()
                .free(self.host_uniform_sub_allocation);
            allocator
                .get_device_uniform_sub_allocator_mut()
                .free(self.device_uniform_sub_allocation);
            if let Some(buffer) = self.device_acceleration_structure_buffer {
                allocator.get_allocator_mut().destroy_buffer(buffer);
            }
            if let Some(acceleration_structure) = self.acceleration_structure {
                unsafe {
                    vk_model
                        .acceleration_structure_fp
                        .unwrap()
                        .destroy_acceleration_structure(acceleration_structure, None);
                }
            }
        }

        vk_model.state = Some(vk_model.transfer_from_device_to_host(
            cb,
            self.device_mesh_indices_buffer,
            self.device_primitives_info,
        ));
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

trait VkModelPostSubmissionCleanup {
    fn cleanup(self: Box<Self>, vk_model: &VkModel);
}

struct HostToDeviceTransfer {
    host_buffer_allocation: BufferAllocation,
}
impl VkModelPostSubmissionCleanup for HostToDeviceTransfer {
    fn cleanup(self: Box<HostToDeviceTransfer>, vk_model: &VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .destroy_buffer(self.host_buffer_allocation);
    }
}

struct DeviceToHostTransfer {
    device_mesh_indices_allocation: BufferAllocation,
    device_images: Vec<ImageAllocation>,
}
impl VkModelPostSubmissionCleanup for DeviceToHostTransfer {
    fn cleanup(mut self: Box<DeviceToHostTransfer>, vk_model: &VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .destroy_buffer(self.device_mesh_indices_allocation);

        self.device_images.drain(..).for_each(|image_allocation| {
            vk_model
                .allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(image_allocation);
        });
    }
}

struct BlasBuild {
    scratch_buffer: BufferAllocation,
}
impl VkModelPostSubmissionCleanup for BlasBuild {
    fn cleanup(self: Box<Self>, vk_model: &VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .destroy_buffer(self.scratch_buffer);
    }
}

pub struct VkModel<'a> {
    device: &'a ash::Device,
    acceleration_structure_fp: Option<&'a khr::AccelerationStructure>,
    allocator: Rc<RefCell<VkAllocator<'a>>>,
    model_path: String,
    model_bounding_sphere: Option<Sphere>,
    uniform: VkModelUniform,
    state: Option<Box<dyn VkModelTransferLocation>>,
    needs_cb_submit: bool,
    post_cb_submit_cleanups: Vec<Box<dyn VkModelPostSubmissionCleanup>>,
}

#[repr(C, packed)]
struct VkModelUniform {
    model_matrix: Matrix3x4<f32>,
}

impl<'a> VkModel<'a> {
    pub fn new(
        device: &'a ash::Device,
        acceleration_structure_fp: Option<&'a khr::AccelerationStructure>,
        allocator: Rc<RefCell<VkAllocator<'a>>>,
        model_path: String,
        model_matrix: Matrix3x4<f32>,
    ) -> Self {
        let mut model = VkModel {
            device,
            acceleration_structure_fp,
            allocator,
            model_path,
            model_bounding_sphere: None,
            uniform: VkModelUniform { model_matrix },
            state: Some(Box::new(Storage {})),
            needs_cb_submit: false,
            post_cb_submit_cleanups: Vec::new(),
        };
        if let Some(state) = model.state.take() {
            // passing a null command buffer is acceptable here since we know the initial
            // state is disk, and disk -> host does not require a submission
            state.to_host(&mut model, vk::CommandBuffer::null());
        }
        model
    }

    pub fn update_model_status(&mut self, camera_pos: &Vector3<f32>, cb: vk::CommandBuffer) {
        let distance = self
            .model_bounding_sphere
            .as_ref()
            .unwrap()
            .transform(self.uniform.model_matrix)
            .get_distance_from_point(*camera_pos);

        let state = self.state.take().unwrap();
        match distance {
            x if x <= 10f32 => state.to_device(self, cb),
            x if x <= 20f32 => state.to_host(self, cb),
            _ => state.to_disk(self),
        };
    }

    pub fn needs_command_buffer_submission(&self) -> bool {
        self.needs_cb_submit
    }

    pub fn reset_command_buffer_submission_status(&mut self) {
        while let Some(elem) = self.post_cb_submit_cleanups.pop() {
            elem.cleanup(self);
        }
        self.needs_cb_submit = false;
    }

    pub fn get_model_matrix(&self) -> vk::TransformMatrixKHR {
        let matrix: [f32; 12] = self.uniform.model_matrix.as_slice().try_into().unwrap();
        vk::TransformMatrixKHR { matrix }
    }

    pub fn get_acceleration_structure_instance(
        &self,
    ) -> Option<vk::AccelerationStructureInstanceKHR> {
        let device_state = self.state.as_ref()?.as_any().downcast_ref::<Device>()?;

        Some(vk::AccelerationStructureInstanceKHR {
            transform: self.get_model_matrix(),
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: device_state
                    .device_acceleration_structure_buffer
                    .as_ref()?
                    .get_device_address()
                    .unwrap(),
            },
        })
    }

    pub fn get_blas_buffer(&self) -> Option<vk::Buffer> {
        Some(
            self.state
                .as_ref()?
                .as_any()
                .downcast_ref::<Device>()?
                .device_acceleration_structure_buffer
                .as_ref()?
                .get_buffer(),
        )
    }

    fn copy_uniform(
        &mut self,
        host_uniform_sub_allocation: &SubAllocationData,
        cb: vk::CommandBuffer,
        device_uniform_sub_allocation: &SubAllocationData,
    ) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.uniform,
                host_uniform_sub_allocation.get_host_ptr().unwrap().as_ptr() as *mut VkModelUniform,
                std::mem::size_of::<VkModelUniform>(),
            );

            let buffer_copy_region = vk::BufferCopy2::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(std::mem::size_of::<VkModelUniform>() as u64);
            let copy_buffer_info = vk::CopyBufferInfo2::builder()
                .src_buffer(host_uniform_sub_allocation.get_buffer())
                .dst_buffer(device_uniform_sub_allocation.get_buffer())
                .regions(std::slice::from_ref(&buffer_copy_region));
            self.device.cmd_copy_buffer2(cb, &copy_buffer_info);
        }
        self.needs_cb_submit = true;
    }

    fn transfer_from_disk_to_host(&mut self) -> (BufferAllocation, ModelCopyInfo) {
        let model = GltfModelReader::open(
            self.model_path.as_ref(),
            true,
            Some(vk::Format::B8G8R8A8_UNORM),
        );

        if self.model_bounding_sphere.is_none() {
            self.model_bounding_sphere = Some(model.get_primitives_bounding_sphere());
        }

        let mesh_attributes = MeshAttributeType::VERTICES
            | MeshAttributeType::TEX_COORDS
            | MeshAttributeType::NORMALS
            | MeshAttributeType::TANGENTS
            | MeshAttributeType::INDICES;
        let texture_types = TextureType::ALBEDO | TextureType::ORM | TextureType::NORMAL;

        let copy_info = model.copy_model_data_to_ptr(mesh_attributes, texture_types, None);

        // creating the host buffer for copying to the device
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(copy_info.compute_total_size() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let mut transient_allocation = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::CpuToGpu);

        // copying contents to host memory
        let dst_ptr = transient_allocation.get_allocation_mut().mapped_slice_mut();
        model.copy_model_data_to_ptr(mesh_attributes, texture_types, dst_ptr);

        (transient_allocation, copy_info)
    }

    fn transfer_from_host_to_device(
        &mut self,
        cb: vk::CommandBuffer,
        host_buffer_allocation: BufferAllocation,
        host_model_copy_info: ModelCopyInfo,
    ) -> (BufferAllocation, Vec<DevicePrimitiveInfo>) {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(host_model_copy_info.compute_mesh_and_indices_size() as u64)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            );
        let device_mesh_indices_buffer = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::GpuOnly);

        let device_images_allocations = host_model_copy_info
            .get_primitive_data()
            .iter()
            .map(|primitive_copy_info| {
                let image_create_info = vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(primitive_copy_info.image_format)
                    .extent(primitive_copy_info.image_extent)
                    .mip_levels(primitive_copy_info.image_mip_levels)
                    .array_layers(primitive_copy_info.image_layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                self.allocator
                    .as_ref()
                    .borrow_mut()
                    .get_allocator_mut()
                    .allocate_image(&image_create_info, MemoryLocation::GpuOnly)
            })
            .collect::<Vec<ImageAllocation>>();

        // record the buffer copies
        let mut buffer_copies = Vec::<vk::BufferCopy>::new();

        let mut destination_offset: u64 = 0;
        for primitive_copy_info in host_model_copy_info.get_primitive_data() {
            // mesh buffer copy
            buffer_copies.push(vk::BufferCopy {
                src_offset: primitive_copy_info.mesh_buffer_offset,
                dst_offset: destination_offset,
                size: primitive_copy_info.mesh_size,
            });
            destination_offset += buffer_copies.last().unwrap().size;

            // indices buffer copy
            buffer_copies.push(vk::BufferCopy {
                src_offset: primitive_copy_info.indices_buffer_offset,
                dst_offset: destination_offset,
                size: primitive_copy_info.indices_size,
            });
            destination_offset += buffer_copies.last().unwrap().size;
        }

        unsafe {
            self.device.cmd_copy_buffer(
                cb,
                host_buffer_allocation.get_buffer(),
                device_mesh_indices_buffer.get_buffer(),
                &buffer_copies,
            );
        }

        let mut image_memory_barriers = device_images_allocations
            .iter()
            .map(|device_image_allocation| {
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(device_image_allocation.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build()
            })
            .collect::<Vec<_>>();

        unsafe {
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);
        }

        // record the image copies
        for (primitive_copy_info, device_image_allocation) in std::iter::zip(
            host_model_copy_info.get_primitive_data(),
            device_images_allocations.iter(),
        ) {
            let buffer_image_copy = vk::BufferImageCopy {
                buffer_offset: primitive_copy_info.image_buffer_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: primitive_copy_info.image_layers,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: primitive_copy_info.image_extent,
            };

            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    cb,
                    host_buffer_allocation.get_buffer(),
                    device_image_allocation.get_image(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&buffer_image_copy),
                );
            }
        }

        image_memory_barriers.iter_mut().for_each(|memory_barrier| {
            memory_barrier.src_stage_mask = vk::PipelineStageFlags2::COPY;
            memory_barrier.src_access_mask = vk::AccessFlags2::TRANSFER_WRITE;
            memory_barrier.dst_stage_mask = vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
            memory_barrier.dst_access_mask = vk::AccessFlags2::SHADER_SAMPLED_READ;
            memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            memory_barrier.new_layout = vk::ImageLayout::READ_ONLY_OPTIMAL;
        });

        unsafe {
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);
        }

        let primitives_model_info = itertools::izip!(
            host_model_copy_info.get_primitive_data(),
            buffer_copies.windows(2),
            device_images_allocations
        )
        .map(
            |(host_copy_info, mesh_and_idx_buffer_copy, image)| DevicePrimitiveInfo {
                mesh_buffer_offset: mesh_and_idx_buffer_copy[0].dst_offset,
                mesh_size: mesh_and_idx_buffer_copy[0].size,
                single_mesh_element_size: host_copy_info.single_mesh_element_size,
                indices_buffer_offset: mesh_and_idx_buffer_copy[1].dst_offset,
                indices_size: mesh_and_idx_buffer_copy[1].size,
                single_index_size: host_copy_info.single_index_size,
                image,
                image_format: host_copy_info.image_format,
                image_extent: host_copy_info.image_extent,
                image_mip_levels: host_copy_info.image_mip_levels,
                image_layers: host_copy_info.image_layers,
            },
        )
        .collect::<Vec<_>>();

        self.needs_cb_submit = true;
        self.post_cb_submit_cleanups
            .push(Box::new(HostToDeviceTransfer {
                host_buffer_allocation,
            }));

        (device_mesh_indices_buffer, primitives_model_info)
    }

    fn transfer_from_device_to_host(
        &mut self,
        cb: vk::CommandBuffer,
        device_mesh_indices_allocation: BufferAllocation,
        mut device_primitives_info: Vec<DevicePrimitiveInfo>,
    ) -> Box<Host> {
        let host_buffer_size =
            device_primitives_info
                .iter()
                .fold(0, |accum: u64, primitive_info| {
                    align_offset(
                        accum + primitive_info.mesh_size + primitive_info.indices_size,
                        4,
                    ) + primitive_info.image.get_allocation().size()
                });

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(host_buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        let host_buffer_allocation = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, MemoryLocation::CpuToGpu);

        let image_memory_barriers = device_primitives_info
            .iter()
            .map(|device_primitive_info| {
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(
                        vk::PipelineStageFlags2::FRAGMENT_SHADER
                            | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    )
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(device_primitive_info.image.get_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .build()
            })
            .collect::<Vec<_>>();

        unsafe {
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);
        }

        let mut destination_offset: u64 = 0;
        let mut buffer_copies = Vec::<vk::BufferCopy>::new();
        let mut primitives_copy_data = Vec::<PrimitiveCopyInfo>::new();
        for device_primitive_info in device_primitives_info.iter() {
            // record the buffer copy
            buffer_copies.push(vk::BufferCopy {
                src_offset: device_primitive_info.mesh_buffer_offset,
                dst_offset: destination_offset,
                size: device_primitive_info.mesh_size,
            });
            destination_offset += device_primitive_info.mesh_size;
            buffer_copies.push(vk::BufferCopy {
                src_offset: device_primitive_info.indices_buffer_offset,
                dst_offset: destination_offset,
                size: device_primitive_info.indices_size,
            });
            destination_offset += device_primitive_info.indices_size;
            destination_offset = align_offset(destination_offset, 4);
            // record the image copy
            let buffer_image_copy = vk::BufferImageCopy {
                buffer_offset: destination_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: device_primitive_info.image_layers,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: device_primitive_info.image_extent,
            };
            destination_offset += device_primitive_info.image.get_allocation().size();

            unsafe {
                self.device.cmd_copy_image_to_buffer(
                    cb,
                    device_primitive_info.image.get_image(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    host_buffer_allocation.get_buffer(),
                    std::slice::from_ref(&buffer_image_copy),
                );
            }

            primitives_copy_data.push(PrimitiveCopyInfo {
                mesh_buffer_offset: buffer_copies
                    .get(buffer_copies.len() - 2)
                    .unwrap()
                    .dst_offset,
                mesh_size: device_primitive_info.mesh_size,
                single_mesh_element_size: device_primitive_info.single_mesh_element_size,
                indices_buffer_offset: buffer_copies.last().unwrap().dst_offset,
                indices_size: device_primitive_info.indices_size,
                single_index_size: device_primitive_info.single_index_size,
                image_buffer_offset: buffer_image_copy.buffer_offset,
                image_size: device_primitive_info.image.get_allocation().size(),
                image_format: device_primitive_info.image_format,
                image_extent: device_primitive_info.image_extent,
                image_mip_levels: device_primitive_info.image_mip_levels,
                image_layers: device_primitive_info.image_layers,
            });
        }

        unsafe {
            self.device.cmd_copy_buffer(
                cb,
                device_mesh_indices_allocation.get_buffer(),
                host_buffer_allocation.get_buffer(),
                &buffer_copies,
            );
        }

        self.needs_cb_submit = true;
        self.post_cb_submit_cleanups
            .push(Box::new(DeviceToHostTransfer {
                device_mesh_indices_allocation,
                device_images: device_primitives_info
                    .drain(..)
                    .map(|elem| elem.image)
                    .collect::<Vec<_>>(),
            }));

        Box::new(Host {
            host_buffer_allocation,
            host_model_copy_info: ModelCopyInfo::new(primitives_copy_data),
        })
    }

    fn create_blas(
        &mut self,
        cb: vk::CommandBuffer,
        device_mesh_allocation: &BufferAllocation,
        device_primitives_info: &[DevicePrimitiveInfo],
        device_uniform_allocation: &SubAllocationData,
    ) -> (vk::AccelerationStructureKHR, BufferAllocation) {
        let as_geom_info = device_primitives_info
            .iter()
            .map(|device_primitive_info| {
                let vertices_address = vk::DeviceOrHostAddressConstKHR {
                    device_address: device_mesh_allocation.get_device_address().unwrap()
                        + device_primitive_info.mesh_buffer_offset,
                };
                let indices_address = vk::DeviceOrHostAddressConstKHR {
                    device_address: device_mesh_allocation.get_device_address().unwrap()
                        + device_primitive_info.indices_buffer_offset,
                };

                let acceleration_structure_geometry_triangles_data =
                    vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vertices_address)
                        .vertex_stride(
                            device_primitive_info.single_mesh_element_size as vk::DeviceSize,
                        )
                        .max_vertex(
                            (device_primitive_info.mesh_size
                                / device_primitive_info.single_mesh_element_size as u64)
                                as u32,
                        )
                        .index_type(device_primitive_info.get_indices_type())
                        .index_data(indices_address)
                        .build();
                let acceleration_structure_geometry_data =
                    vk::AccelerationStructureGeometryDataKHR {
                        triangles: acceleration_structure_geometry_triangles_data,
                    };
                let acceleration_structure_geometry =
                    vk::AccelerationStructureGeometryKHR::builder()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(acceleration_structure_geometry_data)
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                        .build();
                acceleration_structure_geometry
            })
            .collect::<Vec<_>>();

        let as_build_ranges = device_primitives_info
            .iter()
            .map(|device_primitive_info| {
                vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(device_primitive_info.get_triangles_count() as u32)
                    .primitive_offset(0)
                    .first_vertex(0)
                    .transform_offset(0)
                    .build()
            })
            .collect::<Vec<_>>();

        let mut as_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geom_info)
            .build();

        let as_size_info = unsafe {
            self.acceleration_structure_fp
                .unwrap()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &as_build_info,
                    &as_build_ranges
                        .iter()
                        .map(|e| e.primitive_count)
                        .collect::<Vec<_>>(),
                )
        };

        let buffer_create_info = vk::BufferCreateInfo::builder()
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
                .unwrap()
                .create_acceleration_structure(&as_create_info, None)
                .unwrap()
        };

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(as_size_info.build_scratch_size)
            .usage(
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            );
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
            let buffer_memory_barriers = [
                vk::BufferMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .buffer(device_mesh_allocation.get_buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .build(),
                vk::BufferMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .buffer(device_uniform_allocation.get_buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .build(),
            ];
            let dependancy_info =
                vk::DependencyInfo::builder().buffer_memory_barriers(&buffer_memory_barriers);
            self.device.cmd_pipeline_barrier2(cb, &dependancy_info);

            self.acceleration_structure_fp
                .unwrap()
                .cmd_build_acceleration_structures(
                    cb,
                    std::slice::from_ref(&as_build_info),
                    &[&as_build_ranges],
                );
        }

        self.needs_cb_submit = true;
        self.post_cb_submit_cleanups
            .push(Box::new(BlasBuild { scratch_buffer }));
        (as_build_info.dst_acceleration_structure, device_blas_buffer)
    }
}

impl<'a> Drop for VkModel<'a> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state.to_disk(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk_renderer::vk_allocator::VkAllocator;
    use crate::vk_renderer::vk_boot::vk_base::VkBase;
    use crate::vk_renderer::vk_model::VkModel;
    use ash::vk;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_water_bottle() {
        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);
        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::builder().synchronization2(true);
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut vulkan_13_features)
            .push_next(&mut vulkan_12_features);
        let bvk = VkBase::new(
            "",
            &[],
            &[],
            &physical_device_features2,
            &[(vk::QueueFlags::GRAPHICS, 1.0f32)],
            None,
        );

        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(bvk.queue_family_index());
        let command_pool = unsafe {
            bvk.device()
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };

        let allocator = Rc::new(RefCell::new(VkAllocator::new(
            bvk.instance().clone(),
            bvk.device(),
            bvk.physical_device().clone(),
        )));

        let mut water_bottle = VkModel::new(
            bvk.device(),
            None,
            allocator.clone(),
            String::from("assets/models/WaterBottle.glb"),
            Matrix4::<f32>::new_translation(&Vector3::<f32>::from_element(0.0f32)).remove_row(3),
        );
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Host>());

        let command_buffer_create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            bvk.device()
                .allocate_command_buffers(&command_buffer_create_info)
                .unwrap()
                .first()
                .unwrap()
                .clone()
        };
        unsafe {
            let begin_command_buffer = vk::CommandBufferBeginInfo::default();
            bvk.device()
                .begin_command_buffer(command_buffer, &begin_command_buffer)
                .unwrap();
        };

        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = unsafe { bvk.device().create_fence(&fence_create_info, None).unwrap() };

        water_bottle.update_model_status(&Vector3::from_element(100.0f32), command_buffer);
        assert!(water_bottle
            .state
            .as_ref()
            .unwrap()
            .as_any()
            .is::<Storage>());
        assert!(!water_bottle.needs_command_buffer_submission());

        water_bottle.update_model_status(&Vector3::from_element(7.0f32), command_buffer);
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Host>());
        assert!(!water_bottle.needs_command_buffer_submission());

        let from_disk_host_data = {
            let host_buffer_allocation = &water_bottle
                .state
                .as_ref()
                .unwrap()
                .as_any()
                .downcast_ref::<Host>()
                .unwrap()
                .host_buffer_allocation;

            unsafe {
                std::slice::from_raw_parts(
                    host_buffer_allocation
                        .get_allocation()
                        .mapped_ptr()
                        .unwrap()
                        .as_ptr() as *mut u8,
                    host_buffer_allocation.get_allocation().size() as usize,
                )
            }
        };

        water_bottle.update_model_status(&Vector3::from_element(3.0f32), command_buffer);
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Device>());
        assert!(water_bottle.needs_command_buffer_submission());

        let command_buffer_submit_info =
            vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);
        let queue_submit2 = vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&command_buffer_submit_info));
        unsafe {
            bvk.device().end_command_buffer(command_buffer).unwrap();

            bvk.device()
                .queue_submit2(
                    *bvk.queues().first().unwrap(),
                    std::slice::from_ref(&queue_submit2),
                    fence,
                )
                .unwrap();

            bvk.device()
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .unwrap();
            bvk.device()
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        water_bottle.reset_command_buffer_submission_status();

        unsafe {
            let begin_command_buffer = vk::CommandBufferBeginInfo::default();
            bvk.device()
                .begin_command_buffer(command_buffer, &begin_command_buffer)
                .unwrap();
        };

        water_bottle.update_model_status(&Vector3::from_element(7.0f32), command_buffer);
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Host>());
        assert!(water_bottle.needs_command_buffer_submission());

        let command_buffer_submit_info =
            vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);
        let queue_submit2 = vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&command_buffer_submit_info));
        unsafe {
            bvk.device()
                .reset_fences(std::slice::from_ref(&fence))
                .unwrap();
            bvk.device().end_command_buffer(command_buffer).unwrap();

            bvk.device()
                .queue_submit2(
                    *bvk.queues().first().unwrap(),
                    std::slice::from_ref(&queue_submit2),
                    fence,
                )
                .unwrap();

            bvk.device()
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .unwrap();
        }
        water_bottle.reset_command_buffer_submission_status();

        unsafe {
            bvk.device()
                .free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));
            bvk.device().destroy_command_pool(command_pool, None);
            bvk.device().destroy_fence(fence, None);
        }

        let from_device_host_data = {
            let host_buffer_allocation = &water_bottle
                .state
                .as_ref()
                .unwrap()
                .as_any()
                .downcast_ref::<Host>()
                .unwrap()
                .host_buffer_allocation;

            unsafe {
                std::slice::from_raw_parts(
                    host_buffer_allocation
                        .get_allocation()
                        .mapped_ptr()
                        .unwrap()
                        .as_ptr() as *mut u8,
                    host_buffer_allocation.get_allocation().size() as usize,
                )
            }
        };
        assert_eq!(from_disk_host_data.len(), from_device_host_data.len());
        for (from_disk_byte, from_device_byte) in
            std::iter::zip(from_disk_host_data, from_device_host_data)
        {
            assert_eq!(*from_disk_byte, *from_device_byte);
        }
    }

    #[test]
    fn test_water_bottle_blas() {
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
        let acceleration_structure_fp =
            khr::AccelerationStructure::new(bvk.instance(), bvk.device());

        let allocator = Rc::new(RefCell::new(VkAllocator::new(
            bvk.instance().clone(),
            bvk.device(),
            bvk.physical_device().clone(),
        )));
        let mut water_bottle = VkModel::new(
            bvk.device(),
            Some(&acceleration_structure_fp),
            allocator.clone(),
            String::from("assets/models/WaterBottle.glb"),
            Matrix4::<f32>::new_translation(&Vector3::<f32>::from_element(0.0f32)).remove_row(3),
        );
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Host>());

        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(bvk.queue_family_index());
        let command_pool = unsafe {
            bvk.device()
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };
        let command_buffer_create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            bvk.device()
                .allocate_command_buffers(&command_buffer_create_info)
                .unwrap()[0]
        };
        unsafe {
            let begin_command_buffer = vk::CommandBufferBeginInfo::default();
            bvk.device()
                .begin_command_buffer(command_buffer, &begin_command_buffer)
                .unwrap();
        };

        water_bottle.update_model_status(&Vector3::from_element(3.0f32), command_buffer);
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Device>());
        assert!(water_bottle.needs_command_buffer_submission());

        unsafe {
            bvk.device().end_command_buffer(command_buffer).unwrap();
        }

        let command_buffer_submit_info =
            vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);
        let queue_submit2 = vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&command_buffer_submit_info));
        unsafe {
            bvk.device()
                .queue_submit2(
                    *bvk.queues().first().unwrap(),
                    std::slice::from_ref(&queue_submit2),
                    vk::Fence::null(),
                )
                .unwrap();
            bvk.device().device_wait_idle().unwrap();
            bvk.device()
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        water_bottle.reset_command_buffer_submission_status();
        unsafe {
            bvk.device()
                .free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));
            bvk.device().destroy_command_pool(command_pool, None);
        }
        assert!(water_bottle
            .state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<Device>()
            .unwrap()
            .device_acceleration_structure_buffer
            .is_some());
        assert!(water_bottle
            .state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<Device>()
            .unwrap()
            .acceleration_structure
            .is_some());
    }
}
