use std::any::Any;
use std::boxed::Box;
use std::cell::RefCell;
use std::rc::Rc;

use ash::vk;
use ash::vk::BufferCopy;
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
    fn to_disk(self: Box<Self>, vk_model: &mut VkModel) {}
    fn to_host(self: Box<Self>, vk_model: &mut VkModel) {}
    fn to_device(self: Box<Self>, vk_model: &mut VkModel) {}
    fn as_any(&self) -> &dyn Any;
}

struct Disk;
impl VkModelTransferLocation for Disk {
    fn to_host(self: Box<Disk>, vk_model: &mut VkModel) {
        let host = vk_model.transfer_from_disk_to_host();
        vk_model.state = Some(host);
    }

    fn to_device(self: Box<Disk>, vk_model: &mut VkModel) {
        let host = vk_model.transfer_from_disk_to_host();
        let device = vk_model
            .transfer_from_host_to_device(host.host_buffer_allocation, host.host_model_copy_info);
        vk_model.state = Some(device);
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
        vk_model.state = Some(Box::new(Disk {}));
    }

    fn to_device(self: Box<Host>, vk_model: &mut VkModel) {
        let device = vk_model
            .transfer_from_host_to_device(self.host_buffer_allocation, self.host_model_copy_info);
        vk_model.state = Some(device);
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

struct Device {
    device_mesh_suballocation: SubAllocationData,
    device_primitives_info: Vec<DevicePrimitiveInfo>,
}

impl VkModelTransferLocation for Device {
    fn to_disk(mut self: Box<Device>, vk_model: &mut VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_device_mesh_indices_sub_allocator_mut()
            .free(self.device_mesh_suballocation);
        self.device_primitives_info
            .drain(0..)
            .for_each(|primitive_info| {
                vk_model
                    .allocator
                    .as_ref()
                    .borrow_mut()
                    .get_allocator_mut()
                    .destroy_image(primitive_info.image);
            });
        vk_model.state = Some(Box::new(Disk {}))
    }

    fn to_host(self: Box<Device>, vk_model: &mut VkModel) {
        vk_model.state = Some(vk_model.transfer_from_device_to_host(
            self.device_mesh_suballocation,
            self.device_primitives_info,
        ));
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

trait VkModelPostSubmissionCleanup {
    fn cleanup(self: Box<Self>, vk_model: &mut VkModel);
}

struct HostToDevice {
    host_buffer_allocation: BufferAllocation,
}

impl VkModelPostSubmissionCleanup for HostToDevice {
    fn cleanup(self: Box<HostToDevice>, vk_model: &mut VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .destroy_buffer(self.host_buffer_allocation);
    }
}

struct DeviceToHost {
    device_mesh_suballocation: SubAllocationData,
    device_images: Vec<ImageAllocation>,
}

impl VkModelPostSubmissionCleanup for DeviceToHost {
    fn cleanup(mut self: Box<DeviceToHost>, vk_model: &mut VkModel) {
        vk_model
            .allocator
            .as_ref()
            .borrow_mut()
            .get_device_mesh_indices_sub_allocator_mut()
            .free(self.device_mesh_suballocation);

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

struct VkModel<'a> {
    device: &'a ash::Device,
    allocator: Rc<RefCell<VkAllocator<'a>>>,
    transfer_command_buffer: vk::CommandBuffer,
    model_path: String,
    model_bounding_sphere: Option<Sphere>,
    uniform: VkModelUniform,
    state: Option<Box<dyn VkModelTransferLocation>>,
    needs_cb_submit: bool,
    post_cb_submit_cleanup: Option<Box<dyn VkModelPostSubmissionCleanup>>,
}

struct VkModelUniform {
    model_matrix: Matrix4<f32>,
}

impl<'a> VkModel<'a> {
    pub fn new(
        device: &'a ash::Device,
        allocator: Rc<RefCell<VkAllocator<'a>>>,
        model_path: String,
        model_matrix: Matrix4<f32>,
        transfer_command_buffer: vk::CommandBuffer,
    ) -> Self {
        let mut model = VkModel {
            device,
            allocator,
            transfer_command_buffer,
            model_path,
            model_bounding_sphere: None,
            uniform: VkModelUniform { model_matrix },
            state: Some(Box::new(Disk {})),
            needs_cb_submit: false,
            post_cb_submit_cleanup: None,
        };
        if let Some(state) = model.state.take() {
            state.to_host(&mut model);
        }
        model
    }

    pub fn update_model_status(&mut self, camera_pos: &Vector3<f32>) {
        let distance = self
            .model_bounding_sphere
            .as_ref()
            .unwrap()
            .transform(self.uniform.model_matrix)
            .get_distance_from_point(*camera_pos);

        let state = self.state.take().unwrap();
        match distance {
            x if x <= 10f32 => state.to_device(self),
            x if x <= 20f32 => state.to_host(self),
            _ => state.to_disk(self),
        };
    }

    pub fn needs_command_buffer_submission(&self) -> bool {
        self.needs_cb_submit
    }

    pub fn reset_command_buffer_submission_status(&mut self) {
        if let Some(cleanup_struct) = self.post_cb_submit_cleanup.take() {
            cleanup_struct.cleanup(self);
        }
        self.needs_cb_submit = false;
    }

    fn transfer_from_disk_to_host(&mut self) -> Box<Host> {
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

        let copy_info =
            model.copy_model_data_to_ptr(mesh_attributes, texture_types, std::ptr::null_mut());

        // creating the host buffer for copying to the device
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(copy_info.compute_total_size() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let transient_allocation = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_allocator_mut()
            .allocate_buffer(&buffer_create_info, gpu_allocator::MemoryLocation::CpuToGpu);

        // copying contents to host memory
        let dst_ptr = transient_allocation
            .get_allocation()
            .mapped_ptr()
            .unwrap()
            .as_ptr() as *mut u8;
        model.copy_model_data_to_ptr(mesh_attributes, texture_types, dst_ptr);

        Box::new(Host {
            host_buffer_allocation: transient_allocation,
            host_model_copy_info: copy_info,
        })
    }

    fn transfer_from_host_to_device(
        &mut self,
        host_buffer_allocation: BufferAllocation,
        host_model_copy_info: ModelCopyInfo,
    ) -> Box<Device> {
        let device_mesh_suballocation = self
            .allocator
            .as_ref()
            .borrow_mut()
            .get_device_mesh_indices_sub_allocator_mut()
            .allocate(host_model_copy_info.compute_mesh_and_indices_size(), 1);

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
        let mut buffer_copies = Vec::<BufferCopy>::new();

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
                self.transfer_command_buffer,
                host_buffer_allocation.get_buffer(),
                device_mesh_suballocation.get_buffer(),
                &buffer_copies,
            );
        }

        let mut image_memory_barriers = device_images_allocations
            .iter()
            .map(|device_image_allocation| {
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
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
            let dependacy_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device
                .cmd_pipeline_barrier2(self.transfer_command_buffer, &dependacy_info);
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
                    self.transfer_command_buffer,
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
            memory_barrier.dst_stage_mask = vk::PipelineStageFlags2::FRAGMENT_SHADER;
            memory_barrier.dst_access_mask = vk::AccessFlags2::SHADER_SAMPLED_READ;
            memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            memory_barrier.new_layout = vk::ImageLayout::READ_ONLY_OPTIMAL;
        });

        unsafe {
            let dependacy_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device
                .cmd_pipeline_barrier2(self.transfer_command_buffer, &dependacy_info);
        }

        let primitives_model_info = itertools::izip!(
            host_model_copy_info.get_primitive_data(),
            buffer_copies.windows(2),
            device_images_allocations
        )
        .map(
            |(host_copy_info, mesh_buffer_offset, image)| DevicePrimitiveInfo {
                mesh_buffer_offset: mesh_buffer_offset[0].dst_offset,
                mesh_size: mesh_buffer_offset[0].size,
                single_mesh_element_size: host_copy_info.single_mesh_element_size,
                indices_buffer_offset: mesh_buffer_offset[1].dst_offset,
                indices_size: mesh_buffer_offset[1].size,
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
        self.post_cb_submit_cleanup = Some(Box::new(HostToDevice {
            host_buffer_allocation,
        }));

        Box::new(Device {
            device_mesh_suballocation,
            device_primitives_info: primitives_model_info,
        })
    }

    fn transfer_from_device_to_host(
        &mut self,
        device_mesh_suballocation: SubAllocationData,
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

        let mut image_memory_barriers = device_primitives_info
            .iter()
            .map(|device_primitive_info| {
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
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
            let dependacy_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            self.device
                .cmd_pipeline_barrier2(self.transfer_command_buffer, &dependacy_info);
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
                    self.transfer_command_buffer,
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
                self.transfer_command_buffer,
                device_mesh_suballocation.get_buffer(),
                host_buffer_allocation.get_buffer(),
                &buffer_copies,
            );
        }

        self.needs_cb_submit = true;
        self.post_cb_submit_cleanup = Some(Box::new(DeviceToHost {
            device_mesh_suballocation,
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
    use nalgebra::{Matrix4, Vector3};
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

        let mut water_bottle = VkModel::new(
            bvk.device(),
            allocator.clone(),
            String::from("assets/models/WaterBottle.glb"),
            Matrix4::<f32>::new_translation(&Vector3::<f32>::from_element(0.0f32)),
            command_buffer,
        );
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Host>());

        water_bottle.update_model_status(&Vector3::from_element(100.0f32));
        assert!(water_bottle.state.as_ref().unwrap().as_any().is::<Disk>());
        assert!(!water_bottle.needs_command_buffer_submission());

        water_bottle.update_model_status(&Vector3::from_element(7.0f32));
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

        water_bottle.update_model_status(&Vector3::from_element(3.0f32));
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

        water_bottle.update_model_status(&Vector3::from_element(7.0f32));
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
}
