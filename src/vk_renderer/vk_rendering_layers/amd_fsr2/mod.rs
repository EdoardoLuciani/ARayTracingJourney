mod amd_fsr2_api_vk_bindings;

use amd_fsr2_api_vk_bindings::*;
use ash::{extensions::*, vk, Entry, Instance};
use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use std::rc::Rc;

pub struct AmdFsr2 {
    device: Rc<ash::Device>,
    physical_device: vk::PhysicalDevice,
    scratch_buffer: Vec<u8>,
    context_description: FfxFsr2ContextDescription,
    context: FfxFsr2Context,
}

impl AmdFsr2 {
    pub fn new(
        device: Rc<ash::Device>,
        physical_device: vk::PhysicalDevice,
        entry: &Entry,
        instance: &Instance,
        input_res: vk::Extent2D,
        output_res: vk::Extent2D,
    ) -> Self {
        let mut context_description = unsafe {
            FfxFsr2ContextDescription {
                flags: 0,
                maxRenderSize: FfxDimensions2D::default(),
                displaySize: FfxDimensions2D::default(),
                callbacks: std::mem::zeroed(),
                device: ffxGetDeviceVK(std::mem::transmute::<vk::Device, VkDevice>(
                    device.handle(),
                )),
            }
        };

        let scratch_buffer_size = unsafe {
            let pd = std::mem::transmute::<_, _>(physical_device);
            ffxFsr2GetScratchMemorySizeVK(
                pd,
                std::mem::transmute::<_, _>(
                    instance.fp_v1_0().enumerate_device_extension_properties,
                ),
            ) as usize
        };

        let mut scratch_buffer = vec![0u8; scratch_buffer_size];

        unsafe {
            ffxFsr2GetInterfaceVK(
                &mut context_description.callbacks as *mut FfxFsr2Interface,
                scratch_buffer.as_mut_ptr() as *mut c_void,
                scratch_buffer.len() as size_t,
                std::mem::transmute::<_, _>(instance.handle()),
                std::mem::transmute::<_, _>(physical_device),
                std::mem::transmute::<_, _>(entry.static_fn().get_instance_proc_addr),
            );
        };

        let mut ret = unsafe {
            Self {
                device,
                physical_device,
                context_description,
                scratch_buffer,
                context: std::mem::zeroed(),
            }
        };

        ret.resize(input_res, output_res);

        ret
    }

    pub fn resize(&mut self, input_res: vk::Extent2D, output_res: vk::Extent2D) {
        unsafe {
            ffxFsr2ContextDestroy(&mut self.context);

            self.context_description.maxRenderSize = FfxDimensions2D {
                width: input_res.width,
                height: input_res.height,
            };
            self.context_description.displaySize = FfxDimensions2D {
                width: output_res.width,
                height: output_res.height,
            };

            ffxFsr2ContextCreate(&mut self.context, &self.context_description);
        }
    }
}

impl Drop for AmdFsr2 {
    fn drop(&mut self) {
        unsafe {
            ffxFsr2ContextDestroy(&mut self.context);
        }
    }
}
