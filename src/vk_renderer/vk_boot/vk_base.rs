use super::helper::*;
use super::pointer_chain_helpers::*;
use ash::{extensions::*, vk};
use std::borrow::Borrow;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct VkBase {
    entry_fn: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    surface_fn: Option<khr::Surface>,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: ash::Device,
    queues: Vec<vk::Queue>,
    swapchain_fn: Option<khr::Swapchain>,
    swapchain_create_info: Option<vk::SwapchainCreateInfoKHR>,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    #[cfg(debug_assertions)]
    debug_utils_fn: ext::DebugUtils,
    #[cfg(debug_assertions)]
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
}

/**
BaseVk is struct that initializes a single Vulkan 1.1 instance and device with optional surface support.
It supports instance creation with extensions and device selection with Vulkan 1.1 features
and requested queues. It also initializes an allocator that greatly simplifies Vulkan allocations.
Basically it is a bootstrap for a very common vulkan setup.
*/
impl VkBase {
    pub fn new(
        application_name: &str,
        instance_extensions: &[&str],
        device_extensions: &[&str],
        desired_physical_device_features2: &vk::PhysicalDeviceFeatures2,
        desired_queues_with_priorities: &[(vk::QueueFlags, f32)],
        window_display_handle: Option<(RawWindowHandle, RawDisplayHandle)>,
    ) -> Self {
        let mut instance_extensions = Vec::from(instance_extensions);
        cfg_if::cfg_if! {
            if #[cfg(debug_assertions)] {
                instance_extensions.push("VK_EXT_debug_utils");
                instance_extensions.push("VK_EXT_validation_features");
                let layer_names = std::slice::from_ref(&"VK_LAYER_KHRONOS_validation");

                let validation_features_enable = [vk::ValidationFeatureEnableEXT::GPU_ASSISTED, vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
                let validation_features = vk::ValidationFeaturesEXT::builder()
                    .enabled_validation_features(&validation_features_enable)
                    .build();
                let instance_pnext = &validation_features as *const vk::ValidationFeaturesEXT as *const std::ffi::c_void;
            }
            else {
                let layer_names = [];
                let instance_pnext = std::ptr::null();
            }
        }

        // adding the required extensions needed for creating a surface based on the os
        if let Some(handle) = window_display_handle {
            instance_extensions.push("VK_KHR_surface");
            match handle.0 {
                RawWindowHandle::Win32(_) => {
                    instance_extensions.push("VK_KHR_win32_surface");
                }
                RawWindowHandle::Xlib(_) => {
                    instance_extensions.push("VK_KHR_xlib_surface");
                }
                RawWindowHandle::Wayland(_) => {
                    instance_extensions.push("VK_KHR_wayland_surface");
                }
                _ => {
                    panic!("Unrecognized window handle")
                }
            };
        }

        let entry_fn = unsafe { ash::Entry::load().unwrap() };
        let instance = Self::create_instance(
            &entry_fn,
            application_name,
            instance_pnext,
            &instance_extensions,
            layer_names,
        );

        // Creation of an optional debug reporter
        cfg_if::cfg_if! {
            if #[cfg(debug_assertions)] {
                let debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vk_debug_callback));
                let debug_utils_fn = ext::DebugUtils::new(&entry_fn, &instance);
                let debug_utils_messenger = unsafe {
                    debug_utils_fn
                        .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
                        .unwrap()
                };
            }
        }

        // Creating the surface based on os
        let surface = unsafe {
            match window_display_handle {
                Some((RawWindowHandle::Win32(window), RawDisplayHandle::Windows(display))) => {
                    let surface_desc = vk::Win32SurfaceCreateInfoKHR::builder()
                        .hinstance(window.hinstance)
                        .hwnd(window.hwnd);
                    let win_surface_fn = khr::Win32Surface::new(&entry_fn, &instance);
                    win_surface_fn
                        .create_win32_surface(&surface_desc, None)
                        .unwrap()
                }
                Some((RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display))) => {
                    let surface_desc = vk::XlibSurfaceCreateInfoKHR::builder()
                        .dpy(display.display as *mut _)
                        .window(window.window);
                    let xlib_surface_fn = khr::XlibSurface::new(&entry_fn, &instance);
                    xlib_surface_fn
                        .create_xlib_surface(&surface_desc, None)
                        .unwrap()
                }
                Some((RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display))) => {
                    let surface_desc = vk::WaylandSurfaceCreateInfoKHR::builder()
                        .display(display.display)
                        .surface(window.surface);
                    let wayland_surface_fn = khr::WaylandSurface::new(&entry_fn, &instance);
                    wayland_surface_fn
                        .create_wayland_surface(&surface_desc, None)
                        .unwrap()
                }
                None => vk::SurfaceKHR::null(),
                _ => panic!("Unsupported window handle"),
            }
        };

        let mut device_extensions = Vec::from(device_extensions);
        let surface_fn = match surface != vk::SurfaceKHR::null() {
            true => {
                device_extensions.push("VK_KHR_swapchain");
                Some(khr::Surface::new(&entry_fn, &instance))
            }
            false => None,
        };

        let desired_device_extensions_cptr = Self::get_cptr_vec_from_str_slice(&device_extensions);

        let queues_types = &desired_queues_with_priorities
            .iter()
            .map(|q| q.0)
            .collect::<Vec<_>>();
        let good_devices = Self::filter_good_physical_devices(
            &instance,
            desired_physical_device_features2,
            &desired_device_extensions_cptr.1,
            queues_types,
            surface_fn.as_ref(),
            surface,
        );
        if good_devices.len() > 1 {
            println!("More than one device available selecting the first");
        }
        // Always selecting the first available device might not be the best strategy
        let selected_device = good_devices.first().expect("No suitable device found");

        // Device creation
        let device;
        unsafe {
            let queue_priorities = desired_queues_with_priorities
                .iter()
                .map(|q| q.1)
                .collect::<Vec<_>>();
            let queues_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(selected_device.1)
                .queue_priorities(&queue_priorities)
                .build();
            let mut device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queues_create_info))
                .enabled_extension_names(&desired_device_extensions_cptr.0)
                .enabled_features(&desired_physical_device_features2.features);
            device_create_info.p_next = desired_physical_device_features2.p_next;

            device = instance
                .create_device(selected_device.0, &device_create_info, None)
                .expect("Error creating device");
        }

        let swapchain_fn = match window_display_handle.is_some() {
            true => Some(khr::Swapchain::new(&instance, &device)),
            false => None,
        };

        let queues = (0..desired_queues_with_priorities.len() as u32)
            .map(|i| unsafe { device.get_device_queue(selected_device.1, i) })
            .collect::<Vec<_>>();

        VkBase {
            entry_fn,
            instance,
            surface,
            surface_fn,
            physical_device: selected_device.0,
            queue_family_index: selected_device.1,
            device,
            queues,
            swapchain_fn,
            swapchain_create_info: None,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: Vec::default(),
            swapchain_image_views: Vec::default(),
            #[cfg(debug_assertions)]
            debug_utils_fn,
            #[cfg(debug_assertions)]
            debug_utils_messenger,
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        present_mode: vk::PresentModeKHR,
        window_size: vk::Extent2D,
        usage_flags: vk::ImageUsageFlags,
        surface_format: vk::SurfaceFormatKHR,
    ) {
        self.swapchain_create_info = Some(
            vk::SwapchainCreateInfoKHR::builder()
                .image_array_layers(1)
                .surface(self.surface)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .clipped(true)
                .old_swapchain(self.swapchain)
                .build(),
        );
        let swapchain_create_info_ref = self.swapchain_create_info.as_mut().unwrap();
        let surface_capabilities;
        unsafe {
            // getting the present mode for the swapchain
            swapchain_create_info_ref.present_mode = *self
                .surface_fn
                .as_ref()
                .expect("BaseVk has not been created with surface support")
                .borrow()
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .unwrap()
                .iter()
                .find(|m| **m == present_mode)
                .unwrap_or(&vk::PresentModeKHR::FIFO);

            surface_capabilities = self
                .surface_fn
                .as_ref()
                .unwrap()
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap();
        }

        // getting the image count for the swapchain
        swapchain_create_info_ref.min_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count != 0 {
            swapchain_create_info_ref.min_image_count = std::cmp::min(
                swapchain_create_info_ref.min_image_count,
                surface_capabilities.max_image_count,
            );
        }

        // getting the extent of the images for the swapchain
        if surface_capabilities.current_extent.width == 0xFFFFFFFF
            && surface_capabilities.current_extent.height == 0xFFFFFFFF
        {
            swapchain_create_info_ref.image_extent.width = num::clamp(
                window_size.width,
                surface_capabilities.min_image_extent.width,
                surface_capabilities.max_image_extent.width,
            );

            swapchain_create_info_ref.image_extent.height = num::clamp(
                window_size.height,
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.height,
            );
        } else {
            swapchain_create_info_ref.image_extent = surface_capabilities.current_extent;
        }

        // checking if the usage flags are supported
        if !surface_capabilities
            .supported_usage_flags
            .contains(usage_flags)
        {
            panic!("Unsupported image usage flags")
        }
        swapchain_create_info_ref.image_usage = usage_flags;

        // checking if the surface format is supported or a substitute needs to be selected
        unsafe {
            let supported_formats = self
                .surface_fn
                .as_ref()
                .unwrap()
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .unwrap();

            let chosen_format = supported_formats
                .iter()
                .find(|e| **e == surface_format)
                .unwrap_or_else(|| supported_formats.first().unwrap());

            swapchain_create_info_ref.image_format = chosen_format.format;
            swapchain_create_info_ref.image_color_space = chosen_format.color_space;

            self.swapchain = self
                .swapchain_fn
                .as_ref()
                .unwrap()
                .create_swapchain(&self.swapchain_create_info.unwrap(), None)
                .expect("Could not create swapchain");

            self.swapchain_images = self
                .swapchain_fn
                .as_ref()
                .unwrap()
                .get_swapchain_images(self.swapchain)
                .unwrap();

            self.swapchain_image_views
                .drain(..)
                .for_each(|siv| self.device.destroy_image_view(siv, None));
            self.swapchain_image_views = self
                .swapchain_images
                .iter()
                .copied()
                .map(|swapchain_image| {
                    let image_view_create_info = vk::ImageViewCreateInfo::builder()
                        .image(swapchain_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.swapchain_create_info.unwrap().image_format)
                        .components(vk::ComponentMapping::default())
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        );
                    self.device
                        .create_image_view(&image_view_create_info, None)
                        .unwrap()
                })
                .collect::<Vec<_>>();
        }
    }

    pub fn get_instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn get_physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn get_device(&self) -> &ash::Device {
        &self.device
    }

    pub fn get_queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn get_queues(&self) -> &[vk::Queue] {
        self.queues.as_slice()
    }

    pub fn get_swapchain_fn(&self) -> &khr::Swapchain {
        self.swapchain_fn
            .as_ref()
            .expect("Swapchain support is not enabled")
    }

    pub fn get_swapchain_create_info(&self) -> &vk::SwapchainCreateInfoKHR {
        self.swapchain_create_info
            .as_ref()
            .expect("Swapchain support is not enabled")
    }

    pub fn get_swapchain(&self) -> Option<vk::SwapchainKHR> {
        if self.swapchain == vk::SwapchainKHR::null() {
            None
        } else {
            Some(self.swapchain)
        }
    }

    pub fn get_swapchain_images(&self) -> &[vk::Image] {
        &self.swapchain_images
    }

    pub fn get_swapchain_image_views(&self) -> &[vk::ImageView] {
        &self.swapchain_image_views
    }

    fn create_instance(
        entry_fn: &ash::Entry,
        application_name: &str,
        instance_pnext: *const std::ffi::c_void,
        desired_instance_extensions: &[&str],
        desired_layer_names: &[&str],
    ) -> ash::Instance {
        let application_name = CString::new(application_name).unwrap();
        let application_info = vk::ApplicationInfo::builder()
            .application_name(application_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(CStr::from_bytes_with_nul(b"TheVulkanTemple\0").unwrap())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let cstr_layer_names = Self::get_cptr_vec_from_str_slice(desired_layer_names);
        let cstr_extension_names = Self::get_cptr_vec_from_str_slice(desired_instance_extensions);

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&cstr_layer_names.0)
            .enabled_extension_names(&cstr_extension_names.0);
        instance_create_info.p_next = instance_pnext;

        unsafe {
            entry_fn
                .create_instance(&instance_create_info, None)
                .expect("Could not create VkInstance")
        }
    }

    fn filter_good_physical_devices(
        instance: &ash::Instance,
        desired_physical_device_features2: &vk::PhysicalDeviceFeatures2,
        desired_device_extensions: &[CString],
        desired_queues: &[vk::QueueFlags],
        surface_fn: Option<&khr::Surface>,
        surface: vk::SurfaceKHR,
    ) -> Vec<(vk::PhysicalDevice, u32)> {
        let mut available_device_features = unsafe {
            clone_vk_physical_device_features2_structure(desired_physical_device_features2)
        };

        let good_devices;
        unsafe {
            good_devices = instance
                .enumerate_physical_devices()
                .unwrap()
                .iter()
                .filter_map(|physical_device| {
                    // Check if the physical device supports the required extensions
                    let extensions = instance
                        .enumerate_device_extension_properties(*physical_device)
                        .unwrap();
                    let extensions_names: HashSet<&CStr> = extensions
                        .iter()
                        .map(|v| CStr::from_ptr(v.extension_name.as_ptr()))
                        .collect();
                    if !desired_device_extensions
                        .iter()
                        .all(|e| extensions_names.contains(e.as_c_str()))
                    {
                        return None;
                    }

                    // Check if the physical device supports the features requested
                    instance.get_physical_device_features2(
                        *physical_device,
                        &mut available_device_features,
                    );
                    if !compare_vk_physical_device_features2(
                        &available_device_features,
                        desired_physical_device_features2,
                    ) {
                        return None;
                    }

                    // Check if the physical device supports the requested queues
                    let mut queue_family_properties = Vec::<vk::QueueFamilyProperties2>::new();
                    queue_family_properties.resize(
                        instance.get_physical_device_queue_family_properties2_len(*physical_device),
                        vk::QueueFamilyProperties2::default(),
                    );
                    instance.get_physical_device_queue_family_properties2(
                        *physical_device,
                        &mut queue_family_properties,
                    );
                    let good_family_queues =
                        queue_family_properties
                            .iter()
                            .enumerate()
                            .find(|(i, queue_family)| {
                                let mut is_family_queue_good = desired_queues.iter().all(|q| {
                                    queue_family
                                        .queue_family_properties
                                        .queue_flags
                                        .contains(*q)
                                });
                                is_family_queue_good &= desired_queues.len()
                                    <= queue_family.queue_family_properties.queue_count as usize;

                                if surface != vk::SurfaceKHR::null() {
                                    is_family_queue_good &= surface_fn
                                        .as_ref()
                                        .expect("surface_fn is None")
                                        .get_physical_device_surface_support(
                                            *physical_device,
                                            *i as u32,
                                            surface,
                                        )
                                        .unwrap();
                                }
                                is_family_queue_good
                            });

                    if let Some(selected_family_queue) = good_family_queues {
                        return Some((*physical_device, selected_family_queue.0 as u32));
                    }
                    None
                })
                .collect::<Vec<(vk::PhysicalDevice, u32)>>();
            destroy_vk_physical_device_features2(&mut available_device_features);
        }
        good_devices
    }

    fn get_cptr_vec_from_str_slice(input: &[&str]) -> (Vec<*const c_char>, Vec<CString>) {
        let input_cstr_vec: Vec<CString> =
            input.iter().map(|s| CString::new(*s).unwrap()).collect();
        let input_cptr_vec = input_cstr_vec
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();
        (input_cptr_vec, input_cstr_vec)
    }
}

impl Drop for VkBase {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_image_views
                .drain(..)
                .for_each(|swapchain_image_view| {
                    self.device.destroy_image_view(swapchain_image_view, None);
                });
            if let Some(fp) = self.swapchain_fn.as_ref() {
                fp.destroy_swapchain(self.swapchain, None);
            }
            self.device.destroy_device(None);
            if let Some(fp) = self.surface_fn.as_ref() {
                fp.destroy_surface(self.surface, None);
            }
            #[cfg(debug_assertions)]
            self.debug_utils_fn
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
