#![allow(non_snake_case)]
#![allow(unused_variables)]

use ash::vk;
use std::alloc::Layout;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;

pub unsafe fn clone_vk_physical_device_features2_structure(
    source: &vk::PhysicalDeviceFeatures2,
) -> vk::PhysicalDeviceFeatures2 {
    let mut ret_val = vk::PhysicalDeviceFeatures2::default();

    let mut source_ptr = source.p_next;
    let mut dst_ptr = &mut (ret_val.p_next);

    macro_rules! allocate_struct {
        (match $s_type:expr; {
            $( $struct_identifier:pat => $struct_type:ty ),*
            $(,)?
        }) => {{
            match $s_type {
                $( $struct_identifier => allocate_struct!($s_type, $struct_type), )*
                _ => panic!("Found unrecognized struct inside clone_vk_physical_device_features2"),
            }
        }};

        ($struct_identifier:expr, $struct_type:ty) => {{
            let cloned_child_struct_ptr = std::alloc::alloc(Layout::new::<$struct_type>());
            (*(cloned_child_struct_ptr as *mut $struct_type)).s_type = $struct_identifier;
            cloned_child_struct_ptr
        }};
    }

    while !source_ptr.is_null() {
        let cloned_child_struct_ptr = allocate_struct!(match (*(source_ptr as *const vk::PhysicalDeviceFeatures2)).s_type; {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => vk::PhysicalDeviceVulkan11Features,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => vk::PhysicalDeviceVulkan12Features,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => vk::PhysicalDeviceVulkan13Features,
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT => vk::PhysicalDeviceDescriptorIndexingFeatures,
            vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR => vk::PhysicalDeviceSynchronization2FeaturesKHR,
            vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES => vk::PhysicalDeviceImagelessFramebufferFeatures,
            vk::StructureType::PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR => vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
            vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR => vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        });

        (*(cloned_child_struct_ptr as *mut vk::PhysicalDeviceVulkan11Features)).p_next = null_mut();
        *dst_ptr = cloned_child_struct_ptr as *mut c_void;
        dst_ptr = &mut ((*((*dst_ptr) as *mut vk::PhysicalDeviceFeatures2)).p_next);
        source_ptr = (*(source_ptr as *const vk::PhysicalDeviceFeatures2)).p_next;
    }
    ret_val
}

pub unsafe fn destroy_vk_physical_device_features2(source: &mut vk::PhysicalDeviceFeatures2) {
    let mut p_next = source.p_next;

    macro_rules! free_struct_and_advance {
        (match $s_type:expr; {
            $( $feature:pat => $struct_type:ty ),*
            $(,)?
        }) => {
            match $s_type {
                $( $feature => free_struct_and_advance!($struct_type), )*
                _ => panic!("Found unrecognized struct inside destroy_vk_physical_device_features2"),
            }
        };

        ($struct_type:ty) => {{
            let p_next_tmp = p_next;
            p_next = (*(p_next as *const $struct_type)).p_next;
            std::alloc::dealloc(
                p_next_tmp as *mut u8,
                Layout::new::<$struct_type>(),
            );
        }};
    }

    while !p_next.is_null() {
        free_struct_and_advance!(match (*(p_next as *const vk::PhysicalDeviceFeatures2)).s_type; {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => vk::PhysicalDeviceVulkan11Features,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => vk::PhysicalDeviceVulkan12Features,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => vk::PhysicalDeviceVulkan13Features,
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT => vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
            vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR => vk::PhysicalDeviceSynchronization2FeaturesKHR,
            vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES => vk::PhysicalDeviceImagelessFramebufferFeatures,
            vk::StructureType::PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR => vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
            vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR => vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        });
    }
    source.p_next = null_mut();
}

pub unsafe fn compare_device_features_structs(
    baseline: *const c_void,
    desired: *const c_void,
    mut size: usize,
) -> bool {
    // casting the structure to a PhysicalDeviceFeatures2 struct to compare the struct identifier
    if (*(baseline as *const vk::PhysicalDeviceFeatures2)).s_type
        != (*(desired as *const vk::PhysicalDeviceFeatures2)).s_type
    {
        return false;
    }
    // then we know that the structs type are the same so we cast them to a view of u32
    // the offset has a 4 added to it because of struct padding
    let offset = size_of::<ash::vk::StructureType>() + 4 + size_of::<*mut c_void>();

    // struct at the end will have 4 more bytes due to the fact its size has to be divisible by the
    // largest member which in this case is size_of<*mut c_void> = 8
    size -= 4;

    let baseline_data = baseline.add(offset) as *const u32;
    let desired_data = desired.add(offset) as *const u32;
    for i in 0..((size - offset) / size_of::<vk::Bool32>()) {
        if *(desired_data.add(i)) > *(baseline_data.add(i)) {
            return false;
        }
    }
    true
}

pub unsafe fn compare_vk_physical_device_features2(
    baseline: &vk::PhysicalDeviceFeatures2,
    desired: &vk::PhysicalDeviceFeatures2,
) -> bool {
    if !compare_device_features_structs(
        baseline as *const vk::PhysicalDeviceFeatures2 as *const c_void,
        desired as *const vk::PhysicalDeviceFeatures2 as *const c_void,
        size_of::<vk::PhysicalDeviceFeatures2>(),
    ) {
        return false;
    }

    let mut baseline_ptr = baseline.p_next;
    let mut desired_ptr = desired.p_next;

    macro_rules! compare_structs {
        (match $s_type:expr; {
            $( $feature:pat => $struct_type:ty ),*
            $(,)?
        }) => {
            match $s_type {
                $( $feature => compare_structs!($struct_type), )*
                _ => panic!("Found unrecognized struct inside compare_vk_physical_device_features2"),
            }
        };

        ($struct_type:ty) => {{
            compare_device_features_structs(
                baseline_ptr,
                desired_ptr,
                size_of::<$struct_type>(),
            )
        }};
    }

    while !baseline_ptr.is_null() && !desired_ptr.is_null() {
        let res = compare_structs!(match (*(baseline_ptr as *const vk::PhysicalDeviceFeatures2)).s_type; {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => vk::PhysicalDeviceVulkan11Features,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => vk::PhysicalDeviceVulkan12Features,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => vk::PhysicalDeviceVulkan13Features,
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT => vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
            vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR => vk::PhysicalDeviceSynchronization2FeaturesKHR,
            vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES => vk::PhysicalDeviceImagelessFramebufferFeatures,
            vk::StructureType::PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR => vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
            vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR => vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        });

        if !res {
            return false;
        }
        baseline_ptr = (*(baseline_ptr as *const vk::PhysicalDeviceFeatures2)).p_next;
        desired_ptr = (*(desired_ptr as *const vk::PhysicalDeviceFeatures2)).p_next;
    }
    baseline_ptr.is_null() && desired_ptr.is_null()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_correctly() {
        let mut imageless_fb =
            vk::PhysicalDeviceImagelessFramebufferFeatures::builder().imageless_framebuffer(true);
        let mut sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let original_struct = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut sync2)
            .push_next(&mut imageless_fb);

        let cloned_struct =
            unsafe { clone_vk_physical_device_features2_structure(&original_struct) };

        let mut original_pnext = original_struct.p_next;
        let mut cloned_pnext = cloned_struct.p_next;

        while !original_pnext.is_null() && !cloned_pnext.is_null() {
            let next_original_struct =
                unsafe { *(original_pnext as *const vk::PhysicalDeviceFeatures2) };
            let next_cloned_struct =
                unsafe { *(cloned_pnext as *const vk::PhysicalDeviceFeatures2) };

            assert_eq!(next_original_struct.s_type, next_cloned_struct.s_type);

            original_pnext = next_original_struct.p_next;
            cloned_pnext = next_cloned_struct.p_next;
        }
        assert_eq!(original_pnext, null_mut());
        assert_eq!(cloned_pnext, null_mut());
    }

    #[test]
    #[should_panic]
    fn clone_with_unknown_struct() {
        // multiview is a struct that has not yet been added
        let mut multiview = vk::PhysicalDeviceMultiviewFeatures::builder().multiview(true);
        let mut sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let original_struct = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut sync2)
            .push_next(&mut multiview);
        let cloned_struct =
            unsafe { clone_vk_physical_device_features2_structure(&original_struct) };
    }

    #[test]
    fn destroy_correctly() {
        let mut sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let original_struct = vk::PhysicalDeviceFeatures2::builder().push_next(&mut sync2);

        let mut cloned_struct =
            unsafe { clone_vk_physical_device_features2_structure(&original_struct) };
        unsafe { destroy_vk_physical_device_features2(&mut cloned_struct) }
        assert_eq!(cloned_struct.p_next, null_mut());
    }

    #[test]
    fn compare_device_features_struct_compatible() {
        let requested = vk::PhysicalDeviceFeatures::builder()
            .robust_buffer_access(true)
            .build();
        let baseline = vk::PhysicalDeviceFeatures::builder()
            .robust_buffer_access(true)
            .build();
        let requested_ptr = &requested as *const vk::PhysicalDeviceFeatures as *const c_void;
        let baseline_ptr = &baseline as *const vk::PhysicalDeviceFeatures as *const c_void;

        assert!(unsafe {
            compare_device_features_structs(
                baseline_ptr,
                requested_ptr,
                size_of::<vk::PhysicalDeviceFeatures>(),
            )
        });
    }

    #[test]
    fn compare_device_features_struct_compatible_but_pointing_to_others() {
        let mut requested = vk::PhysicalDeviceFeatures2::builder().build();
        requested.features.robust_buffer_access = vk::TRUE;

        let mut sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let mut baseline = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut sync2)
            .build();
        baseline.features.robust_buffer_access = vk::TRUE;

        let requested_ptr = &requested as *const vk::PhysicalDeviceFeatures2 as *const c_void;
        let baseline_ptr = &baseline as *const vk::PhysicalDeviceFeatures2 as *const c_void;

        assert!(unsafe {
            compare_device_features_structs(
                baseline_ptr,
                requested_ptr,
                size_of::<vk::PhysicalDeviceFeatures2>(),
            )
        });
    }

    #[test]
    fn compare_device_features_struct_incompatible() {
        let requested = vk::PhysicalDeviceFeatures::builder()
            .robust_buffer_access(true)
            .build();
        let baseline = vk::PhysicalDeviceFeatures::builder()
            .robust_buffer_access(false)
            .build();
        let requested_ptr = &requested as *const vk::PhysicalDeviceFeatures as *const c_void;
        let baseline_ptr = &baseline as *const vk::PhysicalDeviceFeatures as *const c_void;

        assert!(!unsafe {
            compare_device_features_structs(
                baseline_ptr,
                requested_ptr,
                size_of::<vk::PhysicalDeviceFeatures>(),
            )
        });
    }

    #[test]
    fn compare_vk_physical_device_features2_struct_compatible() {
        let mut requested = vk::PhysicalDeviceFeatures2::default();
        requested.features.robust_buffer_access = vk::FALSE;
        let mut baseline = vk::PhysicalDeviceFeatures2::default();
        baseline.features.robust_buffer_access = vk::TRUE;

        assert!(unsafe { compare_vk_physical_device_features2(&baseline, &requested) });
    }

    #[test]
    fn compare_vk_physical_device_features2_struct_incompatible() {
        let mut requested = vk::PhysicalDeviceFeatures2::default();
        requested.features.robust_buffer_access = vk::TRUE;
        let mut baseline = vk::PhysicalDeviceFeatures2::default();
        baseline.features.robust_buffer_access = vk::FALSE;

        assert!(!unsafe { compare_vk_physical_device_features2(&baseline, &requested) });
    }

    #[test]
    fn compare_vk_physical_device_features2_struct_chained_compatible() {
        let mut requested_sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let mut requested = vk::PhysicalDeviceFeatures2::builder().push_next(&mut requested_sync2);
        requested.features.robust_buffer_access = vk::FALSE;

        let mut baseline_sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let mut baseline = vk::PhysicalDeviceFeatures2::builder().push_next(&mut baseline_sync2);
        baseline.features.robust_buffer_access = vk::TRUE;

        assert!(unsafe { compare_vk_physical_device_features2(&baseline, &requested) });
    }

    #[test]
    fn compare_vk_physical_device_features2_struct_chained_incompatible() {
        let mut requested_sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let mut requested = vk::PhysicalDeviceFeatures2::builder().push_next(&mut requested_sync2);
        requested.features.robust_buffer_access = vk::FALSE;

        let mut baseline_sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(false);
        let mut baseline = vk::PhysicalDeviceFeatures2::builder().push_next(&mut baseline_sync2);
        baseline.features.robust_buffer_access = vk::TRUE;

        assert!(!unsafe { compare_vk_physical_device_features2(&baseline, &requested) });
    }

    #[test]
    fn compare_vk_physical_device_features2_struct_chained_different_incompatible() {
        let mut requested_sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let mut requested = vk::PhysicalDeviceFeatures2::builder().push_next(&mut requested_sync2);
        requested.features.robust_buffer_access = vk::FALSE;

        let mut baseline = vk::PhysicalDeviceFeatures2::default();
        baseline.features.robust_buffer_access = vk::TRUE;

        assert!(!unsafe { compare_vk_physical_device_features2(&baseline, &requested) });
    }
}
