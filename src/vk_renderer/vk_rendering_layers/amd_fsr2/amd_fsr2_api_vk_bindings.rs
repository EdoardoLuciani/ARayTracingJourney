#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(warnings, unused)]

use ash::vk;

pub type PFN_vkEnumerateDeviceExtensionProperties = ::std::option::Option<
    unsafe extern "C" fn(
        physicalDevice: vk::PhysicalDevice,
        pLayerName: *const ::std::os::raw::c_char,
        pPropertyCount: *mut u32,
        pProperties: *mut vk::ExtensionProperties,
    ) -> vk::Result,
>;

pub type PFN_vkGetInstanceProcAddr = ::std::option::Option<
    unsafe extern "C" fn(
        instance: vk::Instance,
        pName: *const ::std::os::raw::c_char,
    ) -> fn(),
>;

#[doc = "< A bit indicating if the input color data provided is using a high-dynamic range."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE:
    FfxFsr2InitializationFlagBits = 1;
#[doc = "< A bit indicating if the motion vectors are rendered at display resolution."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS:
    FfxFsr2InitializationFlagBits = 2;
#[doc = "< A bit indicating that the motion vectors have the jittering pattern applied to them."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION:
    FfxFsr2InitializationFlagBits = 4;
#[doc = "< A bit indicating that the input depth buffer data provided is inverted [1..0]."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_DEPTH_INVERTED:
    FfxFsr2InitializationFlagBits = 8;
#[doc = "< A bit indicating that the input depth buffer data provided is using an infinite far plane."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_DEPTH_INFINITE:
    FfxFsr2InitializationFlagBits = 16;
#[doc = "< A bit indicating if automatic exposure should be applied to input color data."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_AUTO_EXPOSURE:
    FfxFsr2InitializationFlagBits = 32;
#[doc = "< A bit indicating that the application uses dynamic resolution scaling."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_DYNAMIC_RESOLUTION:
    FfxFsr2InitializationFlagBits = 64;
#[doc = "< A bit indicating that the backend should use 1D textures."]
pub const FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_TEXTURE1D_USAGE:
    FfxFsr2InitializationFlagBits = 128;
#[doc = " An enumeration of bit flags used when creating a"]
#[doc = " <c><i>FfxFsr2Context</i></c>. See <c><i>FfxFsr2ContextDescription</i></c>."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2InitializationFlagBits = ::std::os::raw::c_int;
#[doc = "< Indicates a resource is in the state to be used as UAV."]
pub const FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS: FfxResourceStates = 1;
#[doc = "< Indicates a resource is in the state to be read by compute shaders."]
pub const FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ: FfxResourceStates = 2;
#[doc = "< Indicates a resource is in the state to be used as source in a copy command."]
pub const FfxResourceStates_FFX_RESOURCE_STATE_COPY_SRC: FfxResourceStates = 4;
#[doc = "< Indicates a resource is in the state to be used as destination in a copy command."]
pub const FfxResourceStates_FFX_RESOURCE_STATE_COPY_DEST: FfxResourceStates = 8;
#[doc = "< Indicates a resource is in generic (slow) read state."]
pub const FfxResourceStates_FFX_RESOURCE_STATE_GENERIC_READ: FfxResourceStates = 6;
#[doc = " An enumeration of resource states."]
pub type FfxResourceStates = ::std::os::raw::c_int;
#[doc = " Typedef for error codes returned from functions in the FidelityFX SDK."]
pub type FfxErrorCode = i32;#[doc = "< The operation completed successfully."]
pub const FFX_OK: FfxErrorCode = 0;
#[doc = "< The operation failed due to an invalid pointer."]
pub const FFX_ERROR_INVALID_POINTER: FfxErrorCode = -2147483648;
#[doc = "< The operation failed due to an invalid alignment."]
pub const FFX_ERROR_INVALID_ALIGNMENT: FfxErrorCode = -2147483647;
#[doc = "< The operation failed due to an invalid size."]
pub const FFX_ERROR_INVALID_SIZE: FfxErrorCode = -2147483646;
#[doc = "< The end of the file was encountered."]
pub const FFX_EOF: FfxErrorCode = -2147483645;
#[doc = "< The operation failed because the specified path was invalid."]
pub const FFX_ERROR_INVALID_PATH: FfxErrorCode = -2147483644;
#[doc = "< The operation failed because end of file was reached."]
pub const FFX_ERROR_EOF: FfxErrorCode = -2147483643;
#[doc = "< The operation failed because of some malformed data."]
pub const FFX_ERROR_MALFORMED_DATA: FfxErrorCode = -2147483642;
#[doc = "< The operation failed because it ran out memory."]
pub const FFX_ERROR_OUT_OF_MEMORY: FfxErrorCode = -2147483641;
#[doc = "< The operation failed because the interface was not fully configured."]
pub const FFX_ERROR_INCOMPLETE_INTERFACE: FfxErrorCode = -2147483640;
#[doc = "< The operation failed because of an invalid enumeration value."]
pub const FFX_ERROR_INVALID_ENUM: FfxErrorCode = -2147483639;
#[doc = "< The operation failed because an argument was invalid."]
pub const FFX_ERROR_INVALID_ARGUMENT: FfxErrorCode = -2147483638;
#[doc = "< The operation failed because a value was out of range."]
pub const FFX_ERROR_OUT_OF_RANGE: FfxErrorCode = -2147483637;
#[doc = "< The operation failed because a device was null."]
pub const FFX_ERROR_NULL_DEVICE: FfxErrorCode = -2147483636;
#[doc = "< The operation failed because the backend API returned an error code."]
pub const FFX_ERROR_BACKEND_API_ERROR: FfxErrorCode = -2147483635;
#[doc = "< The operation failed because there was not enough memory."]
pub const FFX_ERROR_INSUFFICIENT_MEMORY: FfxErrorCode = -2147483634;
#[doc = "< A pass which prepares input colors for subsequent use."]

#[doc = " A typedef representing the graphics device."]
pub type FfxDevice = *mut ::std::os::raw::c_void;
#[doc = " A typedef representing a command list or command buffer."]
pub type FfxCommandList = *mut ::std::os::raw::c_void;
pub type FfxResourceType = ::std::os::raw::c_int;
#[doc = " An enumeration of surface formats."]
pub type FfxSurfaceFormat = ::std::os::raw::c_int;
#[doc = " An enumeration of surface dimensions."]
pub type FfxResourceFlags = ::std::os::raw::c_int;
#[doc = " An enumeration of all the quality modes supported by FidelityFX Super"]
#[doc = " Resolution 2 upscaling."]
#[doc = ""]
#[doc = " In order to provide a consistent user experience across multiple"]
#[doc = " applications which implement FSR2. It is strongly recommended that the"]
#[doc = " following preset scaling factors are made available through your"]
#[doc = " application's user interface."]
#[doc = ""]
#[doc = " If your application does not expose the notion of preset scaling factors"]
#[doc = " for upscaling algorithms (perhaps instead implementing a fixed ratio which"]
#[doc = " is immutable) or implementing a more dynamic scaling scheme (such as"]
#[doc = " dynamic resolution scaling), then there is no need to use these presets."]
#[doc = ""]
#[doc = " Please note that <c><i>FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> is"]
#[doc = " an optional mode which may introduce significant quality degradation in the"]
#[doc = " final image. As such it is recommended that you evaluate the final results"]
#[doc = " of using this scaling mode before deciding if you should include it in your"]
#[doc = " application."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2QualityMode = ::std::os::raw::c_int;
#[doc = " An enumeration of all supported shader models."]
pub type FfxShaderModel = ::std::os::raw::c_int;
#[doc = " An enumeration for different heap types"]
pub type FfxHeapType = ::std::os::raw::c_int;
#[doc = " An enumeration of resource usage."]
pub type FfxResourceUsage = ::std::os::raw::c_int;
#[doc = " An enumeration of all the passes which constitute the FSR2 algorithm."]
#[doc = ""]
#[doc = " FSR2 is implemented as a composite of several compute passes each"]
#[doc = " computing a key part of the final result. Each call to the"]
#[doc = " <c><i>FfxFsr2ScheduleGpuJobFunc</i></c> callback function will"]
#[doc = " correspond to a single pass included in <c><i>FfxFsr2Pass</i></c>. For a"]
#[doc = " more comprehensive description of each pass, please refer to the FSR2"]
#[doc = " reference documentation."]
#[doc = ""]
#[doc = " Please note in some cases e.g.: <c><i>FFX_FSR2_PASS_ACCUMULATE</i></c>"]
#[doc = " and <c><i>FFX_FSR2_PASS_ACCUMULATE_SHARPEN</i></c> either one pass or the"]
#[doc = " other will be used (they are mutually exclusive). The choice of which will"]
#[doc = " depend on the way the <c><i>FfxFsr2Context</i></c> is created and the"]
#[doc = " precise contents of <c><i>FfxFsr2DispatchParamters</i></c> each time a call"]
#[doc = " is made to <c><i>ffxFsr2ContextDispatch</i></c>."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2Pass = ::std::os::raw::c_int;
#[doc = " An enumeration of surface dimensions."]
pub type FfxResourceDimension = ::std::os::raw::c_int;
#[doc = " An enumeration of all resource view types."]
pub type FfxResourceViewType = ::std::os::raw::c_int;
#[doc = " The type of filtering to perform when reading a texture."]
pub type FfxFilterType = ::std::os::raw::c_int;
#[doc = " An enumeration of all supported shader models."]
pub type FfxGpuJobType = ::std::os::raw::c_int;
#[doc = " A typedef for a root signature."]
pub type FfxRootSignature = *mut ::std::os::raw::c_void;
#[doc = " A typedef for a pipeline state object."]
pub type FfxPipeline = *mut ::std::os::raw::c_void;

#[doc = " A structure encapsulating a 2-dimensional point, using 32bit unsigned integers."]
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct FfxDimensions2D {
    #[doc = "< The width of a 2-dimensional range."]
    pub width: u32,
    #[doc = "< The height of a 2-dimensional range."]
    pub height: u32,
}

#[doc = " A structure encapsulating a 2-dimensional set of floating point coordinates."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxFloatCoords2D {
    #[doc = "< The x coordinate of a 2-dimensional point."]
    pub x: f32,
    #[doc = "< The y coordinate of a 2-dimensional point."]
    pub y: f32,
}

#[doc = " A structure encapasulating a collection of device capabilities."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxDeviceCapabilities {
    #[doc = "< The minimum shader model supported by the device."]
    pub minimumSupportedShaderModel: FfxShaderModel,
    #[doc = "< The minimum supported wavefront width."]
    pub waveLaneCountMin: u32,
    #[doc = "< The maximum supported wavefront width."]
    pub waveLaneCountMax: u32,
    #[doc = "< The device supports FP16 in hardware."]
    pub fp16Supported: bool,
    #[doc = "< The device supports raytracing."]
    pub raytracingSupported: bool,
}

#[doc = " A structure containing the data required to create a resource."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxCreateResourceDescription {
    #[doc = "< The heap type to hold the resource, typically <c><i>FFX_HEAP_TYPE_DEFAULT</i></c>."]
    pub heapType: FfxHeapType,
    #[doc = "< A resource description."]
    pub resourceDescription: FfxResourceDescription,
    #[doc = "< The initial resource state."]
    pub initalState: FfxResourceStates,
    #[doc = "< Size of initial data buffer."]
    pub initDataSize: u32,
    #[doc = "< Buffer containing data to fill the resource."]
    pub initData: *mut ::std::os::raw::c_void,
    #[doc = "< Name of the resource."]
    pub name: *const std::os::raw::c_ushort,
    #[doc = "< Resource usage flags."]
    pub usage: FfxResourceUsage,
    #[doc = "< Internal resource ID."]
    pub id: u32,
}

#[doc = " An internal structure containing a handle to a resource and resource views"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxResourceInternal {
    #[doc = "< The index of the resource."]
    pub internalIndex: i32,
}

#[doc = " A structure containing the description used to create a"]
#[doc = " <c><i>FfxPipeline</i></c> structure."]
#[doc = ""]
#[doc = " A pipeline is the name given to a shader and the collection of state that"]
#[doc = " is required to dispatch it. In the context of FSR2 and its architecture"]
#[doc = " this means that a <c><i>FfxPipelineDescription</i></c> will map to either a"]
#[doc = " monolithic object in an explicit API (such as a"]
#[doc = " <c><i>PipelineStateObject</i></c> in DirectX 12). Or a shader and some"]
#[doc = " ancillary API objects (in something like DirectX 11)."]
#[doc = ""]
#[doc = " The <c><i>contextFlags</i></c> field contains a copy of the flags passed"]
#[doc = " to <c><i>ffxFsr2ContextCreate</i></c> via the <c><i>flags</i></c> field of"]
#[doc = " the <c><i>FfxFsr2InitializationParams</i></c> structure. These flags are"]
#[doc = " used to determine which permutation of a pipeline for a specific"]
#[doc = " <c><i>FfxFsr2Pass</i></c> should be used to implement the features required"]
#[doc = " by each application, as well as to acheive the best performance on specific"]
#[doc = " target hardware configurations."]
#[doc = ""]
#[doc = " When using one of the provided backends for FSR2 (such as DirectX 12 or"]
#[doc = " Vulkan) the data required to create a pipeline is compiled offline and"]
#[doc = " included into the backend library that you are using. For cases where the"]
#[doc = " backend interface is overriden by providing custom callback function"]
#[doc = " implementations care should be taken to respect the contents of the"]
#[doc = " <c><i>contextFlags</i></c> field in order to correctly support the options"]
#[doc = " provided by FSR2, and acheive best performance."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxPipelineDescription {
    #[doc = "< A collection of <c><i>FfxFsr2InitializationFlagBits</i></c> which were passed to the context."]
    pub contextFlags: u32,
    #[doc = "< Array of static samplers."]
    pub samplers: *mut FfxFilterType,
    #[doc = "< The number of samples contained inside <c><i>samplers</i></c>."]
    pub samplerCount: std::os::raw::c_ulonglong,
    #[doc = "< Array containing the sizes of the root constant buffers (count of 32 bit elements)."]
    pub rootConstantBufferSizes: *const u32,
    #[doc = "< The number of root constants contained within <c><i>rootConstantBufferSizes</i></c>."]
    pub rootConstantBufferCount: u32,
}

#[doc = " A structure defining a resource bind point"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxResourceBinding {
    pub slotIndex: u32,
    pub resourceIdentifier: u32,
    pub name: [std::os::raw::c_ushort; 64usize],
}

#[doc = " A structure encapsulating a single pass of an algorithm."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxPipelineState {
    #[doc = "< The pipelines rootSignature"]
    pub rootSignature: FfxRootSignature,
    #[doc = "< The pipeline object"]
    pub pipeline: FfxPipeline,
    #[doc = "< Count of UAVs used in this pipeline"]
    pub uavCount: u32,
    #[doc = "< Count of SRVs used in this pipeline"]
    pub srvCount: u32,
    #[doc = "< Count of constant buffers used in this pipeline"]
    pub constCount: u32,
    #[doc = "< Array of ResourceIdentifiers bound as UAVs"]
    pub uavResourceBindings: [FfxResourceBinding; 8usize],
    #[doc = "< Array of ResourceIdentifiers bound as SRVs"]
    pub srvResourceBindings: [FfxResourceBinding; 16usize],
    #[doc = "< Array of ResourceIdentifiers bound as CBs"]
    pub cbResourceBindings: [FfxResourceBinding; 2usize],
}

#[doc = " A structure describing a clear render job."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxClearFloatJobDescription {
    #[doc = "< The clear color of the resource."]
    pub color: [f32; 4usize],
    #[doc = "< The resource to be cleared."]
    pub target: FfxResourceInternal,
}

#[doc = " A structure describing a copy render job."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxCopyJobDescription {
    #[doc = "< Source resource for the copy."]
    pub src: FfxResourceInternal,
    #[doc = "< Destination resource for the copy."]
    pub dst: FfxResourceInternal,
}

#[doc = " A structure containing a constant buffer."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxConstantBuffer {
    #[doc = "< Size of 32 bit chunks used in the constant buffer"]
    pub uint32Size: u32,
    #[doc = "< Constant buffer data"]
    pub data: [u32; 64usize],
}

#[doc = " A structure describing a compute render job."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxComputeJobDescription {
    #[doc = "< Compute pipeline for the render job."]
    pub pipeline: FfxPipelineState,
    #[doc = "< Dispatch dimensions."]
    pub dimensions: [u32; 3usize],
    #[doc = "< SRV resources to be bound in the compute job."]
    pub srvs: [FfxResourceInternal; 16usize],
    pub srvNames: [[std::os::raw::c_ushort; 64usize]; 16usize],
    #[doc = "< UAV resources to be bound in the compute job."]
    pub uavs: [FfxResourceInternal; 8usize],
    #[doc = "< Mip level of UAV resources to be bound in the compute job."]
    pub uavMip: [u32; 8usize],
    pub uavNames: [[std::os::raw::c_ushort; 64usize]; 8usize],
    #[doc = "< Constant buffers to be bound in the compute job."]
    pub cbs: [FfxConstantBuffer; 2usize],
    pub cbNames: [[std::os::raw::c_ushort; 64usize]; 2usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union FfxGpuJobDescription__bindgen_ty_1 {
    #[doc = "< Clear job descriptor. Valid when <c><i>jobType</i></c> is <c><i>FFX_RENDER_JOB_CLEAR_FLOAT</i></c>."]
    pub clearJobDescriptor: FfxClearFloatJobDescription,
    #[doc = "< Copy job descriptor. Valid when <c><i>jobType</i></c> is <c><i>FFX_RENDER_JOB_COPY</i></c>."]
    pub copyJobDescriptor: FfxCopyJobDescription,
    #[doc = "< Compute job descriptor. Valid when <c><i>jobType</i></c> is <c><i>FFX_RENDER_JOB_COMPUTE</i></c>."]
    pub computeJobDescriptor: FfxComputeJobDescription,
}

#[doc = " A structure describing a single render job."]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct FfxGpuJobDescription {
    #[doc = "< Type of the job."]
    pub jobType: FfxGpuJobType,
    pub __bindgen_anon_1: FfxGpuJobDescription__bindgen_ty_1,
}

#[doc = " Create and initialize the backend context."]
#[doc = ""]
#[doc = " The callback function sets up the backend context for rendering."]
#[doc = " It will create or reference the device and create required internal data structures."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] device                              The FfxDevice obtained by ffxGetDevice(DX12/VK/...)."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2CreateBackendContextFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        device: FfxDevice,
    ) -> FfxErrorCode,
>;
#[doc = " Get a list of capabilities of the device."]
#[doc = ""]
#[doc = " When creating an <c><i>FfxFsr2Context</i></c> it is desirable for the FSR2"]
#[doc = " core implementation to be aware of certain characteristics of the platform"]
#[doc = " that is being targetted. This is because some optimizations which FSR2"]
#[doc = " attempts to perform are more effective on certain classes of hardware than"]
#[doc = " others, or are not supported by older hardware. In order to avoid cases"]
#[doc = " where optimizations actually have the effect of decreasing performance, or"]
#[doc = " reduce the breadth of support provided by FSR2, FSR2 queries the"]
#[doc = " capabilities of the device to make such decisions."]
#[doc = ""]
#[doc = " For target platforms with fixed hardware support you need not implement"]
#[doc = " this callback function by querying the device, but instead may hardcore"]
#[doc = " what features are available on the platform."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [out] outDeviceCapabilities              The device capabilities structure to fill out."]
#[doc = " @param [in] device                              The device to query for capabilities."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2GetDeviceCapabilitiesFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        outDeviceCapabilities: *mut FfxDeviceCapabilities,
        device: FfxDevice,
    ) -> FfxErrorCode,
>;
#[doc = " Destroy the backend context and dereference the device."]
#[doc = ""]
#[doc = " This function is called when the <c><i>FfxFsr2Context</i></c> is destroyed."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2DestroyBackendContextFunc = ::std::option::Option<
    unsafe extern "C" fn(backendInterface: *mut FfxFsr2Interface) -> FfxErrorCode,
>;
#[doc = " Create a resource."]
#[doc = ""]
#[doc = " This callback is intended for the backend to create internal resources."]
#[doc = ""]
#[doc = " Please note: It is also possible that the creation of resources might"]
#[doc = " itself cause additional resources to be created by simply calling the"]
#[doc = " <c><i>FfxFsr2CreateResourceFunc</i></c> function pointer again. This is"]
#[doc = " useful when handling the initial creation of resources which must be"]
#[doc = " initialized. The flow in such a case would be an initial call to create the"]
#[doc = " CPU-side resource, another to create the GPU-side resource, and then a call"]
#[doc = " to schedule a copy render job to move the data between the two. Typically"]
#[doc = " this type of function call flow is only seen during the creation of an"]
#[doc = " <c><i>FfxFsr2Context</i></c>."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] createResourceDescription           A pointer to a <c><i>FfxCreateResourceDescription</i></c>."]
#[doc = " @param [out] outResource                        A pointer to a <c><i>FfxResource</i></c> object."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2CreateResourceFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        createResourceDescription: *const FfxCreateResourceDescription,
        outResource: *mut FfxResourceInternal,
    ) -> FfxErrorCode,
>;
#[doc = " Register a resource in the backend for the current frame."]
#[doc = ""]
#[doc = " Since FSR2 and the backend are not aware how many different"]
#[doc = " resources will get passed to FSR2 over time, it's not safe"]
#[doc = " to register all resources simultaneously in the backend."]
#[doc = " Also passed resources may not be valid after the dispatch call."]
#[doc = " As a result it's safest to register them as FfxResourceInternal"]
#[doc = " and clear them at the end of the dispatch call."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] inResource                          A pointer to a <c><i>FfxResource</i></c>."]
#[doc = " @param [out] outResource                        A pointer to a <c><i>FfxResourceInternal</i></c> object."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2RegisterResourceFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        inResource: *const FfxResource,
        outResource: *mut FfxResourceInternal,
    ) -> FfxErrorCode,
>;
#[doc = " Unregister all temporary FfxResourceInternal from the backend."]
#[doc = ""]
#[doc = " Unregister FfxResourceInternal referencing resources passed to"]
#[doc = " a function as a parameter."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2UnregisterResourcesFunc = ::std::option::Option<
    unsafe extern "C" fn(backendInterface: *mut FfxFsr2Interface) -> FfxErrorCode,
>;
#[doc = " Retrieve a <c><i>FfxResourceDescription</i></c> matching a"]
#[doc = " <c><i>FfxResource</i></c> structure."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] resource                            A pointer to a <c><i>FfxResource</i></c> object."]
#[doc = ""]
#[doc = " @returns"]
#[doc = " A description of the resource."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2GetResourceDescriptionFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        resource: FfxResourceInternal,
    ) -> FfxResourceDescription,
>;
#[doc = " Destroy a resource"]
#[doc = ""]
#[doc = " This callback is intended for the backend to release an internal resource."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] resource                            A pointer to a <c><i>FfxResource</i></c> object."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2DestroyResourceFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        resource: FfxResourceInternal,
    ) -> FfxErrorCode,
>;
#[doc = " Create a render pipeline."]
#[doc = ""]
#[doc = " A rendering pipeline contains the shader as well as resource bindpoints"]
#[doc = " and samplers."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] pass                                The identifier for the pass."]
#[doc = " @param [in] pipelineDescription                 A pointer to a <c><i>FfxPipelineDescription</i></c> describing the pipeline to be created."]
#[doc = " @param [out] outPipeline                        A pointer to a <c><i>FfxPipelineState</i></c> structure which should be populated."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2CreatePipelineFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        pass: FfxFsr2Pass,
        pipelineDescription: *const FfxPipelineDescription,
        outPipeline: *mut FfxPipelineState,
    ) -> FfxErrorCode,
>;
#[doc = " Destroy a render pipeline."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [out] pipeline                           A pointer to a <c><i>FfxPipelineState</i></c> structure which should be released."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2DestroyPipelineFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        pipeline: *mut FfxPipelineState,
    ) -> FfxErrorCode,
>;
#[doc = " Schedule a render job to be executed on the next call of"]
#[doc = " <c><i>FfxFsr2ExecuteGpuJobsFunc</i></c>."]
#[doc = ""]
#[doc = " Render jobs can perform one of three different tasks: clear, copy or"]
#[doc = " compute dispatches."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] job                                 A pointer to a <c><i>FfxGpuJobDescription</i></c> structure."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2ScheduleGpuJobFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        job: *const FfxGpuJobDescription,
    ) -> FfxErrorCode,
>;
#[doc = " Execute scheduled render jobs on the <c><i>comandList</i></c> provided."]
#[doc = ""]
#[doc = " The recording of the graphics API commands should take place in this"]
#[doc = " callback function, the render jobs which were previously enqueued (via"]
#[doc = " callbacks made to <c><i>FfxFsr2ScheduleGpuJobFunc</i></c>) should be"]
#[doc = " processed in the order they were received. Advanced users might choose to"]
#[doc = " reorder the rendering jobs, but should do so with care to respect the"]
#[doc = " resource dependencies."]
#[doc = ""]
#[doc = " Depending on the precise contents of <c><i>FfxFsr2DispatchDescription</i></c> a"]
#[doc = " different number of render jobs might have previously been enqueued (for"]
#[doc = " example if sharpening is toggled on and off)."]
#[doc = ""]
#[doc = " @param [in] backendInterface                    A pointer to the backend interface."]
#[doc = " @param [in] commandList                         A pointer to a <c><i>FfxCommandList</i></c> structure."]
#[doc = ""]
#[doc = " @retval"]
#[doc = " FFX_OK                                          The operation completed successfully."]
#[doc = " @retval"]
#[doc = " Anything else                                   The operation failed."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
pub type FfxFsr2ExecuteGpuJobsFunc = ::std::option::Option<
    unsafe extern "C" fn(
        backendInterface: *mut FfxFsr2Interface,
        commandList: FfxCommandList,
    ) -> FfxErrorCode,
>;

#[doc = " A structure encapsulating the interface between the core implentation of"]
#[doc = " the FSR2 algorithm and any graphics API that it should ultimately call."]
#[doc = ""]
#[doc = " This set of functions serves as an abstraction layer between FSR2 and the"]
#[doc = " API used to implement it. While FSR2 ships with backends for DirectX12 and"]
#[doc = " Vulkan, it is possible to implement your own backend for other platforms or"]
#[doc = " which sits ontop of your engine's own abstraction layer. For details on the"]
#[doc = " expectations of what each function should do you should refer the"]
#[doc = " description of the following function pointer types:"]
#[doc = ""]
#[doc = "     <c><i>FfxFsr2CreateDeviceFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2GetDeviceCapabilitiesFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2DestroyDeviceFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2CreateResourceFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2GetResourceDescriptionFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2DestroyResourceFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2CreatePipelineFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2DestroyPipelineFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2ScheduleGpuJobFunc</i></c>"]
#[doc = "     <c><i>FfxFsr2ExecuteGpuJobsFunc</i></c>"]
#[doc = ""]
#[doc = " Depending on the graphics API that is abstracted by the backend, it may be"]
#[doc = " required that the backend is to some extent stateful. To ensure that"]
#[doc = " applications retain full control to manage the memory used by FSR2, the"]
#[doc = " <c><i>scratchBuffer</i></c> and <c><i>scratchBufferSize</i></c> fields are"]
#[doc = " provided. A backend should provide a means of specifying how much scratch"]
#[doc = " memory is required for its internal implementation (e.g: via a function"]
#[doc = " or constant value). The application is that responsible for allocating that"]
#[doc = " memory and providing it when setting up the FSR2 backend. Backends provided"]
#[doc = " with FSR2 do not perform dynamic memory allocations, and instead"]
#[doc = " suballocate all memory from the scratch buffers provided."]
#[doc = ""]
#[doc = " The <c><i>scratchBuffer</i></c> and <c><i>scratchBufferSize</i></c> fields"]
#[doc = " should be populated according to the requirements of each backend. For"]
#[doc = " example, if using the DirectX 12 backend you should call the"]
#[doc = " <c><i>ffxFsr2GetScratchMemorySizeDX12</i></c> function. It is not required"]
#[doc = " that custom backend implementations use a scratch buffer."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxFsr2Interface {
    #[doc = "< A callback function to create and initialize the backend context."]
    pub fpCreateBackendContext: FfxFsr2CreateBackendContextFunc,
    #[doc = "< A callback function to query device capabilites."]
    pub fpGetDeviceCapabilities: FfxFsr2GetDeviceCapabilitiesFunc,
    #[doc = "< A callback function to destroy the backendcontext. This also dereferences the device."]
    pub fpDestroyBackendContext: FfxFsr2DestroyBackendContextFunc,
    #[doc = "< A callback function to create a resource."]
    pub fpCreateResource: FfxFsr2CreateResourceFunc,
    #[doc = "< A callback function to register an external resource."]
    pub fpRegisterResource: FfxFsr2RegisterResourceFunc,
    #[doc = "< A callback function to unregister external resource."]
    pub fpUnregisterResources: FfxFsr2UnregisterResourcesFunc,
    #[doc = "< A callback function to retrieve a resource description."]
    pub fpGetResourceDescription: FfxFsr2GetResourceDescriptionFunc,
    #[doc = "< A callback function to destroy a resource."]
    pub fpDestroyResource: FfxFsr2DestroyResourceFunc,
    #[doc = "< A callback function to create a render or compute pipeline."]
    pub fpCreatePipeline: FfxFsr2CreatePipelineFunc,
    #[doc = "< A callback function to destroy a render or compute pipeline."]
    pub fpDestroyPipeline: FfxFsr2DestroyPipelineFunc,
    #[doc = "< A callback function to schedule a render job."]
    pub fpScheduleGpuJob: FfxFsr2ScheduleGpuJobFunc,
    #[doc = "< A callback function to execute all queued render jobs."]
    pub fpExecuteGpuJobs: FfxFsr2ExecuteGpuJobsFunc,
    #[doc = "< A preallocated buffer for memory utilized internally by the backend."]
    pub scratchBuffer: *mut ::std::os::raw::c_void,
    #[doc = "< Size of the buffer pointed to by <c><i>scratchBuffer</i></c>."]
    pub scratchBufferSize: std::os::raw::c_ulonglong,
}

#[doc = " A structure encapsulating the parameters required to initialize FidelityFX"]
#[doc = " Super Resolution 2 upscaling."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxFsr2ContextDescription {
    #[doc = "< A collection of <c><i>FfxFsr2InitializationFlagBits</i></c>."]
    pub flags: u32,
    #[doc = "< The maximum size that rendering will be performed at."]
    pub maxRenderSize: FfxDimensions2D,
    #[doc = "< The size of the presentation resolution targeted by the upscaling process."]
    pub displaySize: FfxDimensions2D,
    #[doc = "< A set of pointers to the backend implementation for FSR 2.0."]
    pub callbacks: FfxFsr2Interface,
    #[doc = "< The abstracted device which is passed to some callback functions."]
    pub device: FfxDevice,
}

#[doc = " A structure encapsulating the FidelityFX Super Resolution 2 context."]
#[doc = ""]
#[doc = " This sets up an object which contains all persistent internal data and"]
#[doc = " resources that are required by FSR2."]
#[doc = ""]
#[doc = " The <c><i>FfxFsr2Context</i></c> object should have a lifetime matching"]
#[doc = " your use of FSR2. Before destroying the FSR2 context care should be taken"]
#[doc = " to ensure the GPU is not accessing the resources created or used by FSR2."]
#[doc = " It is therefore recommended that the GPU is idle before destroying the"]
#[doc = " FSR2 context."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxFsr2Context {
    #[doc = "< An opaque set of <c>uint32_t</c> which contain the data for the context."]
    pub data: [u32; 16536usize],
}

#[doc = " A structure describing a resource."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxResourceDescription {
    #[doc = "< The type of the resource."]
    pub type_: FfxResourceType,
    #[doc = "< The surface format."]
    pub format: FfxSurfaceFormat,
    #[doc = "< The width of the resource."]
    pub width: u32,
    #[doc = "< The height of the resource."]
    pub height: u32,
    #[doc = "< The depth of the resource."]
    pub depth: u32,
    #[doc = "< Number of mips (or 0 for full mipchain)."]
    pub mipCount: u32,
    #[doc = "< A set of <c><i>FfxResourceFlags</i></c> flags."]
    pub flags: FfxResourceFlags,
}

#[doc = " An outward facing structure containing a resource"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxResource {
    #[doc = "< pointer to the resource."]
    pub resource: *mut ::std::os::raw::c_void,
    pub name: [std::os::raw::c_ushort; 64usize],
    pub description: FfxResourceDescription,
    pub state: FfxResourceStates,
    pub isDepth: bool,
    pub descriptorData: u64,
}

#[doc = " A structure encapsulating the parameters for dispatching the various passes"]
#[doc = " of FidelityFX Super Resolution 2."]
#[doc = ""]
#[doc = " @ingroup FSR2"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxFsr2DispatchDescription {
    #[doc = "< The <c><i>FfxCommandList</i></c> to record FSR2 rendering commands into."]
    pub commandList: FfxCommandList,
    #[doc = "< A <c><i>FfxResource</i></c> containing the color buffer for the current frame (at render resolution)."]
    pub color: FfxResource,
    #[doc = "< A <c><i>FfxResource</i></c> containing 32bit depth values for the current frame (at render resolution)."]
    pub depth: FfxResource,
    #[doc = "< A <c><i>FfxResource</i></c> containing 2-dimensional motion vectors (at render resolution if <c><i>FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS</i></c> is not set)."]
    pub motionVectors: FfxResource,
    #[doc = "< A optional <c><i>FfxResource</i></c> containing a 1x1 exposure value."]
    pub exposure: FfxResource,
    #[doc = "< A optional <c><i>FfxResource</i></c> containing alpha value of reactive objects in the scene."]
    pub reactive: FfxResource,
    #[doc = "< A optional <c><i>FfxResource</i></c> containing alpha value of special objects in the scene."]
    pub transparencyAndComposition: FfxResource,
    #[doc = "< A <c><i>FfxResource</i></c> containing the output color buffer for the current frame (at presentation resolution)."]
    pub output: FfxResource,
    #[doc = "< The subpixel jitter offset applied to the camera."]
    pub jitterOffset: FfxFloatCoords2D,
    #[doc = "< The scale factor to apply to motion vectors."]
    pub motionVectorScale: FfxFloatCoords2D,
    #[doc = "< The resolution that was used for rendering the input resources."]
    pub renderSize: FfxDimensions2D,
    #[doc = "< Enable an additional sharpening pass."]
    pub enableSharpening: bool,
    #[doc = "< The sharpness value between 0 and 1, where 0 is no additional sharpness and 1 is maximum additional sharpness."]
    pub sharpness: f32,
    #[doc = "< The time elapsed since the last frame (expressed in milliseconds)."]
    pub frameTimeDelta: f32,
    #[doc = "< The exposure value if not using <c><i>FFX_FSR2_ENABLE_AUTO_EXPOSURE</i></c>."]
    pub preExposure: f32,
    #[doc = "< A boolean value which when set to true, indicates the camera has moved discontinuously."]
    pub reset: bool,
    #[doc = "< The distance to the near plane of the camera."]
    pub cameraNear: f32,
    #[doc = "< The distance to the far plane of the camera. This is used only used in case of non infinite depth."]
    pub cameraFar: f32,
    #[doc = "< The camera angle field of view in the vertical direction (expressed in radians)."]
    pub cameraFovAngleVertical: f32,
}

#[doc = " A structure encapsulating the parameters for automatic generation of a reactive mask"]
#[doc = ""]
#[doc = " @ingroup FSR2"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxFsr2GenerateReactiveDescription {
    #[doc = "< The <c><i>FfxCommandList</i></c> to record FSR2 rendering commands into."]
    pub commandList: FfxCommandList,
    #[doc = "< A <c><i>FfxResource</i></c> containing the opaque only color buffer for the current frame (at render resolution)."]
    pub colorOpaqueOnly: FfxResource,
    #[doc = "< A <c><i>FfxResource</i></c> containing the opaque+translucent color buffer for the current frame (at render resolution)."]
    pub colorPreUpscale: FfxResource,
    #[doc = "< A <c><i>FfxResource</i></c> containing the surface to generate the reactive mask into."]
    pub outReactive: FfxResource,
    #[doc = "< The resolution that was used for rendering the input resources."]
    pub renderSize: FfxDimensions2D,
    #[doc = "< A value to scale the output"]
    pub scale: f32,
    #[doc = "< A threshold value to generate a binary reactive mask"]
    pub cutoffThreshold: f32,
    #[doc = "< A value to set for the binary reactive mask"]
    pub binaryValue: f32,
    #[doc = "< Flags to determine how to generate the reactive mask"]
    pub flags: u32,
}


extern "C" {
    #[doc = " Create a FidelityFX Super Resolution 2 context from the parameters"]
    #[doc = " programmed to the <c><i>FfxFsr2CreateParams</i></c> structure."]
    #[doc = ""]
    #[doc = " The context structure is the main object used to interact with the FSR2"]
    #[doc = " API, and is responsible for the management of the internal resources used"]
    #[doc = " by the FSR2 algorithm. When this API is called, multiple calls will be"]
    #[doc = " made via the pointers contained in the <c><i>callbacks</i></c> structure."]
    #[doc = " These callbacks will attempt to retreive the device capabilities, and"]
    #[doc = " create the internal resources, and pipelines required by FSR2's"]
    #[doc = " frame-to-frame function. Depending on the precise configuration used when"]
    #[doc = " creating the <c><i>FfxFsr2Context</i></c> a different set of resources and"]
    #[doc = " pipelines might be requested via the callback functions."]
    #[doc = ""]
    #[doc = " The flags included in the <c><i>flags</i></c> field of"]
    #[doc = " <c><i>FfxFsr2Context</i></c> how match the configuration of your"]
    #[doc = " application as well as the intended use of FSR2. It is important that these"]
    #[doc = " flags are set correctly (as well as a correct programmed"]
    #[doc = " <c><i>FfxFsr2DispatchDescription</i></c>) to ensure correct operation. It is"]
    #[doc = " recommended to consult the overview documentation for further details on"]
    #[doc = " how FSR2 should be integerated into an application."]
    #[doc = ""]
    #[doc = " When the <c><i>FfxFsr2Context</i></c> is created, you should use the"]
    #[doc = " <c><i>ffxFsr2ContextDispatch</i></c> function each frame where FSR2"]
    #[doc = " upscaling should be applied. See the documentation of"]
    #[doc = " <c><i>ffxFsr2ContextDispatch</i></c> for more details."]
    #[doc = ""]
    #[doc = " The <c><i>FfxFsr2Context</i></c> should be destroyed when use of it is"]
    #[doc = " completed, typically when an application is unloaded or FSR2 upscaling is"]
    #[doc = " disabled by a user. To destroy the FSR2 context you should call"]
    #[doc = " <c><i>ffxFsr2ContextDestroy</i></c>."]
    #[doc = ""]
    #[doc = " @param [out] context                A pointer to a <c><i>FfxFsr2Context</i></c> structure to populate."]
    #[doc = " @param [in]  contextDescription     A pointer to a <c><i>FfxFsr2ContextDescription</i></c> structure."]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                              The operation completed successfully."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>contextDescription</i></c> was <c><i>NULL</i></c>."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_INCOMPLETE_INTERFACE      The operation failed because the <c><i>FfxFsr2ContextDescription.callbacks</i></c>  was not fully specified."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2ContextCreate(
        context: *mut FfxFsr2Context,
        contextDescription: *const FfxFsr2ContextDescription,
    ) -> FfxErrorCode;

    #[doc = " Dispatch the various passes that constitute FidelityFX Super Resolution 2."]
    #[doc = ""]
    #[doc = " FSR2 is a composite effect, meaning that it is compromised of multiple"]
    #[doc = " constituent passes (implemented as one or more clears, copies and compute"]
    #[doc = " dispatches). The <c><i>ffxFsr2ContextDispatch</i></c> function is the"]
    #[doc = " function which (via the use of the functions contained in the"]
    #[doc = " <c><i>callbacks</i></c> field of the <c><i>FfxFsr2Context</i></c>"]
    #[doc = " structure) utlimately generates the sequence of graphics API calls required"]
    #[doc = " each frame."]
    #[doc = ""]
    #[doc = " As with the creation of the <c><i>FfxFsr2Context</i></c> correctly"]
    #[doc = " programming the <c><i>FfxFsr2DispatchDescription</i></c> is key to ensuring"]
    #[doc = " the correct operation of FSR2. It is particularly important to ensure that"]
    #[doc = " camera jitter is correctly applied to your application's projection matrix"]
    #[doc = " (or camera origin for raytraced applications). FSR2 provides the"]
    #[doc = " <c><i>ffxFsr2GetJitterPhaseCount</i></c> and"]
    #[doc = " <c><i>ffxFsr2GetJitterOffset</i></c> entry points to help applications"]
    #[doc = " correctly compute the camera jitter. Whatever jitter pattern is used by the"]
    #[doc = " application it should be correctly programmed to the"]
    #[doc = " <c><i>jitterOffset</i></c> field of the <c><i>dispatchDescription</i></c>"]
    #[doc = " structure. For more guidance on camera jitter please consult the"]
    #[doc = " documentation for <c><i>ffxFsr2GetJitterOffset</i></c> as well as the"]
    #[doc = " accompanying overview documentation for FSR2."]
    #[doc = ""]
    #[doc = " @param [in] context                 A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] dispatchDescription     A pointer to a <c><i>FfxFsr2DispatchDescription</i></c> structure."]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                              The operation completed successfully."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>dispatchDescription</i></c> was <c><i>NULL</i></c>."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_OUT_OF_RANGE              The operation failed because <c><i>dispatchDescription.renderSize</i></c> was larger than the maximum render resolution."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_NULL_DEVICE               The operation failed because the device inside the context was <c><i>NULL</i></c>."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2ContextDispatch(
        context: *mut FfxFsr2Context,
        dispatchDescription: *const FfxFsr2DispatchDescription,
    ) -> FfxErrorCode;

    #[doc = " A helper function generate a Reactive mask from an opaque only texure and one containing translucent objects."]
    #[doc = ""]
    #[doc = " @param [in] context                 A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] params                  A pointer to a <c><i>FfxFsr2GenerateReactiveDescription</i></c> structure"]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                              The operation completed successfully."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2ContextGenerateReactiveMask(
        context: *mut FfxFsr2Context,
        params: *const FfxFsr2GenerateReactiveDescription,
    ) -> FfxErrorCode;

    #[doc = " Destroy the FidelityFX Super Resolution context."]
    #[doc = ""]
    #[doc = " @param [out] context                A pointer to a <c><i>FfxFsr2Context</i></c> structure to destroy."]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                              The operation completed successfully."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> was <c><i>NULL</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2ContextDestroy(context: *mut FfxFsr2Context) -> FfxErrorCode;

    #[doc = " Get the upscale ratio from the quality mode."]
    #[doc = ""]
    #[doc = " The following table enumerates the mapping of the quality modes to"]
    #[doc = " per-dimension scaling ratios."]
    #[doc = ""]
    #[doc = " Quality preset                                        | Scale factor"]
    #[doc = " ----------------------------------------------------- | -------------"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_QUALITY</i></c>           | 1.5x"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_BALANCED</i></c>          | 1.7x"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> | 3.0x"]
    #[doc = ""]
    #[doc = " Passing an invalid <c><i>qualityMode</i></c> will return 0.0f."]
    #[doc = ""]
    #[doc = " @param [in] qualityMode             The quality mode preset."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " The upscaling the per-dimension upscaling ratio for"]
    #[doc = " <c><i>qualityMode</i></c> according to the table above."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2GetUpscaleRatioFromQualityMode(qualityMode: FfxFsr2QualityMode) -> f32;

    #[doc = " A helper function to calculate the rendering resolution from a target"]
    #[doc = " resolution and desired quality level."]
    #[doc = ""]
    #[doc = " This function applies the scaling factor returned by"]
    #[doc = " <c><i>ffxFsr2GetUpscaleRatioFromQualityMode</i></c> to each dimension."]
    #[doc = ""]
    #[doc = " @param [out] renderWidth            A pointer to a <c>uint32_t</c> which will hold the calculated render resolution width."]
    #[doc = " @param [out] renderHeight           A pointer to a <c>uint32_t</c> which will hold the calculated render resolution height."]
    #[doc = " @param [in] displayWidth            The target display resolution width."]
    #[doc = " @param [in] displayHeight           The target display resolution height."]
    #[doc = " @param [in] qualityMode             The desired quality mode for FSR 2 upscaling."]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                              The operation completed successfully."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_INVALID_POINTER           Either <c><i>renderWidth</i></c> or <c><i>renderHeight</i></c> was <c>NULL</c>."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_INVALID_ENUM              An invalid quality mode was specified."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2GetRenderResolutionFromQualityMode(
        renderWidth: *mut u32,
        renderHeight: *mut u32,
        displayWidth: u32,
        displayHeight: u32,
        qualityMode: FfxFsr2QualityMode,
    ) -> FfxErrorCode;

    #[doc = " A helper function to calculate the jitter phase count from display"]
    #[doc = " resolution."]
    #[doc = ""]
    #[doc = " For more detailed information about the application of camera jitter to"]
    #[doc = " your application's rendering please refer to the"]
    #[doc = " <c><i>ffxFsr2GetJitterOffset</i></c> function."]
    #[doc = ""]
    #[doc = " The table below shows the jitter phase count which this function"]
    #[doc = " would return for each of the quality presets."]
    #[doc = ""]
    #[doc = " Quality preset                                        | Scale factor  | Phase count"]
    #[doc = " ----------------------------------------------------- | ------------- | ---------------"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_QUALITY</i></c>           | 1.5x          | 18"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_BALANCED</i></c>          | 1.7x          | 23"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x          | 32"]
    #[doc = " <c><i>FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> | 3.0x          | 72"]
    #[doc = " Custom                                                | [1..n]x       | ceil(8*n^2)"]
    #[doc = ""]
    #[doc = " @param [in] renderWidth             The render resolution width."]
    #[doc = " @param [in] displayWidth            The display resolution width."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " The jitter phase count for the scaling factor between <c><i>renderWidth</i></c> and <c><i>displayWidth</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2GetJitterPhaseCount(renderWidth: i32, displayWidth: i32) -> i32;

    #[doc = " A helper function to calculate the subpixel jitter offset."]
    #[doc = ""]
    #[doc = " FSR2 relies on the application to apply sub-pixel jittering while rendering."]
    #[doc = " This is typically included in the projection matrix of the camera. To make"]
    #[doc = " the application of camera jitter simple, the FSR2 API provides a small set"]
    #[doc = " of utility function which computes the sub-pixel jitter offset for a"]
    #[doc = " particular frame within a sequence of separate jitter offsets. To begin, the"]
    #[doc = " index within the jitter phase must be computed. To calculate the"]
    #[doc = " sequence's length, you can call the <c><i>ffxFsr2GetJitterPhaseCount</i></c>"]
    #[doc = " function. The index should be a value which is incremented each frame modulo"]
    #[doc = " the length of the sequence computed by <c><i>ffxFsr2GetJitterPhaseCount</i></c>."]
    #[doc = " The index within the jitter phase  is passed to"]
    #[doc = " <c><i>ffxFsr2GetJitterOffset</i></c> via the <c><i>index</i></c> parameter."]
    #[doc = ""]
    #[doc = " This function uses a Halton(2,3) sequence to compute the jitter offset."]
    #[doc = " The ultimate index used for the sequence is <c><i>index</i></c> %"]
    #[doc = " <c><i>phaseCount</i></c>."]
    #[doc = ""]
    #[doc = " It is important to understand that the values returned from the"]
    #[doc = " <c><i>ffxFsr2GetJitterOffset</i></c> function are in unit pixel space, and"]
    #[doc = " in order to composite this correctly into a projection matrix we must"]
    #[doc = " convert them into projection offsets. This is done as per the pseudo code"]
    #[doc = " listing which is shown below."]
    #[doc = ""]
    #[doc = "     const int32_t jitterPhaseCount = ffxFsr2GetJitterPhaseCount(renderWidth, displayWidth);"]
    #[doc = ""]
    #[doc = "     float jitterX = 0;"]
    #[doc = "     float jitterY = 0;"]
    #[doc = "     ffxFsr2GetJitterOffset(&jitterX, &jitterY, index, jitterPhaseCount);"]
    #[doc = ""]
    #[doc = "     const float jitterX = 2.0f * jitterX / (float)renderWidth;"]
    #[doc = "     const float jitterY = -2.0f * jitterY / (float)renderHeight;"]
    #[doc = "     const Matrix4 jitterTranslationMatrix = translateMatrix(Matrix3::identity, Vector3(jitterX, jitterY, 0));"]
    #[doc = "     const Matrix4 jitteredProjectionMatrix = jitterTranslationMatrix * projectionMatrix;"]
    #[doc = ""]
    #[doc = " Jitter should be applied to all rendering. This includes opaque, alpha"]
    #[doc = " transparent, and raytraced objects. For rasterized objects, the sub-pixel"]
    #[doc = " jittering values calculated by the <c><i>iffxFsr2GetJitterOffset</i></c>"]
    #[doc = " function can be applied to the camera projection matrix which is ultimately"]
    #[doc = " used to perform transformations during vertex shading. For raytraced"]
    #[doc = " rendering, the sub-pixel jitter should be applied to the ray's origin,"]
    #[doc = " often the camera's position."]
    #[doc = ""]
    #[doc = " Whether you elect to use the <c><i>ffxFsr2GetJitterOffset</i></c> function"]
    #[doc = " or your own sequence generator, you must program the"]
    #[doc = " <c><i>jitterOffset</i></c> field of the"]
    #[doc = " <c><i>FfxFsr2DispatchParameters</i></c> structure in order to inform FSR2"]
    #[doc = " of the jitter offset that has been applied in order to render each frame."]
    #[doc = ""]
    #[doc = " If not using the recommended <c><i>ffxFsr2GetJitterOffset</i></c> function,"]
    #[doc = " care should be taken that your jitter sequence never generates a null vector;"]
    #[doc = " that is value of 0 in both the X and Y dimensions."]
    #[doc = ""]
    #[doc = " @param [out] outX                   A pointer to a <c>float</c> which will contain the subpixel jitter offset for the x dimension."]
    #[doc = " @param [out] outY                   A pointer to a <c>float</c> which will contain the subpixel jitter offset for the y dimension."]
    #[doc = " @param [in] index                   The index within the jitter sequence."]
    #[doc = " @param [in] phaseCount              The length of jitter phase. See <c><i>ffxFsr2GetJitterPhaseCount</i></c>."]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                              The operation completed successfully."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_INVALID_POINTER           Either <c><i>outX</i></c> or <c><i>outY</i></c> was <c>NULL</c>."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_INVALID_ARGUMENT          Argument <c><i>phaseCount</i></c> must be greater than 0."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2GetJitterOffset(
        outX: *mut f32,
        outY: *mut f32,
        index: i32,
        phaseCount: i32,
    ) -> FfxErrorCode;

    #[doc = " A helper function to check if a resource is"]
    #[doc = " <c><i>FFX_FSR2_RESOURCE_IDENTIFIER_NULL</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] resource                A <c><i>FfxResource</i></c>."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " true                                The <c><i>resource</i></c> was not <c><i>FFX_FSR2_RESOURCE_IDENTIFIER_NULL</i></c>."]
    #[doc = " @returns"]
    #[doc = " false                               The <c><i>resource</i></c> was <c><i>FFX_FSR2_RESOURCE_IDENTIFIER_NULL</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2"]
    pub fn ffxFsr2ResourceIsNull(resource: FfxResource) -> bool;

    #[doc = " Query how much memory is required for the Vulkan backend's scratch buffer."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " The size (in bytes) of the required scratch memory buffer for the VK backend."]
    pub fn ffxFsr2GetScratchMemorySizeVK(
        physicalDevice: vk::PhysicalDevice,
        vkEnumerateDeviceExtensionProperties: PFN_vkEnumerateDeviceExtensionProperties,
    ) -> std::os::raw::c_ulonglong;

    #[doc = " Populate an interface with pointers for the VK backend."]
    #[doc = ""]
    #[doc = " @param [out] fsr2Interface              A pointer to a <c><i>FfxFsr2Interface</i></c> structure to populate with pointers."]
    #[doc = " @param [in] device                      A Vulkan device."]
    #[doc = " @param [in] scratchBuffer               A pointer to a buffer of memory which can be used by the DirectX(R)12 backend."]
    #[doc = " @param [in] scratchBufferSize           The size (in bytes) of the buffer pointed to by <c><i>scratchBuffer</i></c>."]
    #[doc = " @param [in] physicalDevice              The Vulkan physical device that FSR 2.0 will be executed on."]
    #[doc = " @param [in] getDeviceProcAddr           A function pointer to vkGetDeviceProcAddr which is used to obtain all the other Vulkan functions."]
    #[doc = ""]
    #[doc = " @retval"]
    #[doc = " FFX_OK                                  The operation completed successfully."]
    #[doc = " @retval"]
    #[doc = " FFX_ERROR_CODE_INVALID_POINTER          The <c><i>interface</i></c> pointer was <c><i>NULL</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxFsr2GetInterfaceVK(
        outInterface: *mut FfxFsr2Interface,
        scratchBuffer: *mut ::std::os::raw::c_void,
        scratchBufferSize: std::os::raw::c_ulonglong,
        instance: vk::Instance,
        physicalDevice: vk::PhysicalDevice,
        getInstanceProcAddr: PFN_vkGetInstanceProcAddr,
    ) -> FfxErrorCode;

    #[doc = " Create a <c><i>FfxFsr2Device</i></c> from a <c><i>VkDevice</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] device                      A pointer to the Vulkan logical device."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " An abstract FidelityFX device."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetDeviceVK(device: vk::Device) -> FfxDevice;

    #[doc = " Create a <c><i>FfxCommandList</i></c> from a <c><i>VkCommandBuffer</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] cmdBuf                      A pointer to the Vulkan command buffer."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " An abstract FidelityFX command list."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetCommandListVK(cmdBuf: vk::CommandBuffer) -> FfxCommandList;

    #[doc = " Create a <c><i>FfxResource</i></c> from a <c><i>VkImage</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] context                     A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] imgVk                       A Vulkan image resource."]
    #[doc = " @param [in] imageView                   An image view of the given image resource."]
    #[doc = " @param [in] width                       The width of the image resource."]
    #[doc = " @param [in] height                      The height of the image resource."]
    #[doc = " @param [in] imgFormat                   The format of the image resource."]
    #[doc = " @param [in] name                        (optional) A name string to identify the resource in debug mode."]
    #[doc = " @param [in] state                       The state the resource is currently in."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " An abstract FidelityFX resources."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetTextureResourceVK(
        context: *mut FfxFsr2Context,
        imgVk: vk::Image,
        imageView: vk::ImageView,
        width: u32,
        height: u32,
        imgFormat: vk::Format,
        name: *mut std::os::raw::c_ushort,
        state: FfxResourceStates,
    ) -> FfxResource;

    #[doc = " Create a <c><i>FfxResource</i></c> from a <c><i>VkBuffer</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] context                     A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] bufVk                       A Vulkan buffer resource."]
    #[doc = " @param [in] size                        The size of the buffer resource."]
    #[doc = " @param [in] name                        (optional) A name string to identify the resource in debug mode."]
    #[doc = " @param [in] state                       The state the resource is currently in."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " An abstract FidelityFX resources."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetBufferResourceVK(
        context: *mut FfxFsr2Context,
        bufVk: vk::Buffer,
        size: u32,
        name: *mut std::os::raw::c_ushort,
        state: FfxResourceStates,
    ) -> FfxResource;

    #[doc = " Convert a <c><i>FfxResource</i></c> value to a <c><i>VkImage</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] context                     A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] resId                       A resourceID."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " A <c><i>VkImage</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetVkImage(context: *mut FfxFsr2Context, resId: u32) -> vk::Image;

    #[doc = " Convert a <c><i>FfxResource</i></c> value to a <c><i>VkImageView</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] context                     A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] resId                       A resourceID."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " A <c><i>VkImage</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetVkImageView(context: *mut FfxFsr2Context, resId: u32) -> vk::ImageView;

    #[doc = " Convert a <c><i>FfxResource</i></c> value to a <c><i>VkImageLayout</i></c>."]
    #[doc = ""]
    #[doc = " @param [in] context                     A pointer to a <c><i>FfxFsr2Context</i></c> structure."]
    #[doc = " @param [in] resId                       A resourceID."]
    #[doc = ""]
    #[doc = " @returns"]
    #[doc = " A <c><i>VkImage</i></c>."]
    #[doc = ""]
    #[doc = " @ingroup FSR2 VK"]
    pub fn ffxGetVkImageLayout(context: *mut FfxFsr2Context, resId: u32) -> vk::ImageLayout;
}