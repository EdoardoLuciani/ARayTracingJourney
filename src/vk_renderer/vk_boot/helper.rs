use ash::vk;
use std::ffi::CStr;
use std::fmt::Write;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub unsafe extern "system" fn vk_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let mut message = String::new();
    write!(message, "{:?}: ", message_severity).unwrap();
    write!(
        message,
        "[{:?}][{:?}] : {:?}",
        (*p_callback_data).message_id_number,
        CStr::from_ptr((*p_callback_data).p_message_id_name),
        CStr::from_ptr((*p_callback_data).p_message)
    )
    .unwrap();
    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        eprintln!("{}", message);
    } else {
        println!("{}", message);
    }
    vk::FALSE
}

pub fn get_binary_shader_data<T: AsRef<Path>>(
    path: T,
) -> (Vec<u8>, vk::ShaderStageFlags, vk::ShaderModuleCreateInfo) {
    let shader_type_extension = path
        .as_ref()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .rsplit_once('.')
        .expect("No shader type extension found")
        .1;
    let shader_type = match shader_type_extension {
        "vert" => vk::ShaderStageFlags::VERTEX,
        "frag" => vk::ShaderStageFlags::FRAGMENT,
        "comp" => vk::ShaderStageFlags::COMPUTE,
        "rgen" => vk::ShaderStageFlags::RAYGEN_KHR,
        "rint" => vk::ShaderStageFlags::INTERSECTION_KHR,
        "rahit" => vk::ShaderStageFlags::ANY_HIT_KHR,
        "rchit" => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        "rmiss" => vk::ShaderStageFlags::MISS_KHR,
        "rcall" => vk::ShaderStageFlags::CALLABLE_KHR,
        _ => panic!("Shader type could not be deducted"),
    };
    let mut file = File::open(path).expect("Could not open shader");
    let mut data = Vec::<u8>::new();
    file.read_to_end(&mut data).expect("Could not read shader");

    let module_create_info = vk::ShaderModuleCreateInfo {
        code_size: data.len(),
        p_code: data.as_ptr() as *const u32,
        ..vk::ShaderModuleCreateInfo::default()
    };
    (data, shader_type, module_create_info)
}
