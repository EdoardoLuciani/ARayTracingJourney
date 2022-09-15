pub mod amd_fsr2;
pub mod vk_combine;
pub mod vk_rt_lightning_shadows;
pub mod vk_tonemap;
pub mod vk_xe_gtao;

use ash::vk;
pub struct VkImagePrevState {
    pub src_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub src_layout: vk::ImageLayout,
}
