use hassle_rs::HassleError;
use shaderc::{IncludeType, ResolvedInclude};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

struct ShaderCompiler {
    glsl_compiler: shaderc::Compiler,
}

impl ShaderCompiler {
    pub fn new() -> Self {
        ShaderCompiler {
            glsl_compiler: shaderc::Compiler::new().unwrap(),
        }
    }

    pub fn compile_shaders_recursively(
        &mut self,
        shader_source_dir: &Path,
        shader_spirv_destination: &Path,
    ) -> bool {
        let mut compilation_error: bool = false;
        for entry in std::fs::read_dir(shader_source_dir).unwrap() {
            let entry_path = entry.unwrap().path();
            if entry_path.is_file() {
                compilation_error |= match entry_path.extension().and_then(OsStr::to_str) {
                    Some("glsl") => self.compile_glsl(&entry_path, shader_spirv_destination),
                    Some("hlsl") => Self::compile_hlsl(&entry_path, shader_spirv_destination),
                    _ => false,
                };
            } else if entry_path.is_dir() {
                compilation_error |=
                    self.compile_shaders_recursively(entry_path.as_path(), shader_spirv_destination)
            }
        }
        compilation_error
    }

    fn compile_glsl(&mut self, shader_source: &Path, shader_spirv_destination: &Path) -> bool {
        let shader_kind = match shader_source
            .with_extension("")
            .extension()
            .and_then(|v| v.to_str())
        {
            Some("vert") => shaderc::ShaderKind::Vertex,
            Some("frag") => shaderc::ShaderKind::Fragment,
            Some("comp") => shaderc::ShaderKind::Compute,
            Some("rgen") => shaderc::ShaderKind::RayGeneration,
            Some("rchit") => shaderc::ShaderKind::ClosestHit,
            Some("rmiss") => shaderc::ShaderKind::Miss,
            _ => {
                return false;
            }
        };
        let shader_contents = Self::read_file(shader_source);

        let mut compiler_options = shaderc::CompileOptions::new();
        compiler_options
            .as_mut()
            .unwrap()
            .set_target_spirv(shaderc::SpirvVersion::V1_5);
        compiler_options.as_mut().unwrap().set_include_callback(
            |requested_source, include_type, requestee_source, _| {
                let requested_source_path = match include_type {
                    IncludeType::Relative => {
                        let mut requested_source_path = PathBuf::from(requestee_source);
                        requested_source_path.pop();
                        requested_source_path.push(Path::new(requested_source));
                        requested_source_path
                            .canonicalize()
                            .map_err(|err| err.to_string())?
                    }
                    IncludeType::Standard => PathBuf::from(requested_source),
                };

                let mut file = File::open(&requested_source_path).map_err(|err| err.to_string())?;
                let mut file_contents = String::new();
                file.read_to_string(&mut file_contents)
                    .map_err(|err| err.to_string())?;

                Ok(ResolvedInclude {
                    resolved_name: requested_source_path.to_str().unwrap().to_string(),
                    content: file_contents,
                })
            },
        );
        compiler_options.as_mut().unwrap().set_generate_debug_info();
        let compilation_result = self.glsl_compiler.compile_into_spirv(
            &shader_contents,
            shader_kind,
            shader_source.to_str().unwrap(),
            "main",
            compiler_options.as_ref(),
        );
        return match compilation_result {
            Ok(v) => {
                println!(
                    "Shader {} compiled successfully",
                    shader_source.to_str().unwrap()
                );
                let mut new_shader_name = shader_source
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                new_shader_name = new_shader_name.rsplit_once('.').unwrap().0.to_string();
                new_shader_name.push_str(".spirv");
                Self::save_file(shader_spirv_destination, &new_shader_name, v.as_binary_u8());
                false
            }
            Err(v) => {
                eprintln!("Shader {}", v);
                true
            }
        };
    }

    fn compile_hlsl(shader_source: &Path, shader_spirv_destination: &Path) -> bool {
        let target_profile = match shader_source
            .with_extension("")
            .extension()
            .and_then(|v| v.to_str())
        {
            Some("vert") => "vs_6_7",
            Some("frag") => "ps_6_7",
            Some("comp") => "cs_6_7",
            Some("rgen") => panic!(),
            Some("rchit") => panic!(),
            Some("rmiss") => panic!(),
            _ => {
                return false;
            }
        };
        let shader_contents = Self::read_file(shader_source);

        let compilation_result = hassle_rs::compile_hlsl(
            shader_source.to_str().unwrap(),
            &shader_contents,
            "main",
            target_profile,
            &["-spirv"],
            &[],
        );

        return match compilation_result {
            Ok(v) => {
                println!(
                    "Shader {} compiled successfully",
                    shader_source.to_str().unwrap()
                );
                let mut new_shader_name = shader_source
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                new_shader_name = new_shader_name.rsplit_once('.').unwrap().0.to_string();
                new_shader_name.push_str(".spirv");
                Self::save_file(shader_spirv_destination, &new_shader_name, &v);
                false
            }
            Err(v) => {
                let error_string = match v {
                    HassleError::CompileError(s) => s,
                    HassleError::ValidationError(s) => s,
                    HassleError::WindowsOnly(s) => s,
                    HassleError::LoadLibraryError { filename, inner } => {
                        panic!("Load library error | {} | {}", inner, filename.display());
                    }
                    _ => {
                        panic!("Other error occurred");
                    }
                };
                eprintln!("Shader {}", error_string);
                true
            }
        };
    }

    fn read_file(shader_source: &Path) -> String {
        let mut shader_file = File::open(&shader_source).unwrap();
        let mut shader_contents = String::new();
        shader_file
            .read_to_string(&mut shader_contents)
            .expect("Could not read contents");
        shader_contents
    }

    fn save_file(dir: &Path, file_name: &str, contents: &[u8]) {
        let new_file_path = PathBuf::from(dir).join(file_name);
        File::create(new_file_path)
            .expect("Cannot create shader file")
            .write_all(contents)
            .expect("Cannot write binary contents");
    }
}

fn main() {
    // Tell the build script to only run again if we change our source shaders
    let shaders_source_dir = Path::new("src/vk_renderer/shaders");
    println!(
        "cargo:rerun-if-changed={}",
        shaders_source_dir.to_str().unwrap()
    );

    // Create destination path
    let mut out_dir = PathBuf::new();
    out_dir.push(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .unwrap()
            .to_str()
            .unwrap(),
    );
    out_dir.push("assets");
    out_dir.push("shaders-spirv");
    std::fs::create_dir_all(out_dir.as_path())
        .expect("Could not create assets//shaders-spirv directory in CARGO_MANIFEST_DIR");

    // Create the compiler
    let mut shader_compiler = ShaderCompiler::new();

    let err = shader_compiler.compile_shaders_recursively(shaders_source_dir, out_dir.as_path());
    if err {
        panic!("Some shaders did not compile!")
    }
}
