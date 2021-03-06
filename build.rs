use shaderc::{IncludeCallbackResult, IncludeType, ResolvedInclude};
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

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
    let mut compiler = shaderc::Compiler::new().unwrap();

    let err = compile_recursively(shaders_source_dir, out_dir.as_path(), &mut compiler);
    if err {
        panic!("Some shaders did not compile!")
    }
}

fn compile_recursively<T: AsRef<Path>>(
    source_dir: T,
    out_dir: T,
    compiler: &mut shaderc::Compiler,
) -> bool {
    let mut is_there_an_error = false;
    for entry in std::fs::read_dir(source_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let shader_kind = match path.extension().and_then(|v| v.to_str()) {
                Some("vert") => shaderc::ShaderKind::Vertex,
                Some("frag") => shaderc::ShaderKind::Fragment,
                Some("comp") => shaderc::ShaderKind::Compute,
                Some("rgen") => shaderc::ShaderKind::RayGeneration,
                Some("rchit") => shaderc::ShaderKind::ClosestHit,
                Some("rmiss") => shaderc::ShaderKind::Miss,
                _ => {
                    continue;
                }
            };
            let mut shader_file = File::open(&path).unwrap();
            let mut shader_contents = String::new();
            shader_file
                .read_to_string(&mut shader_contents)
                .expect("Could not read {path} contents");

            let mut compiler_options = shaderc::CompileOptions::new();
            compiler_options
                .as_mut()
                .unwrap()
                .set_target_spirv(shaderc::SpirvVersion::V1_5);
            compiler_options.as_mut().unwrap().set_include_callback(
                |requested_source, include_type, requestee_source, include_depth| {
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

                    let mut file =
                        File::open(&requested_source_path).map_err(|err| err.to_string())?;
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
            let compilation_result = compiler.compile_into_spirv(
                &shader_contents,
                shader_kind,
                path.to_str().unwrap(),
                "main",
                compiler_options.as_ref(),
            );
            match compilation_result {
                Ok(v) => {
                    println!("Shader {} compiled successfully", path.to_str().unwrap());
                    let mut new_shader_name = OsString::from(path.file_name().unwrap());
                    new_shader_name.push(".spirv");
                    let new_shader_path = PathBuf::from(out_dir.as_ref()).join(new_shader_name);
                    File::create(new_shader_path)
                        .expect("Cannot create shader file")
                        .write_all(v.as_binary_u8())
                        .expect("Cannot write binary contents to {new_shader_path}");
                }
                Err(v) => {
                    eprintln!("Shader {}", v);
                    is_there_an_error = true;
                }
            }
        } else if path.is_dir() {
            is_there_an_error |= compile_recursively(path.as_path(), out_dir.as_ref(), compiler);
        }
    }
    is_there_an_error
}
