-- MetaX GPU backend, compiled with mxcc from the MACA toolkit.
-- MetaX sources use the ".mc" extension so that xmake does not route them
-- through the built-in CUDA rule, which requires an NVIDIA CUDA SDK.
--
-- The ".mc" objects are added directly to the llaisys-device and llaisys-ops
-- targets in xmake.lua: objects of a target using only a custom build rule
-- are not known at config time, so a standalone static target would not be
-- propagated to the final shared library link.
local MACA_PATH = os.getenv("MACA_PATH") or "/opt/maca"
local MXCC = path.join(MACA_PATH, "mxgpu_llvm/bin/mxcc")

rule("metax.macac")
    set_extensions(".mc")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        local objectfile = target:objectfile(sourcefile)
        local dependfile = target:dependfile(objectfile)
        os.mkdir(path.directory(objectfile))
        local argv = {
            "-x", "maca",
            "-offload-arch", "native",
            "--maca-path=" .. MACA_PATH,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-I" .. path.join(MACA_PATH, "include"),
            "-I" .. path.join(os.projectdir(), "include"),
            "-c", sourcefile,
            "-o", objectfile,
        }
        depend.on_changed(function ()
            vprint("compiling.$(mode) %s", sourcefile)
            os.vrunv(MXCC, argv)
        end, {dependfile = dependfile, files = {sourcefile}, changed = target:is_rebuilt()})
        table.insert(target:objectfiles(), objectfile)
    end)
