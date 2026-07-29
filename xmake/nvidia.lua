target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("sm_89", "compute_89")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end
    add_links("cudart", {public = true})

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    add_files("../src/ops/nvidia/*.cu")
    add_cugencodes("sm_89", "compute_89")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end
    add_links("cudart", "cublas", {public = true})
    add_values("cuda.build.devlink", true)

    on_install(function (target) end)
target_end()
