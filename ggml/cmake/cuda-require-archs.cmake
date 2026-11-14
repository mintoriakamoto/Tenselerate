# Fail configuration, loudly, when a CUDA architecture this build must serve is not compiled.
#
# Without this a card is lost silently, two ways:
#   - CUDA toolkit >= 13 cannot target sm_70 at all, and the default architecture list simply
#     omits it. The build succeeds and the binary cannot serve the card.
#   - An explicit list that skips an architecture (e.g. 61;80;86 with a Volta card present)
#     makes ggml_cuda_highest_compiled_arch() resolve that card to a lower architecture.
#     volta_mma_available() tests for exactly 700, so the card drops off its tensor-core
#     kernels with no error and no log line.
#
# GGML_CUDA_REQUIRE_ARCHS is a list of bare architecture numbers, e.g. "70;80;86".
# An entry matches the same number with an optional -real / -virtual suffix.
function(ggml_cuda_check_required_archs required archs toolkit_version)
    if (NOT required)
        return()
    endif()

    if ("${archs}" STREQUAL "native")
        message(FATAL_ERROR
            "GGML_CUDA_REQUIRE_ARCHS=${required} cannot be verified against CMAKE_CUDA_ARCHITECTURES=native: "
            "'native' compiles for whatever GPUs happen to be visible at build time. "
            "List the architectures explicitly, e.g. -DCMAKE_CUDA_ARCHITECTURES=\"70-real;80-real;86-real\".")
    endif()

    foreach (req IN LISTS required)
        if (NOT req MATCHES "^[0-9]+[af]?$")
            message(FATAL_ERROR "GGML_CUDA_REQUIRE_ARCHS: '${req}' is not a bare architecture number such as 70")
        endif()

        if (req LESS 75 AND toolkit_version VERSION_GREATER_EQUAL "13")
            message(FATAL_ERROR
                "GGML_CUDA_REQUIRE_ARCHS needs sm_${req}, but CUDA toolkit ${toolkit_version} cannot target "
                "architectures below sm_75. Build with a CUDA 12.x toolkit.")
        endif()

        set(found OFF)
        foreach (arch IN LISTS archs)
            if (arch MATCHES "^${req}(-real|-virtual)?$")
                set(found ON)
            endif()
        endforeach()

        if (NOT found)
            message(FATAL_ERROR
                "GGML_CUDA_REQUIRE_ARCHS needs sm_${req}, but CMAKE_CUDA_ARCHITECTURES is '${archs}'. "
                "A sm_${req} card would run on kernels built for a lower architecture, or not at all.")
        endif()
    endforeach()

    message(STATUS "CUDA required architectures present: ${required}")
endfunction()
