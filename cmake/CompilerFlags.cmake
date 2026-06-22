function(kengine_apply_compiler_flags target)
    target_compile_features(${target} PUBLIC cxx_std_20)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Debug>:-g -O0 -fno-omit-frame-pointer>
            $<$<CONFIG:Release>:-O3 -DNDEBUG>
        )
    endif()
endfunction()

function(kengine_apply_c23_flags target)
    target_compile_features(${target} PUBLIC c_std_23)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Debug>:-g -O0>
            $<$<CONFIG:Release>:-O3 -DNDEBUG -ffast-math>
        )
    endif()
endfunction()