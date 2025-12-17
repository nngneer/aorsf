#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "aorsf::aorsf_core" for configuration ""
set_property(TARGET aorsf::aorsf_core APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(aorsf::aorsf_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libaorsf_core.a"
  )

list(APPEND _cmake_import_check_targets aorsf::aorsf_core )
list(APPEND _cmake_import_check_files_for_aorsf::aorsf_core "${_IMPORT_PREFIX}/lib/libaorsf_core.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
