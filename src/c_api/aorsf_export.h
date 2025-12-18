#ifndef AORSF_EXPORT_H
#define AORSF_EXPORT_H

#ifdef _WIN32
    #ifdef AORSF_BUILD_SHARED
        #define AORSF_API __declspec(dllexport)
    #else
        #define AORSF_API __declspec(dllimport)
    #endif
#else
    #ifdef AORSF_BUILD_SHARED
        #define AORSF_API __attribute__((visibility("default")))
    #else
        #define AORSF_API
    #endif
#endif

#ifdef __cplusplus
    #define AORSF_EXTERN_C extern "C"
#else
    #define AORSF_EXTERN_C
#endif

#define AORSF_C_API AORSF_EXTERN_C AORSF_API

#endif /* AORSF_EXPORT_H */
