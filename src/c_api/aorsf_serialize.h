#ifndef AORSF_SERIALIZE_H
#define AORSF_SERIALIZE_H

/**
 * @file aorsf_serialize.h
 * @brief Serialization format definitions for aorsf
 *
 * This header defines the binary file format structures.
 * The API functions are declared in aorsf_c.h.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* File format version */
#define AORSF_FORMAT_VERSION_MAJOR 1
#define AORSF_FORMAT_VERSION_MINOR 0

/* Magic bytes */
#define AORSF_MAGIC "ORSF"

/* File header structure (32 bytes) */
typedef struct {
    char     magic[4];
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t tree_type;
    uint32_t n_trees;
    uint32_t n_features;
    uint32_t n_class;
    uint32_t flags;
    uint32_t reserved;
} aorsf_file_header_t;

#ifdef __cplusplus
}
#endif

#endif /* AORSF_SERIALIZE_H */
