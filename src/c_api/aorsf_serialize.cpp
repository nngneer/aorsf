/**
 * @file aorsf_serialize.cpp
 * @brief Serialization implementation for aorsf models
 *
 * This file provides binary and JSON serialization for aorsf forests,
 * enabling cross-language model persistence.
 */

#include "aorsf_c.h"
#include "aorsf_internal.h"
#include "aorsf_serialize.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <vector>
#include <new>

using json = nlohmann::json;

namespace {

/* ============== Binary Serialization Helpers ============== */

template<typename T>
bool write_value(unsigned char*& ptr, const unsigned char* end, const T& value) {
    if (ptr + sizeof(T) > end) return false;
    std::memcpy(ptr, &value, sizeof(T));
    ptr += sizeof(T);
    return true;
}

template<typename T>
bool write_array(unsigned char*& ptr, const unsigned char* end,
                 const T* data, size_t count) {
    size_t bytes = count * sizeof(T);
    if (ptr + bytes > end) return false;
    if (count > 0) {
        std::memcpy(ptr, data, bytes);
    }
    ptr += bytes;
    return true;
}

template<typename T>
bool read_value(const unsigned char*& ptr, const unsigned char* end, T& value) {
    if (ptr + sizeof(T) > end) return false;
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return true;
}

template<typename T>
bool read_array(const unsigned char*& ptr, const unsigned char* end,
                T* data, size_t count) {
    size_t bytes = count * sizeof(T);
    if (ptr + bytes > end) return false;
    if (count > 0) {
        std::memcpy(data, ptr, bytes);
    }
    ptr += bytes;
    return true;
}

/* Calculate size needed for a single tree */
size_t calculate_tree_size(
    const std::vector<double>& cutpoints,
    const std::vector<arma::uword>& child_left,
    const std::vector<double>& leaf_summary,
    const std::vector<arma::vec>& coef_values,
    const std::vector<arma::uvec>& coef_indices
) {
    size_t size = 0;
    size_t n_nodes = cutpoints.size();

    size += sizeof(uint32_t);  // n_nodes
    size += n_nodes * sizeof(double);   // cutpoints
    size += n_nodes * sizeof(uint64_t); // child_left
    size += n_nodes * sizeof(double);   // leaf_summary

    // Coefficient data per node
    for (size_t i = 0; i < n_nodes; i++) {
        size += sizeof(uint32_t);  // n_coef
        if (i < coef_values.size() && coef_values[i].n_elem > 0) {
            size += coef_values[i].n_elem * sizeof(uint64_t);  // indices
            size += coef_values[i].n_elem * sizeof(double);    // values
        }
    }

    return size;
}

/* Calculate size for classification leaf data */
size_t calculate_classification_leaf_size(
    const std::vector<arma::vec>& leaf_pred_prob
) {
    size_t size = 0;
    for (const auto& prob : leaf_pred_prob) {
        size += sizeof(uint32_t);  // n_probs
        size += prob.n_elem * sizeof(double);
    }
    return size;
}

/* Calculate size for survival leaf data */
size_t calculate_survival_leaf_size(
    const std::vector<arma::vec>& leaf_pred_indx,
    const std::vector<arma::vec>& leaf_pred_prob,
    const std::vector<arma::vec>& leaf_pred_chaz
) {
    size_t size = 0;
    for (size_t i = 0; i < leaf_pred_indx.size(); i++) {
        size += sizeof(uint32_t);  // n_times
        if (leaf_pred_indx[i].n_elem > 0) {
            size += leaf_pred_indx[i].n_elem * sizeof(double);  // indx
            size += leaf_pred_prob[i].n_elem * sizeof(double);  // prob
            size += leaf_pred_chaz[i].n_elem * sizeof(double);  // chaz
        }
    }
    return size;
}

/* ============== JSON Serialization Helpers ============== */

json arma_vec_to_json(const arma::vec& v) {
    json arr = json::array();
    for (arma::uword i = 0; i < v.n_elem; i++) {
        arr.push_back(v(i));
    }
    return arr;
}

json arma_uvec_to_json(const arma::uvec& v) {
    json arr = json::array();
    for (arma::uword i = 0; i < v.n_elem; i++) {
        arr.push_back(static_cast<uint64_t>(v(i)));
    }
    return arr;
}

arma::vec json_to_arma_vec(const json& arr) {
    arma::vec v(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
        v(i) = arr[i].get<double>();
    }
    return v;
}

arma::uvec json_to_arma_uvec(const json& arr) {
    arma::uvec v(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
        v(i) = static_cast<arma::uword>(arr[i].get<uint64_t>());
    }
    return v;
}

json forest_to_json(aorsf_forest_handle handle, uint32_t flags) {
    json j;

    // Header info
    j["format"] = "aorsf";
    j["version"] = {
        {"major", AORSF_FORMAT_VERSION_MAJOR},
        {"minor", AORSF_FORMAT_VERSION_MINOR}
    };

    // Tree type as string for readability
    const char* tree_type_str = "unknown";
    switch (handle->config.tree_type) {
        case AORSF_TREE_CLASSIFICATION: tree_type_str = "classification"; break;
        case AORSF_TREE_REGRESSION: tree_type_str = "regression"; break;
        case AORSF_TREE_SURVIVAL: tree_type_str = "survival"; break;
    }
    j["tree_type"] = tree_type_str;
    j["n_trees"] = handle->cutpoints.size();
    j["n_features"] = handle->n_features;
    j["n_class"] = handle->n_class;

    // Config
    j["config"] = {
        {"n_tree", handle->config.n_tree},
        {"mtry", handle->config.mtry},
        {"leaf_min_obs", handle->config.leaf_min_obs},
        {"split_min_obs", handle->config.split_min_obs},
        {"split_rule", handle->config.split_rule},
        {"lincomb_type", handle->config.lincomb_type},
        {"lincomb_eps", handle->config.lincomb_eps},
        {"lincomb_iter_max", handle->config.lincomb_iter_max},
        {"lincomb_alpha", handle->config.lincomb_alpha},
        {"vi_type", handle->config.vi_type},
        {"seed", handle->config.seed}
    };

    j["oob_error"] = handle->oob_error;

    // Trees
    json trees = json::array();
    for (size_t t = 0; t < handle->cutpoints.size(); t++) {
        json tree;
        size_t n_nodes = handle->cutpoints[t].size();
        tree["n_nodes"] = n_nodes;

        // Cutpoints
        tree["cutpoints"] = handle->cutpoints[t];

        // Child left (convert to uint64)
        json child_left_arr = json::array();
        for (size_t i = 0; i < n_nodes; i++) {
            child_left_arr.push_back(static_cast<uint64_t>(handle->child_left[t][i]));
        }
        tree["child_left"] = child_left_arr;

        // Leaf summary
        tree["leaf_summary"] = handle->leaf_summary[t];

        // Coefficients per node
        json nodes = json::array();
        for (size_t i = 0; i < n_nodes; i++) {
            json node;
            if (i < handle->coef_values[t].size() && handle->coef_values[t][i].n_elem > 0) {
                node["coef_indices"] = arma_uvec_to_json(handle->coef_indices[t][i]);
                node["coef_values"] = arma_vec_to_json(handle->coef_values[t][i]);
            } else {
                node["coef_indices"] = json::array();
                node["coef_values"] = json::array();
            }
            nodes.push_back(node);
        }
        tree["nodes"] = nodes;

        // Type-specific leaf data
        if (handle->config.tree_type == AORSF_TREE_CLASSIFICATION) {
            if (t < handle->leaf_pred_prob.size()) {
                json leaf_probs = json::array();
                for (size_t i = 0; i < handle->leaf_pred_prob[t].size(); i++) {
                    leaf_probs.push_back(arma_vec_to_json(handle->leaf_pred_prob[t][i]));
                }
                tree["leaf_pred_prob"] = leaf_probs;
            }
        } else if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
            if (t < handle->leaf_pred_indx.size()) {
                json leaf_indx = json::array();
                json leaf_prob = json::array();
                json leaf_chaz = json::array();
                for (size_t i = 0; i < handle->leaf_pred_indx[t].size(); i++) {
                    leaf_indx.push_back(arma_vec_to_json(handle->leaf_pred_indx[t][i]));
                    leaf_prob.push_back(arma_vec_to_json(handle->leaf_pred_prob[t][i]));
                    leaf_chaz.push_back(arma_vec_to_json(handle->leaf_pred_chaz[t][i]));
                }
                tree["leaf_pred_indx"] = leaf_indx;
                tree["leaf_pred_prob"] = leaf_prob;
                tree["leaf_pred_chaz"] = leaf_chaz;
            }
        }

        trees.push_back(tree);
    }
    j["trees"] = trees;

    // Importance
    if ((flags & AORSF_FLAG_HAS_IMPORTANCE) && !handle->importance.empty()) {
        j["importance"] = handle->importance;
    }

    // Unique times for survival
    if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
        j["unique_times"] = handle->unique_times;
    }

    // OOB data
    if ((flags & AORSF_FLAG_HAS_OOB) && !handle->rows_oobag.empty()) {
        json oob_data;
        json rows_oobag_arr = json::array();
        for (const auto& rows : handle->rows_oobag) {
            rows_oobag_arr.push_back(arma_uvec_to_json(rows));
        }
        oob_data["rows_oobag"] = rows_oobag_arr;
        oob_data["oobag_denom"] = handle->oobag_denom;
        j["oob_data"] = oob_data;
    }

    // Metadata
    if (flags & AORSF_FLAG_HAS_METADATA) {
        json metadata;
        if (!handle->feature_names.empty()) {
            metadata["feature_names"] = handle->feature_names;
        }
        if (!handle->feature_means.empty()) {
            metadata["feature_means"] = handle->feature_means;
        }
        if (!handle->feature_stds.empty()) {
            metadata["feature_stds"] = handle->feature_stds;
        }
        if (!metadata.empty()) {
            j["metadata"] = metadata;
        }
    }

    return j;
}

aorsf_error_t json_to_forest(const json& j, aorsf_forest_t** new_handle) {
    // Validate format
    if (!j.contains("format") || j["format"] != "aorsf") {
        set_error("Invalid JSON format (missing or wrong format field)");
        return AORSF_ERROR_FORMAT;
    }

    // Check version
    int major = j["version"]["major"].get<int>();
    if (major > AORSF_FORMAT_VERSION_MAJOR) {
        set_error("JSON format version too new");
        return AORSF_ERROR_FORMAT;
    }

    aorsf_forest_t* handle = new (std::nothrow) aorsf_forest_t();
    if (!handle) {
        set_error("Out of memory");
        return AORSF_ERROR_OUT_OF_MEMORY;
    }

    // Parse tree type
    std::string tree_type_str = j["tree_type"].get<std::string>();
    if (tree_type_str == "classification") {
        handle->config.tree_type = AORSF_TREE_CLASSIFICATION;
    } else if (tree_type_str == "regression") {
        handle->config.tree_type = AORSF_TREE_REGRESSION;
    } else if (tree_type_str == "survival") {
        handle->config.tree_type = AORSF_TREE_SURVIVAL;
    } else {
        delete handle;
        set_error("Unknown tree type");
        return AORSF_ERROR_FORMAT;
    }

    handle->n_features = j["n_features"].get<int32_t>();
    handle->n_class = j["n_class"].get<int32_t>();

    // Config
    const auto& cfg = j["config"];
    handle->config.n_tree = cfg["n_tree"].get<int32_t>();
    handle->config.mtry = cfg["mtry"].get<int32_t>();
    handle->config.leaf_min_obs = cfg["leaf_min_obs"].get<int32_t>();
    handle->config.split_min_obs = cfg["split_min_obs"].get<int32_t>();
    handle->config.split_rule = static_cast<aorsf_split_rule_t>(cfg["split_rule"].get<int>());
    handle->config.lincomb_type = static_cast<aorsf_lincomb_type_t>(cfg["lincomb_type"].get<int>());
    handle->config.lincomb_eps = cfg["lincomb_eps"].get<double>();
    handle->config.lincomb_iter_max = cfg["lincomb_iter_max"].get<int32_t>();
    handle->config.lincomb_alpha = cfg["lincomb_alpha"].get<double>();
    handle->config.vi_type = static_cast<aorsf_vi_type_t>(cfg["vi_type"].get<int>());
    handle->config.seed = cfg["seed"].get<uint32_t>();

    handle->oob_error = j["oob_error"].get<double>();

    // Trees
    const auto& trees = j["trees"];
    size_t n_trees = trees.size();
    handle->cutpoints.resize(n_trees);
    handle->child_left.resize(n_trees);
    handle->leaf_summary.resize(n_trees);
    handle->coef_values.resize(n_trees);
    handle->coef_indices.resize(n_trees);

    if (handle->config.tree_type == AORSF_TREE_CLASSIFICATION) {
        handle->leaf_pred_prob.resize(n_trees);
    } else if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
        handle->leaf_pred_indx.resize(n_trees);
        handle->leaf_pred_prob.resize(n_trees);
        handle->leaf_pred_chaz.resize(n_trees);
    }

    for (size_t t = 0; t < n_trees; t++) {
        const auto& tree = trees[t];
        size_t n_nodes = tree["n_nodes"].get<size_t>();

        // Cutpoints
        handle->cutpoints[t] = tree["cutpoints"].get<std::vector<double>>();

        // Child left
        handle->child_left[t].resize(n_nodes);
        const auto& child_left_arr = tree["child_left"];
        for (size_t i = 0; i < n_nodes; i++) {
            handle->child_left[t][i] = static_cast<arma::uword>(child_left_arr[i].get<uint64_t>());
        }

        // Leaf summary
        handle->leaf_summary[t] = tree["leaf_summary"].get<std::vector<double>>();

        // Nodes with coefficients
        const auto& nodes = tree["nodes"];
        handle->coef_values[t].resize(n_nodes);
        handle->coef_indices[t].resize(n_nodes);
        for (size_t i = 0; i < n_nodes; i++) {
            const auto& node = nodes[i];
            if (node["coef_indices"].size() > 0) {
                handle->coef_indices[t][i] = json_to_arma_uvec(node["coef_indices"]);
                handle->coef_values[t][i] = json_to_arma_vec(node["coef_values"]);
            }
        }

        // Type-specific leaf data
        if (handle->config.tree_type == AORSF_TREE_CLASSIFICATION) {
            if (tree.contains("leaf_pred_prob")) {
                const auto& leaf_probs = tree["leaf_pred_prob"];
                handle->leaf_pred_prob[t].resize(leaf_probs.size());
                for (size_t i = 0; i < leaf_probs.size(); i++) {
                    handle->leaf_pred_prob[t][i] = json_to_arma_vec(leaf_probs[i]);
                }
            }
        } else if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
            if (tree.contains("leaf_pred_indx")) {
                const auto& leaf_indx = tree["leaf_pred_indx"];
                const auto& leaf_prob = tree["leaf_pred_prob"];
                const auto& leaf_chaz = tree["leaf_pred_chaz"];
                size_t n_leaves = leaf_indx.size();
                handle->leaf_pred_indx[t].resize(n_leaves);
                handle->leaf_pred_prob[t].resize(n_leaves);
                handle->leaf_pred_chaz[t].resize(n_leaves);
                for (size_t i = 0; i < n_leaves; i++) {
                    handle->leaf_pred_indx[t][i] = json_to_arma_vec(leaf_indx[i]);
                    handle->leaf_pred_prob[t][i] = json_to_arma_vec(leaf_prob[i]);
                    handle->leaf_pred_chaz[t][i] = json_to_arma_vec(leaf_chaz[i]);
                }
            }
        }
    }

    // Importance
    if (j.contains("importance")) {
        handle->importance = j["importance"].get<std::vector<double>>();
    }

    // Unique times for survival
    if (j.contains("unique_times")) {
        handle->unique_times = j["unique_times"].get<std::vector<double>>();
    }

    // OOB data
    if (j.contains("oob_data")) {
        const auto& oob_data = j["oob_data"];
        const auto& rows_oobag_arr = oob_data["rows_oobag"];
        handle->rows_oobag.resize(rows_oobag_arr.size());
        for (size_t t = 0; t < rows_oobag_arr.size(); t++) {
            handle->rows_oobag[t] = json_to_arma_uvec(rows_oobag_arr[t]);
        }
        handle->oobag_denom = oob_data["oobag_denom"].get<std::vector<double>>();
    }

    // Metadata
    if (j.contains("metadata")) {
        const auto& metadata = j["metadata"];
        if (metadata.contains("feature_names")) {
            handle->feature_names = metadata["feature_names"].get<std::vector<std::string>>();
        }
        if (metadata.contains("feature_means")) {
            handle->feature_means = metadata["feature_means"].get<std::vector<double>>();
        }
        if (metadata.contains("feature_stds")) {
            handle->feature_stds = metadata["feature_stds"].get<std::vector<double>>();
        }
    }

    handle->is_fitted = true;
    *new_handle = handle;
    return AORSF_SUCCESS;
}

} // anonymous namespace

/* ============== Public API Implementation ============== */

AORSF_C_API aorsf_error_t aorsf_forest_get_save_size(
    aorsf_forest_handle handle,
    aorsf_format_t format,
    uint32_t flags,
    size_t* size
) {
    if (!handle || !size) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    if (format == AORSF_FORMAT_JSON) {
        // For JSON, we need to actually generate to know the size
        // Use indent=2 for pretty printing consistency with save
        try {
            json j = forest_to_json(handle, flags);
            std::string json_str = j.dump(2);
            *size = json_str.size();
            return AORSF_SUCCESS;
        } catch (const std::exception& e) {
            set_error(std::string("JSON serialization error: ") + e.what());
            return AORSF_ERROR_IO;
        }
    }

    size_t total = 0;

    // Header (32 bytes)
    total += sizeof(aorsf_file_header_t);

    // Config section (we'll use a fixed size structure)
    total += 96;

    // Per-tree data
    for (size_t t = 0; t < handle->cutpoints.size(); t++) {
        total += calculate_tree_size(
            handle->cutpoints[t],
            handle->child_left[t],
            handle->leaf_summary[t],
            handle->coef_values[t],
            handle->coef_indices[t]
        );

        // Type-specific leaf data
        if (handle->config.tree_type == AORSF_TREE_CLASSIFICATION) {
            if (t < handle->leaf_pred_prob.size()) {
                total += calculate_classification_leaf_size(handle->leaf_pred_prob[t]);
            }
        } else if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
            if (t < handle->leaf_pred_indx.size()) {
                total += calculate_survival_leaf_size(
                    handle->leaf_pred_indx[t],
                    handle->leaf_pred_prob[t],
                    handle->leaf_pred_chaz[t]
                );
            }
        }
    }

    // Importance
    if ((flags & AORSF_FLAG_HAS_IMPORTANCE) && !handle->importance.empty()) {
        total += handle->importance.size() * sizeof(double);
    }

    // Unique times (survival)
    if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
        total += sizeof(uint32_t);  // n_times
        total += handle->unique_times.size() * sizeof(double);
    }

    // OOB data
    if ((flags & AORSF_FLAG_HAS_OOB) && !handle->rows_oobag.empty()) {
        total += sizeof(uint32_t);  // n_trees
        for (const auto& rows : handle->rows_oobag) {
            total += sizeof(uint32_t);  // n_rows for this tree
            total += rows.n_elem * sizeof(uint64_t);  // row indices
        }
        total += sizeof(uint32_t);  // n_obs
        total += handle->oobag_denom.size() * sizeof(double);
    }

    // Metadata
    if (flags & AORSF_FLAG_HAS_METADATA) {
        total += sizeof(uint32_t);  // has_names flag
        if (!handle->feature_names.empty()) {
            total += sizeof(uint32_t);  // n_names
            for (const auto& name : handle->feature_names) {
                total += sizeof(uint32_t);  // string length
                total += name.size();       // string bytes
            }
        }

        total += sizeof(uint32_t);  // has_means flag
        if (!handle->feature_means.empty()) {
            total += sizeof(uint32_t);  // n_means
            total += handle->feature_means.size() * sizeof(double);
        }

        total += sizeof(uint32_t);  // has_stds flag
        if (!handle->feature_stds.empty()) {
            total += sizeof(uint32_t);  // n_stds
            total += handle->feature_stds.size() * sizeof(double);
        }
    }

    *size = total;
    return AORSF_SUCCESS;
}

AORSF_C_API aorsf_error_t aorsf_forest_save(
    aorsf_forest_handle handle,
    aorsf_format_t format,
    uint32_t flags,
    unsigned char* buffer,
    size_t buffer_size,
    size_t* written
) {
    if (!handle || !buffer || !written) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    if (format == AORSF_FORMAT_JSON) {
        try {
            json j = forest_to_json(handle, flags);
            std::string json_str = j.dump(2);  // Pretty print with indent of 2
            if (json_str.size() > buffer_size) {
                set_error("Buffer too small for JSON");
                return AORSF_ERROR_INVALID_ARGUMENT;
            }
            std::memcpy(buffer, json_str.data(), json_str.size());
            *written = json_str.size();
            return AORSF_SUCCESS;
        } catch (const std::exception& e) {
            set_error(std::string("JSON serialization error: ") + e.what());
            return AORSF_ERROR_IO;
        }
    }

    // Verify buffer size
    size_t needed;
    aorsf_error_t err = aorsf_forest_get_save_size(handle, format, flags, &needed);
    if (err != AORSF_SUCCESS) return err;

    if (buffer_size < needed) {
        set_error("Buffer too small");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    unsigned char* ptr = buffer;
    const unsigned char* end = buffer + buffer_size;

    // Write header
    aorsf_file_header_t header;
    std::memcpy(header.magic, AORSF_MAGIC, 4);
    header.version_major = AORSF_FORMAT_VERSION_MAJOR;
    header.version_minor = AORSF_FORMAT_VERSION_MINOR;
    header.tree_type = static_cast<uint32_t>(handle->config.tree_type);
    header.n_trees = static_cast<uint32_t>(handle->cutpoints.size());
    header.n_features = static_cast<uint32_t>(handle->n_features);
    header.n_class = static_cast<uint32_t>(handle->n_class);
    header.flags = flags;
    header.reserved = 0;

    if (!write_value(ptr, end, header)) {
        set_error("Failed to write header");
        return AORSF_ERROR_IO;
    }

    // Write config section (simplified - just key values)
    struct {
        int32_t tree_type;
        int32_t n_tree;
        int32_t mtry;
        int32_t leaf_min_obs;
        int32_t split_min_obs;
        int32_t split_rule;
        int32_t lincomb_type;
        double lincomb_eps;
        int32_t lincomb_iter_max;
        double lincomb_alpha;
        int32_t vi_type;
        uint32_t seed;
        double oob_error;
        int32_t padding[11];  // Pad to 96 bytes
    } config_data;
    std::memset(&config_data, 0, sizeof(config_data));

    config_data.tree_type = handle->config.tree_type;
    config_data.n_tree = handle->config.n_tree;
    config_data.mtry = handle->config.mtry;
    config_data.leaf_min_obs = handle->config.leaf_min_obs;
    config_data.split_min_obs = handle->config.split_min_obs;
    config_data.split_rule = handle->config.split_rule;
    config_data.lincomb_type = handle->config.lincomb_type;
    config_data.lincomb_eps = handle->config.lincomb_eps;
    config_data.lincomb_iter_max = handle->config.lincomb_iter_max;
    config_data.lincomb_alpha = handle->config.lincomb_alpha;
    config_data.vi_type = handle->config.vi_type;
    config_data.seed = handle->config.seed;
    config_data.oob_error = handle->oob_error;

    if (!write_array(ptr, end, reinterpret_cast<const unsigned char*>(&config_data), 96)) {
        set_error("Failed to write config");
        return AORSF_ERROR_IO;
    }

    // Write per-tree data
    for (size_t t = 0; t < handle->cutpoints.size(); t++) {
        uint32_t n_nodes = static_cast<uint32_t>(handle->cutpoints[t].size());
        if (!write_value(ptr, end, n_nodes)) {
            set_error("Failed to write n_nodes");
            return AORSF_ERROR_IO;
        }

        // Cutpoints
        if (!write_array(ptr, end, handle->cutpoints[t].data(), n_nodes)) {
            set_error("Failed to write cutpoints");
            return AORSF_ERROR_IO;
        }

        // Child left (convert arma::uword to uint64_t)
        for (uint32_t i = 0; i < n_nodes; i++) {
            uint64_t val = static_cast<uint64_t>(handle->child_left[t][i]);
            if (!write_value(ptr, end, val)) {
                set_error("Failed to write child_left");
                return AORSF_ERROR_IO;
            }
        }

        // Leaf summary
        if (!write_array(ptr, end, handle->leaf_summary[t].data(), n_nodes)) {
            set_error("Failed to write leaf_summary");
            return AORSF_ERROR_IO;
        }

        // Coefficients per node
        for (uint32_t i = 0; i < n_nodes; i++) {
            uint32_t n_coef = 0;
            if (i < handle->coef_values[t].size()) {
                n_coef = static_cast<uint32_t>(handle->coef_values[t][i].n_elem);
            }
            if (!write_value(ptr, end, n_coef)) {
                set_error("Failed to write n_coef");
                return AORSF_ERROR_IO;
            }

            if (n_coef > 0) {
                // Indices
                for (arma::uword j = 0; j < n_coef; j++) {
                    uint64_t idx = static_cast<uint64_t>(handle->coef_indices[t][i](j));
                    if (!write_value(ptr, end, idx)) {
                        set_error("Failed to write coef_indices");
                        return AORSF_ERROR_IO;
                    }
                }
                // Values
                if (!write_array(ptr, end, handle->coef_values[t][i].memptr(), n_coef)) {
                    set_error("Failed to write coef_values");
                    return AORSF_ERROR_IO;
                }
            }
        }

        // Type-specific leaf data
        if (handle->config.tree_type == AORSF_TREE_CLASSIFICATION) {
            if (t < handle->leaf_pred_prob.size()) {
                for (size_t i = 0; i < handle->leaf_pred_prob[t].size(); i++) {
                    uint32_t n_probs = static_cast<uint32_t>(handle->leaf_pred_prob[t][i].n_elem);
                    if (!write_value(ptr, end, n_probs)) {
                        return AORSF_ERROR_IO;
                    }
                    if (n_probs > 0) {
                        if (!write_array(ptr, end, handle->leaf_pred_prob[t][i].memptr(), n_probs)) {
                            return AORSF_ERROR_IO;
                        }
                    }
                }
            }
        } else if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
            if (t < handle->leaf_pred_indx.size()) {
                for (size_t i = 0; i < handle->leaf_pred_indx[t].size(); i++) {
                    uint32_t n_times = static_cast<uint32_t>(handle->leaf_pred_indx[t][i].n_elem);
                    if (!write_value(ptr, end, n_times)) {
                        return AORSF_ERROR_IO;
                    }
                    if (n_times > 0) {
                        if (!write_array(ptr, end, handle->leaf_pred_indx[t][i].memptr(), n_times)) {
                            return AORSF_ERROR_IO;
                        }
                        if (!write_array(ptr, end, handle->leaf_pred_prob[t][i].memptr(), n_times)) {
                            return AORSF_ERROR_IO;
                        }
                        if (!write_array(ptr, end, handle->leaf_pred_chaz[t][i].memptr(), n_times)) {
                            return AORSF_ERROR_IO;
                        }
                    }
                }
            }
        }
    }

    // Write importance if present
    if ((flags & AORSF_FLAG_HAS_IMPORTANCE) && !handle->importance.empty()) {
        if (!write_array(ptr, end, handle->importance.data(), handle->importance.size())) {
            set_error("Failed to write importance");
            return AORSF_ERROR_IO;
        }
    }

    // Write unique times for survival
    if (handle->config.tree_type == AORSF_TREE_SURVIVAL) {
        uint32_t n_times = static_cast<uint32_t>(handle->unique_times.size());
        if (!write_value(ptr, end, n_times)) {
            return AORSF_ERROR_IO;
        }
        if (!write_array(ptr, end, handle->unique_times.data(), n_times)) {
            return AORSF_ERROR_IO;
        }
    }

    // Write OOB data if present
    if ((flags & AORSF_FLAG_HAS_OOB) && !handle->rows_oobag.empty()) {
        uint32_t n_trees_oob = static_cast<uint32_t>(handle->rows_oobag.size());
        if (!write_value(ptr, end, n_trees_oob)) {
            return AORSF_ERROR_IO;
        }
        for (const auto& rows : handle->rows_oobag) {
            uint32_t n_rows = static_cast<uint32_t>(rows.n_elem);
            if (!write_value(ptr, end, n_rows)) {
                return AORSF_ERROR_IO;
            }
            for (arma::uword i = 0; i < rows.n_elem; i++) {
                uint64_t idx = static_cast<uint64_t>(rows(i));
                if (!write_value(ptr, end, idx)) {
                    return AORSF_ERROR_IO;
                }
            }
        }
        uint32_t n_obs = static_cast<uint32_t>(handle->oobag_denom.size());
        if (!write_value(ptr, end, n_obs)) {
            return AORSF_ERROR_IO;
        }
        if (!write_array(ptr, end, handle->oobag_denom.data(), n_obs)) {
            return AORSF_ERROR_IO;
        }
    }

    // Write metadata if present
    if (flags & AORSF_FLAG_HAS_METADATA) {
        // Feature names
        uint32_t has_names = handle->feature_names.empty() ? 0 : 1;
        if (!write_value(ptr, end, has_names)) {
            return AORSF_ERROR_IO;
        }
        if (has_names) {
            uint32_t n_names = static_cast<uint32_t>(handle->feature_names.size());
            if (!write_value(ptr, end, n_names)) {
                return AORSF_ERROR_IO;
            }
            for (const auto& name : handle->feature_names) {
                uint32_t len = static_cast<uint32_t>(name.size());
                if (!write_value(ptr, end, len)) {
                    return AORSF_ERROR_IO;
                }
                if (len > 0 && !write_array(ptr, end, reinterpret_cast<const unsigned char*>(name.data()), len)) {
                    return AORSF_ERROR_IO;
                }
            }
        }

        // Feature means
        uint32_t has_means = handle->feature_means.empty() ? 0 : 1;
        if (!write_value(ptr, end, has_means)) {
            return AORSF_ERROR_IO;
        }
        if (has_means) {
            uint32_t n_means = static_cast<uint32_t>(handle->feature_means.size());
            if (!write_value(ptr, end, n_means)) {
                return AORSF_ERROR_IO;
            }
            if (!write_array(ptr, end, handle->feature_means.data(), n_means)) {
                return AORSF_ERROR_IO;
            }
        }

        // Feature stds
        uint32_t has_stds = handle->feature_stds.empty() ? 0 : 1;
        if (!write_value(ptr, end, has_stds)) {
            return AORSF_ERROR_IO;
        }
        if (has_stds) {
            uint32_t n_stds = static_cast<uint32_t>(handle->feature_stds.size());
            if (!write_value(ptr, end, n_stds)) {
                return AORSF_ERROR_IO;
            }
            if (!write_array(ptr, end, handle->feature_stds.data(), n_stds)) {
                return AORSF_ERROR_IO;
            }
        }
    }

    *written = static_cast<size_t>(ptr - buffer);
    return AORSF_SUCCESS;
}

AORSF_C_API aorsf_error_t aorsf_forest_load(
    aorsf_forest_handle* handle,
    const unsigned char* buffer,
    size_t buffer_size
) {
    if (!handle || !buffer) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    // Auto-detect format: JSON starts with '{' or whitespace then '{'
    // Binary starts with "ORSF" magic bytes
    size_t start = 0;
    while (start < buffer_size && (buffer[start] == ' ' || buffer[start] == '\n' || buffer[start] == '\r' || buffer[start] == '\t')) {
        start++;
    }

    if (start < buffer_size && buffer[start] == '{') {
        // JSON format
        try {
            std::string json_str(reinterpret_cast<const char*>(buffer), buffer_size);
            json j = json::parse(json_str);
            return json_to_forest(j, handle);
        } catch (const json::parse_error& e) {
            set_error(std::string("JSON parse error: ") + e.what());
            return AORSF_ERROR_FORMAT;
        } catch (const std::exception& e) {
            set_error(std::string("JSON load error: ") + e.what());
            return AORSF_ERROR_FORMAT;
        }
    }

    // Binary format
    const unsigned char* ptr = buffer;
    const unsigned char* end = buffer + buffer_size;

    // Read and validate header
    aorsf_file_header_t header;
    if (!read_value(ptr, end, header)) {
        set_error("Buffer too small for header");
        return AORSF_ERROR_FORMAT;
    }

    if (std::memcmp(header.magic, AORSF_MAGIC, 4) != 0) {
        set_error("Invalid file format (bad magic bytes)");
        return AORSF_ERROR_FORMAT;
    }

    if (header.version_major > AORSF_FORMAT_VERSION_MAJOR) {
        set_error("File format version too new");
        return AORSF_ERROR_FORMAT;
    }

    // Create new handle
    aorsf_forest_t* new_handle = new (std::nothrow) aorsf_forest_t();
    if (!new_handle) {
        set_error("Out of memory");
        return AORSF_ERROR_OUT_OF_MEMORY;
    }

    new_handle->n_features = static_cast<int32_t>(header.n_features);
    new_handle->n_class = static_cast<int32_t>(header.n_class);

    // Read config section
    struct {
        int32_t tree_type;
        int32_t n_tree;
        int32_t mtry;
        int32_t leaf_min_obs;
        int32_t split_min_obs;
        int32_t split_rule;
        int32_t lincomb_type;
        double lincomb_eps;
        int32_t lincomb_iter_max;
        double lincomb_alpha;
        int32_t vi_type;
        uint32_t seed;
        double oob_error;
        int32_t padding[11];
    } config_data;

    if (!read_array(ptr, end, reinterpret_cast<unsigned char*>(&config_data), 96)) {
        delete new_handle;
        set_error("Failed to read config");
        return AORSF_ERROR_FORMAT;
    }

    new_handle->config.tree_type = static_cast<aorsf_tree_type_t>(config_data.tree_type);
    new_handle->config.n_tree = config_data.n_tree;
    new_handle->config.mtry = config_data.mtry;
    new_handle->config.leaf_min_obs = config_data.leaf_min_obs;
    new_handle->config.split_min_obs = config_data.split_min_obs;
    new_handle->config.split_rule = static_cast<aorsf_split_rule_t>(config_data.split_rule);
    new_handle->config.lincomb_type = static_cast<aorsf_lincomb_type_t>(config_data.lincomb_type);
    new_handle->config.lincomb_eps = config_data.lincomb_eps;
    new_handle->config.lincomb_iter_max = config_data.lincomb_iter_max;
    new_handle->config.lincomb_alpha = config_data.lincomb_alpha;
    new_handle->config.vi_type = static_cast<aorsf_vi_type_t>(config_data.vi_type);
    new_handle->config.seed = config_data.seed;
    new_handle->oob_error = config_data.oob_error;

    // Read per-tree data
    new_handle->cutpoints.resize(header.n_trees);
    new_handle->child_left.resize(header.n_trees);
    new_handle->leaf_summary.resize(header.n_trees);
    new_handle->coef_values.resize(header.n_trees);
    new_handle->coef_indices.resize(header.n_trees);

    if (header.tree_type == AORSF_TREE_CLASSIFICATION) {
        new_handle->leaf_pred_prob.resize(header.n_trees);
    } else if (header.tree_type == AORSF_TREE_SURVIVAL) {
        new_handle->leaf_pred_indx.resize(header.n_trees);
        new_handle->leaf_pred_prob.resize(header.n_trees);
        new_handle->leaf_pred_chaz.resize(header.n_trees);
    }

    for (uint32_t t = 0; t < header.n_trees; t++) {
        uint32_t n_nodes;
        if (!read_value(ptr, end, n_nodes)) {
            delete new_handle;
            set_error("Failed to read n_nodes");
            return AORSF_ERROR_FORMAT;
        }

        // Cutpoints
        new_handle->cutpoints[t].resize(n_nodes);
        if (!read_array(ptr, end, new_handle->cutpoints[t].data(), n_nodes)) {
            delete new_handle;
            set_error("Failed to read cutpoints");
            return AORSF_ERROR_FORMAT;
        }

        // Child left
        new_handle->child_left[t].resize(n_nodes);
        for (uint32_t i = 0; i < n_nodes; i++) {
            uint64_t val;
            if (!read_value(ptr, end, val)) {
                delete new_handle;
                set_error("Failed to read child_left");
                return AORSF_ERROR_FORMAT;
            }
            new_handle->child_left[t][i] = static_cast<arma::uword>(val);
        }

        // Leaf summary
        new_handle->leaf_summary[t].resize(n_nodes);
        if (!read_array(ptr, end, new_handle->leaf_summary[t].data(), n_nodes)) {
            delete new_handle;
            set_error("Failed to read leaf_summary");
            return AORSF_ERROR_FORMAT;
        }

        // Coefficients per node
        new_handle->coef_values[t].resize(n_nodes);
        new_handle->coef_indices[t].resize(n_nodes);
        for (uint32_t i = 0; i < n_nodes; i++) {
            uint32_t n_coef;
            if (!read_value(ptr, end, n_coef)) {
                delete new_handle;
                set_error("Failed to read n_coef");
                return AORSF_ERROR_FORMAT;
            }

            if (n_coef > 0) {
                new_handle->coef_indices[t][i].set_size(n_coef);
                new_handle->coef_values[t][i].set_size(n_coef);

                // Indices
                for (uint32_t j = 0; j < n_coef; j++) {
                    uint64_t idx;
                    if (!read_value(ptr, end, idx)) {
                        delete new_handle;
                        return AORSF_ERROR_FORMAT;
                    }
                    new_handle->coef_indices[t][i](j) = static_cast<arma::uword>(idx);
                }

                // Values
                if (!read_array(ptr, end, new_handle->coef_values[t][i].memptr(), n_coef)) {
                    delete new_handle;
                    return AORSF_ERROR_FORMAT;
                }
            }
        }

        // Type-specific leaf data
        if (header.tree_type == AORSF_TREE_CLASSIFICATION) {
            new_handle->leaf_pred_prob[t].resize(n_nodes);
            for (uint32_t i = 0; i < n_nodes; i++) {
                uint32_t n_probs;
                if (!read_value(ptr, end, n_probs)) {
                    delete new_handle;
                    return AORSF_ERROR_FORMAT;
                }
                if (n_probs > 0) {
                    new_handle->leaf_pred_prob[t][i].set_size(n_probs);
                    if (!read_array(ptr, end, new_handle->leaf_pred_prob[t][i].memptr(), n_probs)) {
                        delete new_handle;
                        return AORSF_ERROR_FORMAT;
                    }
                }
            }
        } else if (header.tree_type == AORSF_TREE_SURVIVAL) {
            new_handle->leaf_pred_indx[t].resize(n_nodes);
            new_handle->leaf_pred_prob[t].resize(n_nodes);
            new_handle->leaf_pred_chaz[t].resize(n_nodes);
            for (uint32_t i = 0; i < n_nodes; i++) {
                uint32_t n_times;
                if (!read_value(ptr, end, n_times)) {
                    delete new_handle;
                    return AORSF_ERROR_FORMAT;
                }
                if (n_times > 0) {
                    new_handle->leaf_pred_indx[t][i].set_size(n_times);
                    new_handle->leaf_pred_prob[t][i].set_size(n_times);
                    new_handle->leaf_pred_chaz[t][i].set_size(n_times);

                    if (!read_array(ptr, end, new_handle->leaf_pred_indx[t][i].memptr(), n_times)) {
                        delete new_handle;
                        return AORSF_ERROR_FORMAT;
                    }
                    if (!read_array(ptr, end, new_handle->leaf_pred_prob[t][i].memptr(), n_times)) {
                        delete new_handle;
                        return AORSF_ERROR_FORMAT;
                    }
                    if (!read_array(ptr, end, new_handle->leaf_pred_chaz[t][i].memptr(), n_times)) {
                        delete new_handle;
                        return AORSF_ERROR_FORMAT;
                    }
                }
            }
        }
    }

    // Read importance if present
    if (header.flags & AORSF_FLAG_HAS_IMPORTANCE) {
        new_handle->importance.resize(header.n_features);
        if (!read_array(ptr, end, new_handle->importance.data(), header.n_features)) {
            delete new_handle;
            set_error("Failed to read importance");
            return AORSF_ERROR_FORMAT;
        }
    }

    // Read unique times for survival
    if (header.tree_type == AORSF_TREE_SURVIVAL) {
        uint32_t n_times;
        if (!read_value(ptr, end, n_times)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
        new_handle->unique_times.resize(n_times);
        if (!read_array(ptr, end, new_handle->unique_times.data(), n_times)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
    }

    // Read OOB data if present
    if (header.flags & AORSF_FLAG_HAS_OOB) {
        uint32_t n_trees_oob;
        if (!read_value(ptr, end, n_trees_oob)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
        new_handle->rows_oobag.resize(n_trees_oob);
        for (uint32_t t = 0; t < n_trees_oob; t++) {
            uint32_t n_rows;
            if (!read_value(ptr, end, n_rows)) {
                delete new_handle;
                return AORSF_ERROR_FORMAT;
            }
            new_handle->rows_oobag[t].set_size(n_rows);
            for (uint32_t i = 0; i < n_rows; i++) {
                uint64_t idx;
                if (!read_value(ptr, end, idx)) {
                    delete new_handle;
                    return AORSF_ERROR_FORMAT;
                }
                new_handle->rows_oobag[t](i) = static_cast<arma::uword>(idx);
            }
        }
        uint32_t n_obs;
        if (!read_value(ptr, end, n_obs)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
        new_handle->oobag_denom.resize(n_obs);
        if (!read_array(ptr, end, new_handle->oobag_denom.data(), n_obs)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
    }

    // Read metadata if present
    if (header.flags & AORSF_FLAG_HAS_METADATA) {
        // Feature names
        uint32_t has_names;
        if (!read_value(ptr, end, has_names)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
        if (has_names) {
            uint32_t n_names;
            if (!read_value(ptr, end, n_names)) {
                delete new_handle;
                return AORSF_ERROR_FORMAT;
            }
            new_handle->feature_names.resize(n_names);
            for (uint32_t i = 0; i < n_names; i++) {
                uint32_t len;
                if (!read_value(ptr, end, len)) {
                    delete new_handle;
                    return AORSF_ERROR_FORMAT;
                }
                if (len > 0) {
                    new_handle->feature_names[i].resize(len);
                    if (!read_array(ptr, end, reinterpret_cast<unsigned char*>(&new_handle->feature_names[i][0]), len)) {
                        delete new_handle;
                        return AORSF_ERROR_FORMAT;
                    }
                }
            }
        }

        // Feature means
        uint32_t has_means;
        if (!read_value(ptr, end, has_means)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
        if (has_means) {
            uint32_t n_means;
            if (!read_value(ptr, end, n_means)) {
                delete new_handle;
                return AORSF_ERROR_FORMAT;
            }
            new_handle->feature_means.resize(n_means);
            if (!read_array(ptr, end, new_handle->feature_means.data(), n_means)) {
                delete new_handle;
                return AORSF_ERROR_FORMAT;
            }
        }

        // Feature stds
        uint32_t has_stds;
        if (!read_value(ptr, end, has_stds)) {
            delete new_handle;
            return AORSF_ERROR_FORMAT;
        }
        if (has_stds) {
            uint32_t n_stds;
            if (!read_value(ptr, end, n_stds)) {
                delete new_handle;
                return AORSF_ERROR_FORMAT;
            }
            new_handle->feature_stds.resize(n_stds);
            if (!read_array(ptr, end, new_handle->feature_stds.data(), n_stds)) {
                delete new_handle;
                return AORSF_ERROR_FORMAT;
            }
        }
    }

    new_handle->is_fitted = true;
    *handle = new_handle;
    return AORSF_SUCCESS;
}

AORSF_C_API aorsf_error_t aorsf_forest_save_file(
    aorsf_forest_handle handle,
    const char* filepath,
    aorsf_format_t format,
    uint32_t flags
) {
    if (!handle || !filepath) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    // Get required size
    size_t size;
    aorsf_error_t err = aorsf_forest_get_save_size(handle, format, flags, &size);
    if (err != AORSF_SUCCESS) return err;

    // Allocate buffer
    std::vector<unsigned char> buffer(size);

    // Serialize
    size_t written;
    err = aorsf_forest_save(handle, format, flags, buffer.data(), size, &written);
    if (err != AORSF_SUCCESS) return err;

    // Write to file
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        set_error("Failed to open file for writing");
        return AORSF_ERROR_IO;
    }

    file.write(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(written));
    if (!file) {
        set_error("Failed to write to file");
        return AORSF_ERROR_IO;
    }

    return AORSF_SUCCESS;
}

AORSF_C_API aorsf_error_t aorsf_forest_load_file(
    aorsf_forest_handle* handle,
    const char* filepath
) {
    if (!handle || !filepath) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    // Open file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        set_error("Failed to open file for reading");
        return AORSF_ERROR_IO;
    }

    // Get size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read entire file
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        set_error("Failed to read file");
        return AORSF_ERROR_IO;
    }

    // Parse
    return aorsf_forest_load(handle, buffer.data(), static_cast<size_t>(size));
}
